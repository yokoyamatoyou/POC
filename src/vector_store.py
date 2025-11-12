from __future__ import annotations

import copy
import hashlib
import logging
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

try:
    from src.metadata_separation import MetadataSeparator, MetadataPersistenceManager
except ImportError:  # pragma: no cover - optional dependency
    MetadataSeparator = None  # type: ignore
    MetadataPersistenceManager = None  # type: ignore


class VectorStoreManager:
    """
    シンプルなインメモリ実装。
    - add_document: 文章チャンクを登録し、ベクトル埋め込みとBM25コーパスを構築
    - delete_document: 登録解除（ベクトル・BM25インデックスから除外）
    - search: ベクトル＋BM25ハイブリッド検索（RRF再ランキング対応）
    - rerank_results: RRFを用いた再ランキング結果を返す
    """

    def __init__(
        self,
        client: OpenAI = None,
        *,
        docs_path: Optional[Path | str] = None,
        index_path: Optional[Path | str] = None,
        metadata_db_path: Optional[Path | str] = None,
        load_existing: bool = True,
    ) -> None:
        self.client = client
        self.documents: Dict[str, List[Dict[str, Any]]] = {}

        resolved_docs_path = Path(docs_path) if docs_path else Path("docs_store")
        resolved_docs_path.mkdir(parents=True, exist_ok=True)
        self.docs_path = resolved_docs_path

        resolved_index_path = Path(index_path) if index_path else Path("data/indexes")
        resolved_index_path.mkdir(parents=True, exist_ok=True)
        self.index_root = resolved_index_path
        
        # text-embedding-3-largeを使用したベクトル化
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dim: int = 3072
        self.embeddings: Dict[str, np.ndarray] = {}
        self.documents_info: Dict[str, Dict[str, Any]] = {}
        self.user_context: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self.audit_enabled = True
        self.audit_log_path = Path(os.getenv("ACL_AUDIT_LOG_PATH", "logs/access_audit.log"))
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # BM25（レキシカル検索）用のコーパス
        self.bm25_entries: List[Dict[str, Any]] = []
        self.bm25_model: Optional[BM25Okapi] = None

        self.metadata_separator = MetadataSeparator() if MetadataSeparator else None

        if MetadataPersistenceManager:
            metadata_db = Path(metadata_db_path) if metadata_db_path else Path("metadata.db")
            metadata_db.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_manager = MetadataPersistenceManager(str(metadata_db))
        else:
            self.metadata_manager = None

        # ハッシュの照合用マップ
        self.source_hash_index: Dict[str, str] = {}
        self.content_hash_index: Dict[str, str] = {}

        # 既存ドキュメントをロード
        if load_existing:
            self._load_existing_documents()

    def _doc_id_for_path(self, file_path: Path) -> str:
        h = hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:16]
        return h

    def _persist_source_file(self, file_path: Path, doc_dir: Path, source_hash: str) -> Optional[Path]:
        """docs_store配下に原本を保存し、保存先パスを返す"""
        if not file_path or not file_path.exists():
            return None

        source_dir = doc_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        destination = source_dir / f"{source_hash}{file_path.suffix}"

        try:
            if not destination.exists():
                shutil.copy2(file_path, destination)
            return destination
        except Exception as exc:
            self.logger.warning("Failed to persist source file: %s", exc)
            return None

    @staticmethod
    def _tokenize_for_bm25(text: str) -> List[str]:
        """BM25用の簡易トークナイズ（英数字+日本語2-gram）"""
        normalized = (text or "").lower()
        if not normalized:
            return []

        # 空白除去
        normalized = re.sub(r"\s+", "", normalized)
        if not normalized:
            return []

        tokens: List[str] = []

        # 英数字トークン
        word_tokens = re.findall(r"[a-z0-9]+", normalized)
        if word_tokens:
            tokens.extend(word_tokens)

        # 残りの文字列を2文字ずつ分割（日本語対応）
        remaining = re.sub(r"[a-z0-9]+", "", normalized)
        if remaining:
            if len(remaining) > 1:
                tokens.extend([remaining[i : i + 2] for i in range(len(remaining) - 1)])
            else:
                tokens.append(remaining)

        if not tokens:
            tokens.append(normalized)
        return tokens

    def _refresh_bm25_index(self) -> None:
        """BM25インデックスを再構築"""
        if not self.bm25_entries:
            self.bm25_model = None
            return

        corpus = [entry["tokens"] for entry in self.bm25_entries]
        try:
            self.bm25_model = BM25Okapi(corpus)
        except Exception as exc:
            self.logger.warning(f"BM25インデックス再構築に失敗しました: {exc}")
            self.bm25_model = None

    @staticmethod
    def _build_summary(text: str, metadata: Dict[str, Any], max_length: int = 180) -> str:
        summaries: List[str] = []

        existing_summary = metadata.get("summary")
        if isinstance(existing_summary, str) and existing_summary.strip():
            summaries.append(existing_summary.strip())

        stage4 = metadata.get("stage4_search") or {}
        stage4_summary = stage4.get("summary") or stage4.get("search_summary")
        if isinstance(stage4_summary, str) and stage4_summary.strip():
            summaries.append(stage4_summary.strip())

        stage2 = metadata.get("stage2_processing") or {}
        stage2_summary = stage2.get("summary") or stage2.get("processing_summary")
        if isinstance(stage2_summary, str) and stage2_summary.strip():
            summaries.append(stage2_summary.strip())

        image_summary = stage2.get("image_summary")
        if isinstance(image_summary, str) and image_summary.strip():
            summaries.append(image_summary.strip())

        if summaries:
            summary = summaries[0]
        else:
            snippet = re.sub(r"\s+", " ", (text or "")).strip()
            if len(snippet) > max_length:
                snippet = snippet[:max_length].rstrip() + "…"
            summary = snippet

        return summary

    def _augment_chunk_text(self, base_text: str, metadata: Dict[str, Any]) -> str:
        """チャンク本文に参照・コンテキスト情報を付加して検索品質を向上させる"""

        extras: List[str] = []

        stage4 = metadata.get("stage4_search") or {}

        reference_digest = metadata.get("reference_digest")
        if isinstance(reference_digest, list) and reference_digest:
            extras.append("参照要約:\n" + "\n".join(str(item) for item in reference_digest if str(item).strip()))
        stage4_reference_digest = stage4.get("reference_digest")
        if isinstance(stage4_reference_digest, list) and stage4_reference_digest:
            extras.append("参照要約(Stage4):\n" + "\n".join(str(item) for item in stage4_reference_digest if str(item).strip()))

        selected_excerpt = metadata.get("context_selected_excerpt") or stage4.get("context_selected_excerpt")
        if isinstance(selected_excerpt, str) and selected_excerpt.strip():
            extras.append("厳選コンテキスト:\n" + selected_excerpt.strip())

        context_excerpt = metadata.get("context_excerpt") or stage4.get("context_excerpt")
        if isinstance(context_excerpt, str) and context_excerpt.strip():
            extras.append("周辺セル:\n" + context_excerpt.strip())

        selected_cells = metadata.get("context_selected_cells") or stage4.get("context_selected_cells")
        if isinstance(selected_cells, list) and selected_cells:
            cell_lines = []
            for cell in selected_cells:
                if not isinstance(cell, dict):
                    continue
                label = cell.get("label")
                text_value = cell.get("text")
                if label and text_value:
                    cell_lines.append(f"{label}: {text_value}")
            if cell_lines:
                extras.append("選択セル:\n" + "\n".join(cell_lines))

        captions = metadata.get("captions")
        if isinstance(captions, list) and captions:
            extras.append("キャプション:\n" + "\n".join(str(cap) for cap in captions if str(cap).strip()))

        if not extras:
            return base_text

        base = base_text.strip()
        if base:
            return base + "\n\n" + "\n\n".join(extras)
        return "\n\n".join(extras)

    def _prepare_chunk(self, chunk: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
        """チャンク情報を正規化"""
        prepared = copy.deepcopy(chunk)
        prepared.setdefault("content", prepared.get("text", ""))
        prepared.setdefault("file_name", prepared.get("metadata", {}).get("file_name"))
        prepared.setdefault("file_type", prepared.get("metadata", {}).get("file_type"))

        metadata = copy.deepcopy(prepared.get("metadata") or {})
        metadata.setdefault("doc_id", doc_id)
        metadata.setdefault("file_name", prepared.get("file_name"))
        metadata.setdefault("file_type", prepared.get("file_type"))
        metadata.setdefault("stage1_basic", metadata.get("stage1_basic", {}))
        metadata.setdefault("stage2_processing", metadata.get("stage2_processing", {}))
        metadata.setdefault("stage3_business", metadata.get("stage3_business", {}))
        metadata.setdefault("stage4_search", metadata.get("stage4_search", {}))

        if prepared.get("file_type") == "image_caption":
            stage4 = metadata.get("stage4_search") or {}
            keywords = stage4.get("keywords") or []
            if isinstance(keywords, list):
                metadata.setdefault("tags", [])
                if isinstance(metadata["tags"], list):
                    for kw in keywords:
                        if kw not in metadata["tags"]:
                            metadata["tags"].append(kw)

        augmented_content = self._augment_chunk_text(prepared.get("content", ""), metadata)
        prepared["content"] = augmented_content
        prepared["metadata"] = metadata
        return prepared

    def _embed_text(self, content: str) -> np.ndarray:
        """テキストを埋め込みベクトルに変換（正規化済み）"""
        if not content or not self.client:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=content,
            )
            embedding = np.asarray(response.data[0].embedding, dtype=np.float32)
            if embedding.size:
                self.embedding_dim = embedding.shape[0]
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            return embedding
        except Exception as exc:
            self.logger.warning("Failed to embed text: %s", exc)
            return np.zeros(self.embedding_dim, dtype=np.float32)

    @staticmethod
    def _build_bm25_source_text(chunk: Dict[str, Any]) -> str:
        content = chunk.get("content") or chunk.get("text") or ""
        metadata = chunk.get("metadata") or {}

        parts: List[str] = [content]

        title = metadata.get("title") or metadata.get("file_name") or metadata.get("source_file")
        if title:
            parts.append(str(title))

        tags: List[str] = []
        raw_tags = metadata.get("tags") or metadata.get("user_tags")
        if isinstance(raw_tags, (list, tuple, set)):
            tags.extend(map(str, raw_tags))
        elif isinstance(raw_tags, str) and raw_tags.strip():
            tags.extend([tag.strip() for tag in raw_tags.split(",") if tag.strip()])

        stage2 = metadata.get("stage2_processing") or {}
        stage3 = metadata.get("stage3_business") or {}
        department = stage3.get("department") or metadata.get("department")
        importance = stage3.get("importance_level") or metadata.get("importance_level")
        user_tags = stage3.get("user_tags")
        if isinstance(user_tags, (list, tuple, set)):
            tags.extend(map(str, user_tags))

        stage4 = metadata.get("stage4_search") or {}
        keywords = stage4.get("keywords")
        if isinstance(keywords, (list, tuple, set)):
            tags.extend(map(str, keywords))
        elif isinstance(keywords, str) and keywords.strip():
            tags.extend([kw.strip() for kw in keywords.split(",") if kw.strip()])

        search_hints = metadata.get("search_metadata") or stage4.get("search_metadata")
        if isinstance(search_hints, (list, tuple, set)):
            tags.extend(map(str, search_hints))
        elif isinstance(search_hints, str) and search_hints.strip():
            tags.extend([hint.strip() for hint in search_hints.split(",") if hint.strip()])

        top_keywords = metadata.get("keywords")
        if isinstance(top_keywords, (list, tuple, set)):
            tags.extend(map(str, top_keywords))
        elif isinstance(top_keywords, str) and top_keywords.strip():
            tags.extend([kw.strip() for kw in top_keywords.split(",") if kw.strip()])

        image_meta = metadata.get("image_metadata") or {}
        image_keywords = image_meta.get("keywords")
        if isinstance(image_keywords, (list, tuple, set)):
            tags.extend(map(str, image_keywords))
        detected_elements = image_meta.get("detected_elements")
        if isinstance(detected_elements, (list, tuple, set)):
            tags.extend(map(str, detected_elements))

        scene_type = metadata.get("scene_type") or stage2.get("scene_type")
        if scene_type:
            parts.append(str(scene_type))

        actions = metadata.get("actions") or stage2.get("actions")
        if isinstance(actions, (list, tuple, set)):
            tags.extend(map(str, actions))
            parts.extend(map(str, actions))

        objects = metadata.get("objects") or stage2.get("objects")
        if isinstance(objects, (list, tuple, set)):
            tags.extend(map(str, objects))
            parts.extend(map(str, objects))

        visible_text = metadata.get("visible_text") or stage2.get("visible_text")
        if isinstance(visible_text, (list, tuple, set)):
            parts.extend(map(str, visible_text))

        numbers = metadata.get("numbers") or stage2.get("numbers")
        if isinstance(numbers, (list, tuple, set)):
            tags.extend(map(str, numbers))

        environment = metadata.get("environment") or stage2.get("environment")
        if environment:
            parts.append(str(environment))

        text_content = metadata.get("text_content") or stage2.get("text_content")
        if text_content:
            parts.append(str(text_content))

        technical_details = metadata.get("technical_details") or stage2.get("technical_details")
        if technical_details:
            parts.append(str(technical_details))

        recommended_queries = metadata.get("recommended_queries") or stage4.get("recommended_queries")
        if isinstance(recommended_queries, (list, tuple, set)):
            tags.extend(map(str, recommended_queries))

        search_terms = metadata.get("search_terms")
        if isinstance(search_terms, (list, tuple, set)):
            tags.extend(map(str, search_terms))
        elif isinstance(search_terms, str) and search_terms.strip():
            tags.extend([term.strip() for term in search_terms.split(",") if term.strip()])

        reference_digest = metadata.get("reference_digest")
        if isinstance(reference_digest, list) and reference_digest:
            parts.extend(str(item) for item in reference_digest if str(item).strip())
        stage4_reference_digest = stage4.get("reference_digest")
        if isinstance(stage4_reference_digest, list) and stage4_reference_digest:
            parts.extend(str(item) for item in stage4_reference_digest if str(item).strip())

        context_selected_excerpt = metadata.get("context_selected_excerpt") or stage4.get("context_selected_excerpt")
        if isinstance(context_selected_excerpt, str) and context_selected_excerpt.strip():
            parts.append(context_selected_excerpt.strip())

        context_excerpt = metadata.get("context_excerpt") or stage4.get("context_excerpt")
        if isinstance(context_excerpt, str) and context_excerpt.strip():
            parts.append(context_excerpt.strip())

        selected_cells = metadata.get("context_selected_cells") or stage4.get("context_selected_cells")
        if isinstance(selected_cells, list) and selected_cells:
            parts.extend(
                f"{cell.get('label')}: {cell.get('text')}"
                for cell in selected_cells
                if isinstance(cell, dict) and cell.get("label") and cell.get("text")
            )

        if department:
            parts.append(str(department))
        if importance:
            parts.append(str(importance))
        if tags:
            parts.append(" ".join(tags))

        return " ".join(part for part in parts if part)

    def _initialize_stage_metadata(
        self,
        *,
        doc_id: str,
        file_name: str,
        file_type: str,
        chunk_count: int,
        file_size: Optional[int],
        stored_path: Optional[Path],
        existing_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        now_iso = datetime.now().isoformat()

        stage1_existing = existing_metadata.get("stage1_basic") or {}
        stage1_defaults = {
            "doc_id": doc_id,
            "file_name": file_name,
            "file_type": file_type,
            "chunk_count": chunk_count,
            "file_size": file_size,
            "stored_path": str(stored_path) if stored_path else stage1_existing.get("stored_path"),
            "last_indexed_at": now_iso,
        }
        stage1_metadata = {**stage1_defaults, **stage1_existing}

        stage2_metadata = existing_metadata.get("stage2_processing") or {
            "processing_quality": existing_metadata.get("processing_quality", "unknown"),
            "chunk_count": chunk_count,
            "last_processed_at": now_iso,
        }

        stage3_metadata = existing_metadata.get("stage3_business") or {
            "department": "未設定",
            "importance_level": "中",
            "access_level": "一般",
            "user_tags": []
        }

        stage4_metadata = existing_metadata.get("stage4_search") or {
            "keywords": [],
            "category": "未分類",
            "search_priority": 1.0,
            "last_accessed": None,
        }

        return {
            "stage1_basic": stage1_metadata,
            "stage2_processing": stage2_metadata,
            "stage3_business": stage3_metadata,
            "stage4_search": stage4_metadata,
            "raw": existing_metadata,
        }

    def _build_document_info(
        self,
        *,
        doc_id: str,
        chunks: List[Dict[str, Any]],
        original_path: Optional[Path],
        stored_path: Optional[Path],
    ) -> Dict[str, Any]:
        first_chunk = chunks[0] if chunks else {}
        chunk_metadata = copy.deepcopy(first_chunk.get("metadata") or {})

        file_name = first_chunk.get("file_name")
        if not file_name and original_path is not None:
            file_name = original_path.name
        if not file_name:
            file_name = f"{doc_id}.txt"

        file_type = first_chunk.get("file_type") or chunk_metadata.get("file_type")
        if not file_type and original_path is not None:
            file_type = original_path.suffix.lstrip(".") or "text"
        file_type = file_type or "text"

        file_size = None
        resolved_path = stored_path or original_path
        if resolved_path and resolved_path.exists():
            try:
                file_size = resolved_path.stat().st_size
            except OSError:
                file_size = None

        stage_metadata = self._initialize_stage_metadata(
            doc_id=doc_id,
            file_name=file_name,
            file_type=file_type,
            chunk_count=len(chunks),
            file_size=file_size,
            stored_path=resolved_path,
            existing_metadata=chunk_metadata,
        )

        return {
            "doc_id": doc_id,
            "file_name": file_name,
            "file_type": file_type,
            "file_size": file_size,
            "created_at": chunk_metadata.get("created_at"),
            "updated_at": chunk_metadata.get("updated_at"),
            "stored_path": str(resolved_path) if resolved_path else None,
            "chunks": chunks,
            "metadata": stage_metadata,
        }

    def _clone_document_info(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(doc)

    @staticmethod
    def _sanitize_tenant_id(tenant_id: Optional[str]) -> str:
        value = str(tenant_id or "default").strip()
        if not value:
            value = "default"
        sanitized = re.sub(r"[^0-9A-Za-z_-]", "_", value)
        return sanitized or "default"

    def _ensure_doc_directory(self, tenant_segment: str, doc_id: str) -> Path:
        doc_dir = self.docs_path / f"tenant_{tenant_segment}" / f"doc_{doc_id}"
        (doc_dir / "processed").mkdir(parents=True, exist_ok=True)
        (doc_dir / "processed" / "metadata").mkdir(exist_ok=True)
        (doc_dir / "processed" / "thumbnails").mkdir(exist_ok=True)
        return doc_dir

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        hasher = hashlib.sha256()
        try:
            with file_path.open("rb") as file_obj:
                for chunk in iter(lambda: file_obj.read(8192), b""):
                    hasher.update(chunk)
        except Exception:
            return ""
        return hasher.hexdigest()

    @staticmethod
    def _compute_content_hash(chunks: Sequence[Dict[str, Any]]) -> str:
        hasher = hashlib.sha256()
        for chunk in chunks:
            text = (chunk.get("content") or chunk.get("text") or "").strip()
            if text:
                hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def _ensure_chunk_identity(self, doc_id: str, index: int, chunk: Dict[str, Any]) -> Tuple[str, str]:
        metadata = chunk.setdefault("metadata", {})
        chunk_id = metadata.get("chunk_id") or chunk.get("chunk_id")
        if not chunk_id:
            chunk_id = f"{doc_id}_chunk_{index:04d}"
        metadata["chunk_id"] = chunk_id
        chunk["chunk_id"] = chunk_id

        content = (chunk.get("content") or chunk.get("text") or "").strip()
        chunk_hash = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else hashlib.sha256(f"{doc_id}_{index}".encode("utf-8")).hexdigest()
        metadata["chunk_hash"] = chunk_hash
        metadata.setdefault("doc_id", doc_id)
        return chunk_id, chunk_hash

    @staticmethod
    def _json_default(obj: Any) -> Any:  # pragma: no cover - JSON変換補助
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    def _persist_chunks_file(self, doc_dir: Path, chunks: Sequence[Dict[str, Any]]) -> Path:
        processed_dir = doc_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = processed_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as file_obj:
            for record in chunks:
                json.dump(record, file_obj, ensure_ascii=False, default=self._json_default)
                file_obj.write("\n")
        return chunks_path

    def _persist_manifest(self, doc_dir: Path, manifest: Dict[str, Any]) -> Path:
        manifest_path = doc_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file_obj:
            json.dump(manifest, file_obj, ensure_ascii=False, indent=2, default=self._json_default)
        return manifest_path

    def _persist_indexes(
        self,
        tenant_segment: str,
        doc_id: str,
        embeddings: Optional[np.ndarray],
        bm25_tokens: Sequence[Sequence[str]],
    ) -> Dict[str, Any]:
        indexes_info: Dict[str, Any] = {}
        index_dir = self.index_root / f"tenant_{tenant_segment}" / f"doc_{doc_id}"
        index_dir.mkdir(parents=True, exist_ok=True)

        if embeddings is not None and embeddings.size:
            embeddings_path = index_dir / "embeddings.npy"
            np.save(embeddings_path, embeddings)
            indexes_info["vector_path"] = embeddings_path.as_posix()

        if bm25_tokens:
            tokens_path = index_dir / "bm25_tokens.json"
            with tokens_path.open("w", encoding="utf-8") as file_obj:
                json.dump(list(bm25_tokens), file_obj, ensure_ascii=False)
            indexes_info["bm25_path"] = tokens_path.as_posix()

        indexes_info["last_updated_at"] = datetime.now().isoformat()
        return indexes_info

    def _persist_chunk_metadata_record(self, doc_id: str, chunk: Dict[str, Any]) -> None:
        if not (self.metadata_manager and self.metadata_separator):
            return

        metadata = chunk.get("metadata") or {}
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            return

        try:
            separated = self.metadata_separator.separate_metadata(metadata, doc_id, chunk_id)
            self.metadata_manager.save_metadata(separated)
        except Exception as exc:  # pragma: no cover - 永続化失敗時はログのみ
            self.logger.warning("Failed to persist separated metadata: %s", exc)

    @staticmethod
    def _extract_tenant_id_from_chunks(chunks: Sequence[Dict[str, Any]]) -> str:
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            tenant_id = metadata.get("tenant_id")
            if tenant_id:
                return str(tenant_id)
            external = metadata.get("external") or {}
            if isinstance(external, dict):
                access_acl = external.get("access_acl") or {}
                if isinstance(access_acl, dict):
                    tenant_id = access_acl.get("tenant_id")
                    if tenant_id:
                        return str(tenant_id)
        return "default"

    def _build_manifest(
        self,
        *,
        tenant_id: str,
        tenant_segment: str,
        doc_id: str,
        doc_info: Dict[str, Any],
        source_hash: str,
        content_hash: str,
        stored_path: Optional[Path],
        chunk_hashes: Sequence[str],
        indexes_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata = copy.deepcopy(doc_info.get("metadata") or {})
        now_iso = datetime.now().isoformat()

        manifest: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "tenant_segment": tenant_segment,
            "doc_id": doc_id,
            "file_name": doc_info.get("file_name"),
            "file_type": doc_info.get("file_type"),
            "chunk_count": len(doc_info.get("chunks", [])),
            "version": 1,
            "source_hash": source_hash,
            "content_hash": content_hash,
            "chunk_hashes": list(chunk_hashes),
            "stored_path": str(stored_path) if stored_path else None,
            "created_at": metadata.get("stage1_basic", {}).get("created_at") or now_iso,
            "updated_at": now_iso,
            "stage1_basic": metadata.get("stage1_basic", {}),
            "stage2_processing": metadata.get("stage2_processing", {}),
            "stage3_business": metadata.get("stage3_business", {}),
            "stage4_search": metadata.get("stage4_search", {}),
            "indexes": indexes_info,
        }

        return manifest

    @staticmethod
    def _locate_source_file(doc_dir: Path) -> Optional[Path]:
        source_dir = doc_dir / "source"
        if not source_dir.exists():
            return None
        for candidate in sorted(source_dir.iterdir()):
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _load_chunks_jsonl(chunks_path: Path) -> List[Dict[str, Any]]:
        if not chunks_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with chunks_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        return records

    def _find_doc_directory(self, doc_id: str) -> Optional[Path]:
        for tenant_dir in self.docs_path.glob("tenant_*"):
            candidate = tenant_dir / f"doc_{doc_id}"
            if candidate.exists():
                return candidate
        return None

    def _remove_index_directory(self, doc_id: str) -> None:
        for tenant_dir in self.index_root.glob("tenant_*"):
            candidate = tenant_dir / f"doc_{doc_id}"
            if candidate.exists():
                shutil.rmtree(candidate, ignore_errors=True)


    def add_document(self, file_path: Path, chunks: List[Dict[str, Any]], thumbnail_path: Optional[Path]) -> str:
        if not chunks:
            return "error"
        doc_id = self._doc_id_for_path(file_path)
        if doc_id in self.documents:
            return "duplicate"

        tenant_id = self._extract_tenant_id_from_chunks(chunks)
        tenant_segment = self._sanitize_tenant_id(tenant_id)

        source_hash = self._compute_file_hash(file_path)
        if source_hash and source_hash in self.source_hash_index:
            self.logger.info("Duplicate source detected: %s", file_path)
            return "duplicate"

        content_hash = self._compute_content_hash(chunks)
        if content_hash and content_hash in self.content_hash_index:
            self.logger.info("Duplicate content detected: %s", file_path)
            return "duplicate"

        doc_dir = self._ensure_doc_directory(tenant_segment, doc_id)
        stored_path = self._persist_source_file(file_path, doc_dir, source_hash or doc_id)
        stored_path_str = str(stored_path) if stored_path else None

        prepared_chunks: List[Dict[str, Any]] = []
        chunk_embeddings: List[np.ndarray] = []
        bm25_entries: List[Dict[str, Any]] = []
        bm25_tokens: List[List[str]] = []
        chunk_hashes: List[str] = []
        page_to_chunk_ids: Dict[int, List[str]] = {}

        for idx, chunk in enumerate(chunks):
            prepared_chunk = self._prepare_chunk(chunk, doc_id)
            chunk_id, chunk_hash = self._ensure_chunk_identity(doc_id, idx, prepared_chunk)
            chunk_hashes.append(chunk_hash)
            metadata = prepared_chunk.setdefault("metadata", {})
            metadata.setdefault("tenant_id", tenant_id)
            if stored_path_str and not metadata.get("stored_path"):
                metadata["stored_path"] = stored_path_str
            stage1_basic = metadata.setdefault("stage1_basic", {})
            if stored_path_str and not stage1_basic.get("stored_path"):
                stage1_basic["stored_path"] = stored_path_str

            page_index_raw = metadata.get("page_index")
            try:
                page_index = int(page_index_raw)
            except (TypeError, ValueError):
                page_index = None
            if page_index is not None:
                metadata["page_index"] = page_index
                metadata.setdefault("page_label", page_index + 1)
                page_to_chunk_ids.setdefault(page_index, []).append(chunk_id)

            embedding = self._embed_text(prepared_chunk.get("content", ""))
            chunk_embeddings.append(embedding)

            source_text = self._build_bm25_source_text(prepared_chunk)
            tokens = self._tokenize_for_bm25(source_text)
            bm25_entries.append({
                "doc_id": doc_id,
                "chunk_index": idx,
                "tokens": tokens,
            })
            bm25_tokens.append(tokens)

            prepared_chunks.append(prepared_chunk)
            self._persist_chunk_metadata_record(doc_id, prepared_chunk)

        if prepared_chunks:
            for prepared_chunk in prepared_chunks:
                metadata = prepared_chunk.get("metadata") or {}
                chunk_id = metadata.get("chunk_id")
                if not chunk_id:
                    continue

                reference_links = metadata.get("reference_links")
                if not isinstance(reference_links, list) or not reference_links:
                    continue

                stage3 = metadata.setdefault("stage3_business", {})
                stage4 = metadata.setdefault("stage4_search", {})
                stage3.setdefault("reference_links", reference_links)
                stage4.setdefault("reference_links", reference_links)

                context_window = metadata.get("context_window")
                if context_window and "context_window" not in stage4:
                    stage4["context_window"] = context_window
                context_cells = metadata.get("context_cells")
                if context_cells and "context_cells" not in stage4:
                    stage4["context_cells"] = context_cells
                selected_excerpt = metadata.get("context_selected_excerpt")
                if selected_excerpt and "context_selected_excerpt" not in stage4:
                    stage4["context_selected_excerpt"] = selected_excerpt
                selected_cells = metadata.get("context_selected_cells")
                if selected_cells and "context_selected_cells" not in stage4:
                    stage4["context_selected_cells"] = selected_cells
                selection_method = metadata.get("context_selection_method")
                if selection_method and "context_selection_method" not in stage4:
                    stage4["context_selection_method"] = selection_method
                selection_reason = metadata.get("context_selection_reason")
                if selection_reason and "context_selection_reason" not in stage4:
                    stage4["context_selection_reason"] = selection_reason

                related_chunks = metadata.setdefault("related_chunks", [])
                stage4_related = stage4.setdefault("related_chunks", [])
                stage3.setdefault("related_chunks", related_chunks)

                resolved_count = 0
                for entry in reference_links:
                    if not isinstance(entry, dict):
                        continue
                    entry_type = entry.get("type")
                    if entry_type != "page":
                        continue

                    page_index_raw = entry.get("page_index")
                    try:
                        page_index = int(page_index_raw)
                    except (TypeError, ValueError):
                        page_index = None

                    target_ids: List[str] = []
                    if page_index is not None:
                        for target_chunk_id in page_to_chunk_ids.get(page_index, []):
                            if target_chunk_id == chunk_id:
                                continue
                            target_ids.append(target_chunk_id)

                    if target_ids:
                        entry["target_chunks"] = target_ids[:3]
                        entry["resolved"] = True
                        entry["status"] = "resolved"
                        if page_index is not None:
                            entry["target_page_label"] = page_index + 1
                        resolved_count += 1
                    else:
                        entry.setdefault("target_chunks", [])
                        entry["resolved"] = False
                        entry["status"] = entry.get("status") or "pending"

                    for target_chunk_id in entry.get("target_chunks", []):
                        if target_chunk_id not in related_chunks:
                            related_chunks.append(target_chunk_id)
                        if target_chunk_id not in stage4_related:
                            stage4_related.append(target_chunk_id)

                total_refs = len(reference_links)
                metadata["ref_total_count"] = total_refs
                metadata["ref_resolved_count"] = resolved_count
                metadata["ref_unresolved_count"] = max(0, total_refs - resolved_count)

        self.documents[doc_id] = prepared_chunks

        embeddings_matrix: Optional[np.ndarray] = None
        if chunk_embeddings:
            embeddings_matrix = np.vstack(chunk_embeddings)
            self.embeddings[doc_id] = embeddings_matrix

        doc_info = self._build_document_info(
            doc_id=doc_id,
            chunks=prepared_chunks,
            original_path=file_path,
            stored_path=stored_path,
        )
        self.documents_info[doc_id] = doc_info

        if bm25_entries:
            self.bm25_entries.extend(bm25_entries)
            self._refresh_bm25_index()

        self._persist_chunks_file(doc_dir, prepared_chunks)
        indexes_info = self._persist_indexes(tenant_segment, doc_id, embeddings_matrix, bm25_tokens)
        manifest = self._build_manifest(
            tenant_id=tenant_id,
            tenant_segment=tenant_segment,
            doc_id=doc_id,
            doc_info=doc_info,
            source_hash=source_hash,
            content_hash=content_hash,
            stored_path=stored_path,
            chunk_hashes=chunk_hashes,
            indexes_info=indexes_info,
        )
        self._persist_manifest(doc_dir, manifest)

        if source_hash:
            self.source_hash_index[source_hash] = doc_id
        if content_hash:
            self.content_hash_index[content_hash] = doc_id

        return "success"

    def delete_document(self, doc_id: str) -> None:
        self.documents.pop(doc_id, None)
        self.embeddings.pop(doc_id, None)
        doc_info = self.documents_info.pop(doc_id, None)
        doc_dir = None
        if doc_info and doc_info.get("stored_path"):
            try:
                stored = Path(doc_info["stored_path"])
                if stored.exists():
                    doc_dir = stored.parent.parent if stored.parent.name == "source" else stored.parent
                    stored.unlink()
            except Exception:
                doc_dir = None

        if doc_dir is None:
            doc_dir = self._find_doc_directory(doc_id)

        if doc_dir and doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)

        self._remove_index_directory(doc_id)

        if self.bm25_entries:
            self.bm25_entries = [entry for entry in self.bm25_entries if entry["doc_id"] != doc_id]
            self._refresh_bm25_index()

        # ハッシュマップの更新
        if self.source_hash_index:
            self.source_hash_index = {k: v for k, v in self.source_hash_index.items() if v != doc_id}
        if self.content_hash_index:
            self.content_hash_index = {k: v for k, v in self.content_hash_index.items() if v != doc_id}

    def _load_existing_documents(self) -> None:
        if not self.docs_path.exists():
            return

        loaded_bm25: List[Dict[str, Any]] = []

        for tenant_dir in sorted(self.docs_path.glob("tenant_*")):
            for doc_dir in sorted(tenant_dir.glob("doc_*")):
                manifest_path = doc_dir / "manifest.json"
                if not manifest_path.exists():
                    continue

                try:
                    with manifest_path.open("r", encoding="utf-8") as file_obj:
                        manifest = json.load(file_obj)
                except Exception as exc:  # pragma: no cover - 起動ログ用
                    self.logger.warning("Failed to load manifest %s: %s", manifest_path, exc)
                    continue

                doc_id = manifest.get("doc_id") or doc_dir.name.replace("doc_", "")
                chunks_path = doc_dir / "processed" / "chunks.jsonl"
                chunk_records = self._load_chunks_jsonl(chunks_path)
                if not chunk_records:
                    continue

                normalized_chunks: List[Dict[str, Any]] = []
                for idx, record in enumerate(chunk_records):
                    metadata = record.setdefault("metadata", {})
                    metadata.setdefault("doc_id", doc_id)
                    record.setdefault("chunk_index", idx)
                    self._ensure_chunk_identity(doc_id, idx, record)
                    normalized_chunks.append(record)

                self.documents[doc_id] = normalized_chunks

                stored_path: Optional[Path] = None
                stored_path_str = manifest.get("stored_path")
                if stored_path_str:
                    stored_path = Path(stored_path_str)
                if stored_path is None or not stored_path.exists():
                    stored_path = self._locate_source_file(doc_dir)

                doc_info = self._build_document_info(
                    doc_id=doc_id,
                    chunks=normalized_chunks,
                    original_path=stored_path,
                    stored_path=stored_path,
                )

                metadata = doc_info.setdefault("metadata", {})
                for stage_key in ("stage1_basic", "stage2_processing", "stage3_business", "stage4_search"):
                    stage_value = manifest.get(stage_key)
                    if isinstance(stage_value, dict) and stage_value:
                        metadata[stage_key] = stage_value
                doc_info["metadata"] = metadata
                self.documents_info[doc_id] = doc_info

                source_hash = manifest.get("source_hash")
                if source_hash:
                    self.source_hash_index[source_hash] = doc_id
                content_hash = manifest.get("content_hash")
                if content_hash:
                    self.content_hash_index[content_hash] = doc_id

                indexes_info = manifest.get("indexes") or {}
                embeddings_path = indexes_info.get("vector_path")
                if embeddings_path:
                    embeddings_file = Path(embeddings_path)
                    if not embeddings_file.is_absolute():
                        embeddings_file = Path(embeddings_path)
                    if embeddings_file.exists():
                        try:
                            embeddings_matrix = np.load(embeddings_file)
                            if embeddings_matrix.ndim == 1:
                                embeddings_matrix = embeddings_matrix.reshape(1, -1)
                            self.embeddings[doc_id] = embeddings_matrix.astype(np.float32)
                            self.embedding_dim = embeddings_matrix.shape[1]
                        except Exception as exc:
                            self.logger.warning("Failed to load embeddings for %s: %s", doc_id, exc)

                tokens_path = indexes_info.get("bm25_path")
                bm25_tokens: List[List[str]] = []
                if tokens_path:
                    tokens_file = Path(tokens_path)
                    if not tokens_file.is_absolute():
                        tokens_file = Path(tokens_path)
                    if tokens_file.exists():
                        try:
                            with tokens_file.open("r", encoding="utf-8") as file_obj:
                                data = json.load(file_obj)
                                if isinstance(data, list):
                                    bm25_tokens = [list(item) if isinstance(item, (list, tuple, set)) else [str(item)] for item in data]
                        except Exception as exc:
                            self.logger.warning("Failed to load BM25 tokens for %s: %s", doc_id, exc)

                if bm25_tokens:
                    for idx, tokens in enumerate(bm25_tokens):
                        loaded_bm25.append({"doc_id": doc_id, "chunk_index": idx, "tokens": list(tokens)})
                else:
                    for idx, chunk_record in enumerate(normalized_chunks):
                        source_text = self._build_bm25_source_text(chunk_record)
                        tokens = self._tokenize_for_bm25(source_text)
                        loaded_bm25.append({"doc_id": doc_id, "chunk_index": idx, "tokens": tokens})

        if loaded_bm25:
            self.bm25_entries = loaded_bm25
            self._refresh_bm25_index()

    def set_user_context(
        self,
        tenant_id: str,
        departments: Sequence[str],
        role_level: int,
        role_label: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        deps = [str(dept).strip() for dept in departments if str(dept).strip()]
        self.user_context = {
            "tenant_id": tenant_id.strip() if tenant_id else "",
            "departments": deps,
            "role_level": int(role_level) if role_level else 0,
            "role_label": role_label or "",
            "user_id": user_id or "",
        }

    def _passes_acl(self, metadata: Dict[str, Any]) -> bool:
        if not self.user_context:
            return True

        tenant_id = self.user_context.get("tenant_id") or ""
        user_departments = set(self.user_context.get("departments") or [])
        user_role_level = int(self.user_context.get("role_level") or 0)

        doc_tenant = metadata.get("tenant_id") or metadata.get("external", {}).get("access_acl", {}).get("tenant_id")
        if tenant_id and doc_tenant and doc_tenant != tenant_id:
            self._record_audit_event(metadata, "DENY_TENANT_MISMATCH", "tenant mismatch")
            return False

        access_scope = metadata.get("access_scope")
        if not access_scope:
            access_scope = metadata.get("external", {}).get("access_acl", {}).get("access_scope")

        min_role_level = metadata.get("min_role_level")
        if min_role_level is None:
            min_role_level = metadata.get("external", {}).get("access_acl", {}).get("min_role_level")
        if min_role_level is None:
            min_role_level = 0

        if user_role_level and min_role_level and user_role_level < int(min_role_level):
            self._record_audit_event(metadata, "DENY_ROLE_LEVEL", "role level too low")
            return False

        if access_scope == "department":
            allowed_departments = metadata.get("allowed_departments")
            if allowed_departments is None:
                allowed_departments = metadata.get("external", {}).get("access_acl", {}).get("allowed_departments")
            allowed_departments = set((allowed_departments or []))
            if user_departments and allowed_departments:
                if user_departments.isdisjoint(allowed_departments):
                    self._record_audit_event(metadata, "DENY_DEPARTMENT", "department not allowed")
                    return False
            elif allowed_departments:
                self._record_audit_event(metadata, "DENY_DEPARTMENT", "no matching department")
                return False

        if access_scope == "confidential":
            allowed_users = metadata.get("allowed_users")
            if allowed_users is None:
                allowed_users = metadata.get("external", {}).get("access_acl", {}).get("allowed_users")
            allowed_users = set((allowed_users or []))
            user_id = self.user_context.get("user_id")
            if allowed_users and user_id not in allowed_users:
                self._record_audit_event(metadata, "DENY_USER", "user not allowed")
                return False

        self._record_audit_event(metadata, "ALLOW", "passed acl")
        return True

    def _record_audit_event(self, metadata: Dict[str, Any], decision: str, reason: str) -> None:
        if not self.audit_enabled:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": metadata.get("doc_id"),
            "tenant_id": metadata.get("tenant_id") or metadata.get("external", {}).get("access_acl", {}).get("tenant_id"),
            "access_scope": metadata.get("access_scope") or metadata.get("external", {}).get("access_acl", {}).get("access_scope"),
            "min_role_level": metadata.get("min_role_level") or metadata.get("external", {}).get("access_acl", {}).get("min_role_level"),
            "decision": decision,
            "reason": reason,
            "user": self.user_context,
        }

        try:
            with self.audit_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{json.dumps(event, ensure_ascii=False)}\n")
        except Exception as exc:
            self.logger.warning(f"ACL監査ログの書き込みに失敗しました: {exc}")

    def get_document_id(self, file_path: Path) -> str:
        return self._doc_id_for_path(file_path)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        doc = self.documents_info.get(doc_id)
        if doc is None and doc_id in self.documents:
            doc_info = self._build_document_info(
                doc_id=doc_id,
                chunks=self.documents[doc_id],
                original_path=None,
                stored_path=None,
            )
            self.documents_info[doc_id] = doc_info
            doc = doc_info
        return self._clone_document_info(doc) if doc else None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return [self._clone_document_info(doc) for doc in self.documents_info.values()]

    def get_statistics(self) -> Dict[str, Any]:
        """ドキュメント/チャンクの統計情報を返す"""

        document_count = len(self.documents_info)
        chunk_count = 0
        chunk_lengths: List[int] = []
        file_type_distribution: Dict[str, int] = {}
        last_indexed_candidates: List[datetime] = []

        for doc_id, doc in self.documents_info.items():
            file_type = doc.get("file_type") or "unknown"
            file_type_distribution[file_type] = file_type_distribution.get(file_type, 0) + 1

            stored_chunks = self.documents.get(doc_id) or doc.get("chunks", []) or []
            chunk_count += len(stored_chunks)

            for chunk in stored_chunks:
                chunk_text = (chunk.get("content") or chunk.get("text") or "")
                chunk_lengths.append(len(chunk_text))

            stage1_basic = ((doc.get("metadata") or {}).get("stage1_basic") or {})
            timestamp = stage1_basic.get("last_indexed_at")
            if timestamp:
                try:
                    last_indexed_candidates.append(datetime.fromisoformat(timestamp))
                except ValueError:
                    # 文字列として解釈できない場合はスキップ
                    pass

        average_chunk_length = 0.0
        if chunk_lengths:
            average_chunk_length = sum(chunk_lengths) / len(chunk_lengths)

        average_chunks_per_document = 0.0
        if document_count:
            average_chunks_per_document = chunk_count / document_count

        embedded_document_count = len(self.embeddings)
        documents_without_embeddings = max(document_count - embedded_document_count, 0)

        last_indexed_at = None
        if last_indexed_candidates:
            last_indexed_at = max(last_indexed_candidates).isoformat()

        return {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "average_chunk_length": average_chunk_length,
            "average_chunks_per_document": average_chunks_per_document,
            "embedded_document_count": embedded_document_count,
            "documents_without_embeddings": documents_without_embeddings,
            "file_type_distribution": file_type_distribution,
            "last_indexed_at": last_indexed_at,
        }

    def update_document_metadata(self, doc_id: str, stage: str, updated_metadata: Dict[str, Any]) -> bool:
        doc_info = self.documents_info.get(doc_id)
        if not doc_info:
            return False

        metadata = doc_info.setdefault("metadata", {})
        stage_key = stage if stage.startswith("stage") else f"stage3_{stage}"
        stage_metadata = metadata.setdefault(stage_key, {})
        stage_metadata.update(updated_metadata)

        # チャンク側のメタデータも更新
        for chunk in self.documents.get(doc_id, []):
            chunk_meta = chunk.setdefault("metadata", {})
            stage_meta = chunk_meta.setdefault(stage_key, {})
            stage_meta.update(updated_metadata)

        self.documents_info[doc_id] = doc_info
        return True

    def update_document_acl(self, doc_id: str, updated_acl: Dict[str, Any]) -> bool:
        doc_info = self.documents_info.get(doc_id)
        if not doc_info:
            return False

        metadata = doc_info.setdefault("metadata", {})

        tenant_id = updated_acl.get("tenant_id")
        department = updated_acl.get("department")
        access_scope = updated_acl.get("access_scope")
        min_role_level = updated_acl.get("min_role_level")
        min_role_label = updated_acl.get("min_role_label")
        allowed_departments = updated_acl.get("allowed_departments") or []
        allowed_users = updated_acl.get("allowed_users") or []

        external_meta = metadata.setdefault("external", {})
        external_acl = external_meta.setdefault("access_acl", {})
        external_acl.update(
            {
                "tenant_id": tenant_id,
                "department": department,
                "access_scope": access_scope,
                "min_role_level": min_role_level,
                "min_role_label": min_role_label,
                "allowed_departments": allowed_departments,
                "allowed_users": allowed_users,
            }
        )

        priority_meta = metadata.setdefault("priority", {})
        priority_acl = priority_meta.setdefault("access_acl", {})
        priority_acl.update(
            {
                "tenant_id": tenant_id,
                "access_scope": access_scope,
                "min_role_level": min_role_level,
            }
        )

        metadata["tenant_id"] = tenant_id
        metadata["department"] = department
        metadata["access_scope"] = access_scope
        metadata["min_role_level"] = min_role_level
        metadata["min_role_label"] = min_role_label
        metadata["allowed_departments"] = list(allowed_departments)
        metadata["allowed_users"] = list(allowed_users)

        # チャンク側にも反映
        for chunk in self.documents.get(doc_id, []):
            chunk_meta = chunk.setdefault("metadata", {})
            chunk_meta.setdefault("external", {}).setdefault("access_acl", {}).update(external_acl)
            chunk_meta.setdefault("priority", {}).setdefault("access_acl", {}).update(priority_acl)
            chunk_meta["tenant_id"] = tenant_id
            chunk_meta["department"] = department
            chunk_meta["access_scope"] = access_scope
            chunk_meta["min_role_level"] = min_role_level
            chunk_meta["min_role_label"] = min_role_label
            chunk_meta["allowed_departments"] = list(allowed_departments)
            chunk_meta["allowed_users"] = list(allowed_users)

        self.documents_info[doc_id] = doc_info
        return True

    def record_document_access(self, doc_id: str) -> None:
        doc_info = self.documents_info.get(doc_id)
        if not doc_info:
            return
        metadata = doc_info.setdefault("metadata", {})
        stage4 = metadata.setdefault("stage4_search", {})
        stage4["last_accessed"] = datetime.now().isoformat()
        self.documents_info[doc_id] = doc_info

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        strategy: str = "hybrid_rrf",
        w_vec: float = 0.6,
        w_bm25: float = 0.4,
        top_k: int = 10,
        *,
        search_mode: Optional[str] = None,
        use_adaptive: bool = False,
    ) -> List[Dict[str, Any]]:
        query_text = (query or "").strip()
        if not query_text:
            return []

        normalized_query = query_text.lower()
        query_words = [word for word in normalized_query.split() if word]
        mode = (search_mode or strategy or "hybrid_rrf").lower()

        results: List[Dict[str, Any]] = []
        accessed_doc_ids: Set[str] = set()

        query_embedding = None
        if mode in {"hybrid", "hybrid_rrf", "vector_only", "vector"}:
            embedding = self._embed_text(query_text)
            if embedding.size:
                query_embedding = embedding.astype(np.float32)

        bm25_lookup: Dict[Tuple[str, int], float] = {}
        if self.bm25_model is not None:
            query_tokens = self._tokenize_for_bm25(query_text)
            if query_tokens:
                try:
                    raw_scores = np.array(self.bm25_model.get_scores(query_tokens), dtype=np.float32)
                    if raw_scores.size and raw_scores.size == len(self.bm25_entries):
                        max_score = float(np.max(raw_scores))
                        if max_score > 0:
                            for idx_entry, entry in enumerate(self.bm25_entries):
                                normalized_score = float(raw_scores[idx_entry] / max_score)
                                bm25_lookup[(entry["doc_id"], entry["chunk_index"])] = normalized_score
                        else:
                            bm25_lookup = {}
                except Exception as exc:
                    self.logger.warning(f"BM25スコア計算に失敗しました: {exc}")

        for doc_id, chunks in self.documents.items():
            doc_info = self.documents_info.get(doc_id)
            doc_stage_metadata = copy.deepcopy(doc_info.get("metadata", {})) if doc_info else {}
            chunk_embeddings = self.embeddings.get(doc_id)

            for idx, chunk in enumerate(chunks):
                metadata = copy.deepcopy(chunk.get("metadata") or {})

                if not self._passes_acl(metadata):
                    continue

                if doc_stage_metadata:
                    for stage_value in doc_stage_metadata.values():
                        if isinstance(stage_value, dict):
                            for key, value in stage_value.items():
                                metadata.setdefault(key, value)

                if doc_info:
                    metadata.setdefault("file_name", doc_info.get("file_name"))
                    metadata.setdefault("file_type", doc_info.get("file_type"))
                    metadata.setdefault("stored_path", doc_info.get("stored_path"))

                metadata.setdefault("doc_id", doc_id)

                if not self._passes_acl(metadata):
                    continue

                if not self._passes_filters(metadata, filters):
                    continue

                text = (chunk.get("content") or chunk.get("text") or "").strip()
                lexical_score = bm25_lookup.get((doc_id, idx))
                if lexical_score is None:
                    text_lower = text.lower()
                    lexical_score = self._calculate_lexical_score(query_words, normalized_query, text_lower)

                summary = self._build_summary(text, metadata)
                metadata.setdefault("summary", summary)

                vector_score = 0.0
                if (
                    query_embedding is not None
                    and chunk_embeddings is not None
                    and idx < len(chunk_embeddings)
                ):
                    vector_score = float(np.dot(query_embedding, chunk_embeddings[idx]))
                    vector_score = max(vector_score, 0.0)

                combined_score = self._combine_scores(
                    vector_score=vector_score,
                    lexical_score=lexical_score,
                    mode=mode,
                    w_vec=w_vec,
                    w_bm25=w_bm25,
                    use_adaptive=use_adaptive,
                )

                if combined_score <= 0.0:
                    continue

                accessed_doc_ids.add(doc_id)

                metadata.setdefault("bm25_score", lexical_score)

                results.append(
                    {
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "content": text,
                        "relevance_score": combined_score,
                        "vector_score": vector_score,
                        "lexical_score": lexical_score,
                        "metadata": metadata,
                    }
                )

        results.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)

        for doc_id in accessed_doc_ids:
            self.record_document_access(doc_id)

        return results[:top_k]

    def _calculate_lexical_score(
        self,
        query_words: List[str],
        query_text: str,
        text_lower: str,
    ) -> float:
        if not text_lower or not query_words:
            return 0.0

        if query_text in text_lower:
            return 1.0

        text_words = text_lower.split()
        exact_word_matches = sum(1 for word in query_words if word in text_words)
        if exact_word_matches:
            return min(1.0, 0.7 + (exact_word_matches / len(query_words)) * 0.3)

        partial_word_matches = sum(
            1 for word in query_words if any(word in candidate for candidate in text_words)
        )
        if partial_word_matches:
            return min(0.8, 0.4 + (partial_word_matches / len(query_words)) * 0.4)

        char_matches = sum(1 for char in query_text if char in text_lower)
        if query_text and char_matches > len(query_text) * 0.5:
            return 0.3

        return 0.0

    def _combine_scores(
        self,
        *,
        vector_score: float,
        lexical_score: float,
        mode: str,
        w_vec: float,
        w_bm25: float,
        use_adaptive: bool,
    ) -> float:
        if mode in {"vector_only", "vector"}:
            return vector_score
        if mode in {"lexical", "bm25"}:
            return lexical_score

        vector_weight = max(w_vec, 0.0)
        lexical_weight = max(w_bm25, 0.0)
        total_weight = vector_weight + lexical_weight
        if total_weight <= 0:
            vector_weight = 0.6
            lexical_weight = 0.4
            total_weight = 1.0
        vector_weight /= total_weight
        lexical_weight /= total_weight

        if use_adaptive:
            if lexical_score < 0.2 and vector_score > 0.05:
                vector_weight = min(0.85, max(vector_weight, 0.75))
                lexical_weight = 1.0 - vector_weight
            elif lexical_score > 0.7:
                lexical_weight = min(0.85, max(lexical_weight, 0.7))
                vector_weight = 1.0 - lexical_weight

        return vector_weight * vector_score + lexical_weight * lexical_score

    def rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        provider: str = "rrf",
        top_n: int = 30,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        capped = results[: max(len(results), top_n)]

        vector_sorted = sorted(
            enumerate(capped), key=lambda item: item[1].get("vector_score", 0.0), reverse=True
        )
        lexical_sorted = sorted(
            enumerate(capped), key=lambda item: item[1].get("lexical_score", 0.0), reverse=True
        )
        combined_sorted = sorted(
            enumerate(capped), key=lambda item: item[1].get("relevance_score", 0.0), reverse=True
        )

        rrf_scores: Dict[int, float] = defaultdict(float)
        constant = 60.0

        for rank, (idx, _) in enumerate(vector_sorted, start=1):
            rrf_scores[idx] += 1.0 / (constant + rank)
        for rank, (idx, _) in enumerate(lexical_sorted, start=1):
            rrf_scores[idx] += 1.0 / (constant + rank)
        for rank, (idx, _) in enumerate(combined_sorted, start=1):
            rrf_scores[idx] += 1.0 / (constant + rank)

        reranked: List[Dict[str, Any]] = []
        for idx, item in enumerate(capped):
            cloned = copy.deepcopy(item)
            metadata = cloned.setdefault("metadata", {})
            metadata["rerank_score"] = float(rrf_scores.get(idx, 0.0))
            reranked.append(cloned)

        reranked.sort(key=lambda item: item["metadata"].get("rerank_score", 0.0), reverse=True)
        return reranked[:top_n]

    @staticmethod
    def _normalize_to_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            if not value:
                return []
            return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, Sequence):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value)]

    @classmethod
    def _sensitivity_rank(cls, level: str) -> int:
        order = ["公開", "社内", "機密", "最高機密"]
        try:
            return order.index(level)
        except ValueError:
            return 0

    def _passes_filters(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True

        tenant_filter = str(filters.get("tenant_id", "")).strip()
        if tenant_filter:
            doc_tenant = str(metadata.get("tenant_id", "")).strip()
            if doc_tenant != tenant_filter:
                return False

        groups_filter = self._normalize_to_list(filters.get("allowed_groups"))
        if groups_filter:
            doc_groups = self._normalize_to_list(metadata.get("allowed_groups"))
            if not doc_groups:
                return False
            if not any(group in doc_groups for group in groups_filter):
                return False

        max_level = filters.get("max_sensitivity_level") or filters.get("sensitivity")
        if max_level:
            doc_level = metadata.get("sensitivity") or metadata.get("sensitivity_level") or "公開"
            if self._sensitivity_rank(str(doc_level)) > self._sensitivity_rank(str(max_level)):
                return False

        return True







