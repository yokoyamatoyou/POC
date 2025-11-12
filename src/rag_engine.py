from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import textwrap
import logging
from src.error_handler import error_logger, ErrorSeverity, ErrorCategory
from src.app.utils.summary_metrics import record_summary_event

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    最小実装のRAGエンジン。
    - VectorStoreManager を受け取り、簡易検索と回答合成を行う。
    """

    def __init__(self, vector_store, client) -> None:
        self.vector_store = vector_store
        self.client = client

    def answer_question(
        self,
        query: str,
        filters: Dict[str, Any] | None = None,
        strategy: str = "hybrid_rrf",
        w_vec: float = 0.6,
        w_bm25: float = 0.4,
        rerank: bool = True,
        rerank_provider: str = "cohere",
        rerank_top_n: int = 30,
        acl_filters: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        results = self.vector_store.search(
            query, filters=filters, strategy=strategy, w_vec=w_vec, w_bm25=w_bm25, top_k=10
        )
        if rerank:
            results = self.vector_store.rerank_results(results, query, provider=rerank_provider, top_n=rerank_top_n)

        if acl_filters:
            results = [r for r in results if self._is_accessible(r, acl_filters)]

        if not results:
            return {"answer": "関連する情報が見つかりませんでした。", "sources": []}

        # 最小実装: 上位の内容を単純連結
        top_texts = [r.get("content", "") for r in results[:3]]
        answer = "\n".join(top_texts)
        return {
            "answer": answer or "関連する情報が見つかりませんでした。",
            "sources": results[:3],
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        search_mode: str = "hybrid",
        use_adaptive: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        w_vec: float = 0.6,
        w_bm25: float = 0.4,
    ) -> List[Dict[str, Any]]:
        mode = self._map_search_mode(search_mode)
        normalized = str(search_mode or "").lower()
        adaptive = use_adaptive or normalized == "adaptive"

        results = self.vector_store.search(
            query,
            filters=filters,
            strategy=mode,
            w_vec=w_vec,
            w_bm25=w_bm25,
            top_k=top_k,
            search_mode=mode,
            use_adaptive=adaptive,
        )

        if results:
            results = self.vector_store.rerank_results(results, query, top_n=top_k)

        formatted_results: List[Dict[str, Any]] = []
        doc_chunk_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for result in results:
            metadata = result.get("metadata") or {}
            chunk_text = result.get("content", "")
            summary = metadata.get("summary") or (
                (metadata.get("stage4_search") or {}).get("summary")
            )

            if not summary and self.client:
                summary_payload = self._generate_result_summary(query, chunk_text, metadata)
                if summary_payload and summary_payload.get("summary"):
                    summary_text = summary_payload.get("summary", "").strip()
                    summary = summary_text
                    query_summaries = metadata.get("query_summaries")
                    if not isinstance(query_summaries, list):
                        query_summaries = []
                    metadata["query_summaries"] = query_summaries
                    query_summaries.append(
                        {
                            "query": query,
                            "summary": summary_text,
                            "keywords": summary_payload.get("keywords") or [],
                            "hallucination_notice": summary_payload.get("notice"),
                        }
                    )

            doc_id = result.get("doc_id")
            if doc_id and doc_id not in doc_chunk_cache:
                chunk_lookup: Dict[str, Dict[str, Any]] = {}
                for chunk_record in self.vector_store.documents.get(doc_id, []) or []:
                    chunk_meta = chunk_record.get("metadata") or {}
                    chunk_id = chunk_meta.get("chunk_id")
                    if chunk_id:
                        chunk_lookup[chunk_id] = chunk_record
                doc_chunk_cache[doc_id] = chunk_lookup

            reference_links = metadata.get("reference_links")
            if not isinstance(reference_links, list) or not reference_links:
                stage4_references = (metadata.get("stage4_search") or {}).get("reference_links")
                if isinstance(stage4_references, list) and stage4_references:
                    reference_links = metadata.setdefault("reference_links", stage4_references)

            if doc_id and isinstance(reference_links, list) and reference_links:
                chunk_lookup = doc_chunk_cache.get(doc_id, {})
                digest_lines: List[str] = []

                for entry in reference_links:
                    if not isinstance(entry, dict):
                        continue

                    target_chunk_ids = entry.get("target_chunks") or []
                    target_summaries: List[Dict[str, Any]] = []

                    for target_chunk_id in target_chunk_ids:
                        target_record = chunk_lookup.get(target_chunk_id)
                        if not target_record:
                            continue

                        target_meta = target_record.get("metadata") or {}
                        target_summary = (
                            target_meta.get("summary")
                            or (target_meta.get("stage4_search") or {}).get("summary")
                        )
                        if not target_summary:
                            target_text = target_record.get("content") or target_record.get("text") or ""
                            target_summary = self._fallback_summary(target_text)

                        page_label = target_meta.get("page_label") or target_meta.get("page_number")
                        target_summaries.append(
                            {
                                "chunk_id": target_chunk_id,
                                "summary": target_summary,
                                "page": page_label,
                            }
                        )

                    if target_summaries:
                        entry["target_chunk_summaries"] = target_summaries

                        label = entry.get("label")
                        if not label:
                            if entry.get("type") == "page" and entry.get("page_number"):
                                label = f"p.{entry['page_number']}"
                            else:
                                label = "参照"

                        for target_info in target_summaries:
                            page_label = target_info.get("page")
                            target_summary = target_info.get("summary")
                            digest_label = label
                            if page_label is not None:
                                digest_label = f"p.{page_label}"
                            digest_lines.append(f"{digest_label}: {target_summary}")

                if digest_lines:
                    metadata["reference_digest"] = digest_lines

            final_summary = summary or self._fallback_summary(chunk_text)
            if not summary and final_summary:
                record_summary_event(
                    "search_summary",
                    "fallback",
                    {
                        "chunk_id": metadata.get('chunk_id') or metadata.get('chunk_index') or 'unknown',
                        "file_name": metadata.get('file_name') or metadata.get('source_file') or 'unknown',
                        "query": query[:100] if query else "",
                    },
                )

            formatted_results.append(
                {
                    "doc_id": result.get("doc_id"),
                    "chunk_index": result.get("chunk_index"),
                    "score": result.get("relevance_score", 0.0),
                    "vector_score": result.get("vector_score", 0.0),
                    "lexical_score": result.get("lexical_score", 0.0),
                    "content": chunk_text,
                    "summary": final_summary,
                    "metadata": metadata,
                    "title": metadata.get("title")
                    or metadata.get("file_name")
                    or metadata.get("source_file")
                    or "無題",
                    "file_path": metadata.get("stored_path") or metadata.get("source_file"),
                }
            )

        return formatted_results

    def _generate_result_summary(
        self,
        query: str,
        chunk_text: str,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        chunk_id = metadata.get('chunk_id') or metadata.get('chunk_index') or 'unknown'
        file_name = metadata.get('file_name') or metadata.get('source_file') or 'unknown'

        if not self.client:
            record_summary_event(
                "search_summary",
                "failure",
                {
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "reason": "client_missing",
                    "query": query[:100] if query else "",
                },
            )
            return None

        if not chunk_text or not str(chunk_text).strip():
            record_summary_event(
                "search_summary",
                "failure",
                {
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "reason": "empty_text",
                    "query": query[:100] if query else "",
                },
            )
            return None

        instruction = """あなたは企業内ナレッジ検索のアシスタントです。以下の検索クエリとチャンク内容、メタデータに基づき、ユーザーの質問に直接答える120文字以内の日本語要約を1文で作成してください。事実はチャンクに含まれる情報のみを利用し、推測・誇張・架空の情報は含めないでください。不明な場合は『不明』と明記してください。形式はJSONで、summary(str)とkeywords(list[str],最大5件)の2項目のみ出力してください。"""
        instruction = textwrap.dedent(instruction).strip()

        stage2 = metadata.get("stage2_processing") or {}
        stage3 = metadata.get("stage3_business") or {}
        stage4 = metadata.get("stage4_search") or {}

        meta_lines: List[str] = []
        if metadata.get("summary"):
            meta_lines.append(f"既存要約: {metadata['summary']}")
        if stage2.get("image_summary"):
            meta_lines.append(f"画像要約: {stage2['image_summary']}")
        if stage3.get("image_use_case"):
            meta_lines.append(f"想定用途: {stage3['image_use_case']}")
        if stage4.get("keywords"):
            meta_lines.append("キーワード: " + ", ".join(map(str, stage4['keywords'])))
        if stage4.get("search_metadata"):
            meta_lines.append("検索ヒント: " + ", ".join(map(str, stage4['search_metadata'])))

        context_lines: List[str] = []
        for key in ("scene_type", "actions", "objects", "visible_text", "numbers", "environment"):
            value = metadata.get(key) or stage2.get(key) or stage3.get(key)
            if isinstance(value, (list, tuple, set)) and value:
                context_lines.append(f"{key}: {', '.join(map(str, value))}")
            elif isinstance(value, str) and value.strip():
                context_lines.append(f"{key}: {value.strip()}")

        user_blocks: List[Dict[str, Any]] = []
        user_blocks.append({"type": "input_text", "text": f"ユーザーの質問: {query}"})
        if meta_lines:
            user_blocks.append({"type": "input_text", "text": "メタデータ:\n" + "\n".join(meta_lines)})
        if context_lines:
            user_blocks.append({"type": "input_text", "text": "コンテキスト:\n" + "\n".join(context_lines)})
        user_blocks.append({"type": "input_text", "text": "チャンク本文:\n" + str(chunk_text)})

        try:
            if hasattr(self.client, "responses"):
                responses_kwargs = {
                    "model": "gpt-4.1-mini-2025-04-14",
                    "temperature": 0.2,
                    "input": [
                        {"role": "system", "content": [{"type": "input_text", "text": instruction}]},
                        {"role": "user", "content": user_blocks},
                    ],
                    "max_output_tokens": 300,
                }
                try:
                    response = self.client.responses.create(
                        response_format={"type": "json_object"},
                        **responses_kwargs,
                    )
                except TypeError as type_exc:
                    if "response_format" in str(type_exc):
                        logger.debug(
                            "responses.createがresponse_format未対応のためフォールバック: %s",
                            type_exc,
                        )
                        response = self.client.responses.create(**responses_kwargs)
                    else:
                        raise
                text = self._collect_response_text(response)
            elif hasattr(self.client, "chat"):
                user_text_parts: List[str] = []
                for block in user_blocks:
                    if isinstance(block, dict):
                        if block.get("type") == "input_text" and block.get("text"):
                            user_text_parts.append(str(block["text"]))
                        elif block.get("type") == "input_image_url" and block.get("image_url"):
                            user_text_parts.append(f"[image]: {block['image_url']}")
                    elif isinstance(block, str):
                        user_text_parts.append(block)
                user_message = "\n\n".join(user_text_parts) if user_text_parts else ""
                chat_kwargs = {
                    "model": "gpt-4.1-mini-2025-04-14",
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_message},
                    ],
                    "max_tokens": 300,
                }
                try:
                    response = self.client.chat.completions.create(
                        response_format={"type": "json_object"},
                        **chat_kwargs,
                    )
                except TypeError as type_exc:
                    if "response_format" in str(type_exc):
                        logger.debug(
                            "chat.completions.createがresponse_format未対応のためフォールバック: %s",
                            type_exc,
                        )
                        response = self.client.chat.completions.create(**chat_kwargs)
                    else:
                        raise
                text = self._collect_response_text(response)
            else:
                # クライアント未対応の場合もエラーログに記録（情報レベル）
                error_logger.handle_error(
                    Exception("OpenAIクライアントがResponses/Chat双方に未対応"),
                    context_data={
                        "module": "rag_engine",
                        "action": "generate_result_summary",
                        "query": query[:100] if query else "empty",
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "client_type": type(self.client).__name__ if self.client else None,
                        "has_client": self.client is not None,
                        "reason": "client_not_supported",
                    },
                    severity=ErrorSeverity.LOW,
                    category=ErrorCategory.CONFIGURATION,
                )
                record_summary_event(
                    "search_summary",
                    "failure",
                    {
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "reason": "client_not_supported",
                        "query": query[:100] if query else "",
                    },
                )
                return None

            if not text:
                # レスポンスが空の場合もエラーログに記録
                error_logger.handle_error(
                    Exception("LLMレスポンスが空"),
                    context_data={
                        "module": "rag_engine",
                        "action": "generate_result_summary",
                        "query": query[:100] if query else "empty",
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "reason": "empty_response",
                        "has_response": response is not None,
                        "response_type": type(response).__name__ if response else None,
                    },
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.EXTERNAL_API,
                )
                record_summary_event(
                    "search_summary",
                    "failure",
                    {
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "reason": "empty_response",
                        "query": query[:100] if query else "",
                    },
                )
                return None

            parsed = json.loads(text)
            summary_text = str(parsed.get("summary", "")).strip()
            if not summary_text:
                record_summary_event(
                    "search_summary",
                    "failure",
                    {
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "reason": "empty_summary",
                        "query": query[:100] if query else "",
                    },
                )
                return None

            summary_text = summary_text.replace("\n", " ").strip()
            if len(summary_text) > 0 and summary_text[0] in {"。", "、", "・", "-"}:
                summary_text = summary_text[1:].strip()
            if len(summary_text) > 120:
                summary_text = summary_text[:117].rstrip() + "…"
            if not summary_text.endswith(tuple("。.!?")):
                summary_text += "。"

            keywords = parsed.get("keywords")
            keywords_list = []
            if isinstance(keywords, list):
                for kw in keywords[:5]:
                    kw_text = str(kw).strip()
                    if kw_text:
                        keywords_list.append(kw_text)

            record_summary_event(
                "search_summary",
                "success",
                {
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "summary_length": len(summary_text),
                    "keyword_count": len(keywords_list),
                    "query": query[:100] if query else "",
                },
            )
            return {
                "summary": summary_text,
                "keywords": keywords_list,
            }

        except Exception as exc:
            # エラーログに詳細を記録
            chunk_text_preview = str(chunk_text)[:200] if chunk_text else 'empty'
            
            error_logger.handle_error(
                exc,
                context_data={
                    "module": "rag_engine",
                    "action": "generate_result_summary",
                    "query": query[:100] if query else "empty",
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "file_type": metadata.get('file_type', 'unknown'),
                    "chunk_text_preview": chunk_text_preview,
                    "chunk_text_length": len(str(chunk_text)) if chunk_text else 0,
                    "has_client": self.client is not None,
                    "client_type": type(self.client).__name__ if self.client else None,
                    "client_has_responses": hasattr(self.client, "responses") if self.client else False,
                    "client_has_chat": hasattr(self.client, "chat") if self.client else False,
                },
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.EXTERNAL_API,
            )
            logging.getLogger(__name__).warning(f"検索結果要約生成エラー: {exc} (詳細はエラーログを確認してください)")
            record_summary_event(
                "search_summary",
                "failure",
                {
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "reason": type(exc).__name__,
                    "query": query[:100] if query else "",
                },
            )
            return None

    @staticmethod
    def _fallback_summary(text: str, max_length: int = 120) -> str:
        snippet = " ".join(str(text).split())
        if len(snippet) <= max_length:
            return snippet
        return snippet[: max_length - 1].rstrip() + "…"

    @staticmethod
    def _collect_response_text(response: Any) -> Optional[str]:
        if response is None:
            return None

        for attr in ("output_text", "text"):
            value = getattr(response, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        collected: List[str] = []
        for attr in ("output", "data", "choices"):
            blocks = getattr(response, attr, None)
            if not blocks:
                continue
            try:
                for block in blocks:
                    content = None
                    if hasattr(block, "content"):
                        content = block.content
                    elif hasattr(block, "message"):
                        message = block.message
                        if hasattr(message, "content"):
                            content = message.content
                    elif isinstance(block, dict):
                        if "content" in block:
                            content = block.get("content")
                        elif "message" in block:
                            content = block["message"].get("content")

                    if isinstance(content, list):
                        for item in content:
                            text_value = None
                            if hasattr(item, "text"):
                                text_value = item.text
                            elif isinstance(item, dict):
                                text_value = item.get("text")
                            if isinstance(text_value, str) and text_value.strip():
                                collected.append(text_value.strip())
                    elif hasattr(content, "text") and isinstance(content.text, str) and content.text.strip():
                        collected.append(content.text.strip())
                    elif isinstance(content, str) and content.strip():
                        collected.append(content.strip())
            except Exception:
                continue

        if collected:
            return "\n".join(collected)
        return None

    @staticmethod
    def _sensitivity_rank(level: str) -> int:
        order = ["公開", "社内", "機密", "最高機密"]
        try:
            return order.index(level)
        except ValueError:
            return 0

    @classmethod
    def _is_accessible(cls, result: Dict[str, Any], acl_filters: Dict[str, Any]) -> bool:
        metadata = result.get("metadata") or {}

        tenant_filter = str(acl_filters.get("tenant_id", "")).strip()
        if tenant_filter:
            doc_tenant = str(metadata.get("tenant_id", "")).strip()
            if doc_tenant != tenant_filter:
                return False

        allowed_groups = acl_filters.get("allowed_groups") or []
        if isinstance(allowed_groups, str):
            allowed_groups = [g.strip() for g in allowed_groups.split(",") if g.strip()]
        doc_groups = metadata.get("allowed_groups") or []
        if isinstance(doc_groups, str):
            doc_groups = [g.strip() for g in doc_groups.split(",") if g.strip()]
        if allowed_groups:
            if not doc_groups:
                return False
            if not any(group in doc_groups for group in allowed_groups):
                return False

        max_level = acl_filters.get("max_sensitivity_level") or acl_filters.get("sensitivity")
        if max_level:
            doc_level = metadata.get("sensitivity") or metadata.get("sensitivity_level") or "公開"
            if cls._sensitivity_rank(str(doc_level)) > cls._sensitivity_rank(str(max_level)):
                return False

        return True

    @staticmethod
    def _map_search_mode(search_mode: Optional[str]) -> str:
        if not search_mode:
            return "hybrid"
        normalized = str(search_mode).lower()
        mapping = {
            "hybrid_rrf": "hybrid",
            "adaptive": "hybrid",
            "hybrid": "hybrid",
            "vector": "vector_only",
            "vector_only": "vector_only",
            "lexical": "lexical",
            "bm25": "lexical",
        }
        return mapping.get(normalized, "hybrid")







