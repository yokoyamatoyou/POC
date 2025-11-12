"""
Phase 4: 統合処理メインクラス
全フェーズの統合処理を管理するメインクラス
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import json
import textwrap
import re
from src.error_handler import error_logger, ErrorSeverity, ErrorCategory
from src.app.utils.summary_metrics import record_summary_event

# Phase 1-3のモジュールをインポート
try:
    from .document_classifier import classify_document
    EXCEL_PROCESSING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Excel処理モジュールのインポートに失敗: {e}")
    EXCEL_PROCESSING_AVAILABLE = False

try:
    from .multimodal import MultimodalProcessor, ProcessingMode
    MULTIMODAL_PROCESSING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"マルチモーダル処理モジュールのインポートに失敗: {e}")
    MULTIMODAL_PROCESSING_AVAILABLE = False

try:
    from .pdf_processor import PDFProcessor
    PDF_PROCESSING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PDF処理モジュールのインポートに失敗: {e}")
    PDF_PROCESSING_AVAILABLE = False

# Phase 4のモジュールをインポート
try:
    from .semantic_search import SemanticSearchEngine
    from .query_processor import QueryProcessor
    from .result_ranker import ResultRanker
    from .unified_search import UnifiedSearchEngine
    PHASE4_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase 4モジュールのインポートに失敗: {e}")
    PHASE4_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """処理ステータス"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingResult:
    """処理結果"""
    status: ProcessingStatus
    success: bool
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None
    quality_score: float = 0.0


@dataclass
class DocumentInfo:
    """文書情報"""
    file_path: str
    file_type: str
    file_size: int
    processing_method: str
    document_type: str
    detailed_type: str
    metadata: Dict[str, Any]


class IntegratedProcessor:
    """統合処理メインクラス"""
    
    def __init__(self, client=None, temp_dir: Optional[Path] = None):
        """
        初期化
        
        Args:
            client: OpenAIクライアント
            temp_dir: 一時ディレクトリ
        """
        self.client = client
        self.temp_dir = temp_dir or Path("temp_processing")
        self.temp_dir.mkdir(exist_ok=True)
        
        # 処理統計
        self.processing_stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0
        }
        
        # Phase 4のエンジンを初期化
        self.semantic_engine: Optional[SemanticSearchEngine] = None
        self.query_processor: Optional[QueryProcessor] = None
        self.result_ranker: Optional[ResultRanker] = None
        self.unified_search: Optional[UnifiedSearchEngine] = None
        if PHASE4_AVAILABLE:
            self.semantic_engine = SemanticSearchEngine()
            self.query_processor = QueryProcessor()
            self.result_ranker = ResultRanker()
            self.unified_search = UnifiedSearchEngine(
                semantic_engine=self.semantic_engine,
                query_processor=self.query_processor,
                result_ranker=self.result_ranker
            )
    
    def process_document(self, file_path: str, base_metadata: Dict[str, Any]) -> ProcessingResult:
        """
        文書の統合処理
        
        Args:
            file_path: ファイルパス
            base_metadata: 基本メタデータ
            
        Returns:
            ProcessingResult: 処理結果
        """
        start_time = time.time()
        file_path_obj = Path(file_path)
        
        try:
            # 文書情報の取得
            doc_info = self._get_document_info(file_path_obj, base_metadata)
            
            # 文書タイプに応じた処理方法の選択
            processing_method = self._select_processing_method(doc_info)
            
            # 処理の実行
            chunks, metadata = self._execute_processing(doc_info, processing_method)
            
            # 品質スコアの計算
            quality_score = self._calculate_quality_score(chunks, metadata)
            
            # 処理時間の計算
            processing_time = time.time() - start_time
            
            # 統計の更新
            self._update_stats(True, len(chunks), processing_time, quality_score)
            
            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                success=True,
                chunks=chunks,
                metadata=metadata,
                processing_time=processing_time,
                quality_score=quality_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            logger.error(f"文書処理エラー: {file_path}, {error_message}")
            
            # 統計の更新
            self._update_stats(False, 0, processing_time, 0.0)
            
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                success=False,
                chunks=[],
                metadata={},
                processing_time=processing_time,
                error_message=error_message
            )
    
    def enhance_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        チャンクの品質向上処理
        
        Args:
            chunks: 処理対象のチャンクリスト
            metadata: メタデータ
            
        Returns:
            List[Dict[str, Any]]: 強化されたチャンクリスト
        """
        try:
            logger.info(f"チャンク強化処理開始: {len(chunks)}チャンク")
            
            # 基本的な品質向上処理
            enhanced_chunks = []
            for chunk in chunks:
                enhanced_chunk = chunk.copy()
                
                # テキストの正規化
                if 'text' in enhanced_chunk:
                    text = enhanced_chunk['text']
                    # 余分な空白の除去
                    text = ' '.join(text.split())
                    enhanced_chunk['text'] = text
                
                metadata_context = metadata.copy() if isinstance(metadata, dict) else {}
                chunk_metadata = enhanced_chunk.get('metadata') or {}
                if not isinstance(chunk_metadata, dict):
                    chunk_metadata = {}
                enhanced_chunk['metadata'] = chunk_metadata

                # 品質スコアの計算
                if 'text' in enhanced_chunk and enhanced_chunk['text']:
                    quality_score = self._calculate_chunk_quality(enhanced_chunk)
                    enhanced_chunk['quality_score'] = quality_score
                else:
                    enhanced_chunk['quality_score'] = 0.0
                
                # LLMによる要約生成（日本語）
                summary = chunk_metadata.get('summary') or (
                    (chunk_metadata.get('stage4_search') or {}).get('summary')
                )
                summary_required = summary is None or not str(summary).strip()

                if summary_required:
                    summary_payload = self._generate_chunk_summary(enhanced_chunk, metadata_context)
                    if summary_payload and summary_payload.get('summary'):
                        summary_text = summary_payload['summary'].strip()
                        chunk_metadata['summary'] = summary_text
                        stage4_meta = chunk_metadata.setdefault('stage4_search', {})
                        stage4_meta['summary'] = summary_text

                        keywords = summary_payload.get('keywords') or []
                        if keywords:
                            stage4_keywords = stage4_meta.setdefault('keywords', [])
                            existing_set = {str(kw).strip() for kw in stage4_keywords if str(kw).strip()}
                            for kw in keywords:
                                kw_text = str(kw).strip()
                                if kw_text and kw_text not in existing_set:
                                    stage4_keywords.append(kw_text)
                                    existing_set.add(kw_text)

                enhanced_chunk['enhanced'] = True
                enhanced_chunk['enhancement_timestamp'] = time.time()
                
                enhanced_chunks.append(enhanced_chunk)
            
            logger.info(f"チャンク強化処理完了: {len(enhanced_chunks)}チャンク")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"チャンク強化処理エラー: {e}")
            return chunks  # エラー時は元のチャンクを返す

    def _generate_chunk_summary(
        self,
        chunk: Dict[str, Any],
        document_metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """チャンク内容から日本語要約を生成"""

        chunk_metadata = chunk.get('metadata') or {}
        chunk_id = chunk.get('chunk_id') or chunk_metadata.get('chunk_id') or 'unknown'
        document_name = chunk_metadata.get('file_name') or document_metadata.get('file_name') or 'unknown'

        if not self.client:
            record_summary_event(
                "chunk_summary",
                "failure",
                {
                    "chunk_id": chunk_id,
                    "file_name": document_name,
                    "reason": "client_missing",
                },
            )
            return None

        chunk_text = (
            chunk.get('content')
            or chunk.get('text')
            or (chunk.get('metadata') or {}).get('text_content')
        )
        if not chunk_text or not str(chunk_text).strip():
            record_summary_event(
                "chunk_summary",
                "failure",
                {
                    "chunk_id": chunk_id,
                    "file_name": document_name,
                    "reason": "empty_text",
                },
            )
            return None

        summary_hint = chunk_metadata.get('summary')
        if summary_hint and str(summary_hint).strip():
            record_summary_event(
                "chunk_summary",
                "skipped",
                {
                    "chunk_id": chunk_id,
                    "file_name": document_name,
                    "reason": "already_present",
                },
            )
            return None

        prompt_payload = self._build_chunk_summary_prompt(chunk, document_metadata)
        if prompt_payload is None:
            record_summary_event(
                "chunk_summary",
                "failure",
                {
                    "chunk_id": chunk_id,
                    "file_name": document_name,
                    "reason": "prompt_build_failed",
                },
            )
            return None

        system_prompt, user_blocks = prompt_payload

        try:
            if hasattr(self.client, "responses"):
                response = None
                responses_kwargs = {
                    "model": "gpt-4.1-mini-2025-04-14",
                    "temperature": 0.2,
                    "input": [
                        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                        {"role": "user", "content": user_blocks},
                    ],
                    "max_output_tokens": 600,
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
                content_text = self._collect_response_text(response)
            elif hasattr(self.client, "chat"):
                user_text = []
                for block in user_blocks:
                    if isinstance(block, dict):
                        if block.get("type") == "input_text" and block.get("text"):
                            user_text.append(str(block["text"]))
                        elif block.get("type") == "input_image_url" and block.get("image_url"):
                            user_text.append(f"[image]: {block['image_url']}")
                    elif isinstance(block, str):
                        user_text.append(block)
                user_message = "\n\n".join(user_text) if user_text else ""
                chat_kwargs = {
                    "model": "gpt-4.1-mini-2025-04-14",
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "max_tokens": 600,
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
                content_text = self._collect_response_text(response)
            else:
                # クライアント未対応の場合もエラーログに記録（情報レベル）
                error_logger.handle_error(
                    Exception("OpenAIクライアントがResponses/Chat双方に未対応"),
                    context_data={
                        "module": "integrated_processor",
                        "action": "generate_chunk_summary",
                        "chunk_id": chunk_id,
                        "file_name": document_name,
                        "client_type": type(self.client).__name__ if self.client else None,
                        "has_client": self.client is not None,
                        "reason": "client_not_supported",
                    },
                    severity=ErrorSeverity.LOW,
                    category=ErrorCategory.CONFIGURATION,
                )
                logger.debug("OpenAIクライアントがResponses/Chat双方に未対応のため要約をスキップしました")
                record_summary_event(
                    "chunk_summary",
                    "failure",
                    {
                        "chunk_id": chunk_id,
                        "file_name": document_name,
                        "reason": "client_not_supported",
                    },
                )
                return None

            if not content_text:
                # レスポンスが空の場合もエラーログに記録
                error_logger.handle_error(
                    Exception("LLMレスポンスが空"),
                    context_data={
                        "module": "integrated_processor",
                        "action": "generate_chunk_summary",
                        "chunk_id": chunk_id,
                        "file_name": document_name,
                        "reason": "empty_response",
                        "has_response": response is not None,
                        "response_type": type(response).__name__ if response else None,
                    },
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.EXTERNAL_API,
                )
                record_summary_event(
                    "chunk_summary",
                    "failure",
                    {
                        "chunk_id": chunk_id,
                        "file_name": document_name,
                        "reason": "empty_response",
                    },
                )
                return None

            parsed = json.loads(content_text)
            summary_text = str(parsed.get('summary', '')).strip()
            keywords = parsed.get('keywords') or []

            payload = {
                "summary": summary_text,
                "keywords": keywords if isinstance(keywords, list) else [],
            }
            record_summary_event(
                "chunk_summary",
                "success",
                {
                    "chunk_id": chunk_id,
                    "file_name": document_name,
                    "summary_length": len(summary_text),
                    "keyword_count": len(payload["keywords"]),
                },
            )
            return payload

        except Exception as exc:
            # エラーログに詳細を記録
            chunk_text_preview = str(chunk_text)[:200] if chunk_text else 'empty'
            
            error_logger.handle_error(
                exc,
                context_data={
                    "module": "integrated_processor",
                    "action": "generate_chunk_summary",
                    "chunk_id": chunk_id,
                    "file_name": document_name,
                    "file_type": chunk_metadata.get('file_type', 'unknown'),
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
            logger.warning(f"チャンク要約生成エラー: {exc} (詳細はエラーログを確認してください)")
            record_summary_event(
                "chunk_summary",
                "failure",
                {
                    "chunk_id": chunk_id,
                    "file_name": document_name,
                    "reason": type(exc).__name__,
                },
            )
            return None

    def _build_chunk_summary_prompt(
        self,
        chunk: Dict[str, Any],
        document_metadata: Dict[str, Any],
    ) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        chunk_text = (
            chunk.get('content')
            or chunk.get('text')
            or (chunk.get('metadata') or {}).get('text_content')
        )
        if not chunk_text or not str(chunk_text).strip():
            return None

        chunk_metadata = chunk.get('metadata') or {}
        file_type = (chunk_metadata.get('file_type') or '').lower()
        chunk_type = (chunk_metadata.get('chunk_type') or '').lower()

        stage2 = chunk_metadata.get('stage2_processing') or {}
        stage3 = chunk_metadata.get('stage3_business') or {}
        stage4 = chunk_metadata.get('stage4_search') or {}

        summary_elements: List[str] = []
        summary_hint = chunk_metadata.get('summary')
        if summary_hint:
            summary_elements.append(f"既存サマリ: {summary_hint}")

        if stage2.get('image_summary'):
            summary_elements.append(f"画像要約: {stage2['image_summary']}")
        if stage3.get('image_use_case'):
            summary_elements.append(f"想定用途: {stage3['image_use_case']}")
        if stage4.get('keywords'):
            summary_elements.append("キーワード: " + ", ".join(map(str, stage4['keywords'])))
        if stage4.get('search_metadata'):
            summary_elements.append("検索ヒント: " + ", ".join(map(str, stage4['search_metadata'])))

        doc_context_lines = []
        for key in ("document_type", "detailed_type", "processing_method"):
            value = document_metadata.get(key)
            if value:
                doc_context_lines.append(f"{key}: {value}")

        if document_metadata.get('title'):
            doc_context_lines.append(f"title: {document_metadata['title']}")

        chunk_context_lines = []
        for key in (
            "scene_type",
            "actions",
            "objects",
            "visible_text",
            "numbers",
            "environment",
            "technical_details",
        ):
            value = chunk_metadata.get(key) or stage2.get(key) or stage3.get(key)
            if isinstance(value, (list, tuple, set)) and value:
                chunk_context_lines.append(f"{key}: {', '.join(map(str, value))}")
            elif isinstance(value, str) and value.strip():
                chunk_context_lines.append(f"{key}: {value.strip()}")

        instruction = """あなたは日本語のナレッジベース要約アシスタントです。
        以下のチャンク内容とメタデータを理解し、業務ユーザーが検索結果を素早く把握できるように、
        120文字程度の自然な日本語で客観的な要約を作成してください。
        表示形式は JSON とし、キーは必ず "summary" (str) と "keywords" (list[str, 最大10件]) の2つのみを使用してください。
        キーワードは検索時に有用な短い語句やフレーズを日本語で列挙してください。
        画像や数値データがある場合は内容を簡潔に含め、推測は避けてください。
        """

        instruction = textwrap.dedent(instruction).strip()

        user_blocks: List[Dict[str, Any]] = []
        user_blocks.append({"type": "input_text", "text": instruction})

        if doc_context_lines:
            user_blocks.append({
                "type": "input_text",
                "text": "文書メタデータ:\n" + "\n".join(doc_context_lines),
            })

        if summary_elements:
            user_blocks.append({
                "type": "input_text",
                "text": "参考情報:\n" + "\n".join(summary_elements),
            })

        if chunk_context_lines:
            user_blocks.append({
                "type": "input_text",
                "text": "チャンクメタデータ:\n" + "\n".join(chunk_context_lines),
            })

        chunk_header = f"チャンク種類: {chunk_type or file_type or 'unknown'}"
        user_blocks.append({
            "type": "input_text",
            "text": chunk_header + "\n---\n" + str(chunk_text),
        })

        return instruction, user_blocks

    @staticmethod
    def _collect_response_text(response: Any) -> Optional[str]:
        if response is None:
            return None

        for attr in ("output_text", "text"):
            text = getattr(response, attr, None)
            if isinstance(text, str) and text.strip():
                return text.strip()

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
    
    def _calculate_chunk_quality(self, chunk: Dict[str, Any]) -> float:
        """チャンクの品質スコア計算"""
        try:
            text = chunk.get('text', '')
            if not text:
                return 0.0
            
            # 基本的な品質指標
            length_score = min(len(text) / 100, 1.0)  # 長さスコア（100文字で1.0）
            
            # 文字の多様性スコア
            unique_chars = len(set(text.lower()))
            diversity_score = min(unique_chars / 20, 1.0)  # 20種類で1.0
            
            # 文の完全性スコア（句点で終わっているか）
            completeness_score = 1.0 if text.strip().endswith(('。', '.', '!', '?')) else 0.8
            
            # 総合品質スコア
            quality_score = (length_score * 0.4 + diversity_score * 0.3 + completeness_score * 0.3)
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"品質スコア計算エラー: {e}")
            return 0.5  # デフォルト値
    
    def _get_document_info(self, file_path: Path, base_metadata: Dict[str, Any]) -> DocumentInfo:
        """文書情報の取得"""
        file_type = file_path.suffix.lower()
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        # 文書分類の実行
        if EXCEL_PROCESSING_AVAILABLE:
            classification = classify_document(str(file_path))
            document_type = classification.get("basic_type", "unknown")
            detailed_type = classification.get("detailed_type", "unknown")
            processing_method = classification.get("processing_method", "unknown")
        else:
            document_type = "unknown"
            detailed_type = "unknown"
            processing_method = "unknown"
        
        return DocumentInfo(
            file_path=str(file_path),
            file_type=file_type,
            file_size=file_size,
            processing_method=processing_method,
            document_type=document_type,
            detailed_type=detailed_type,
            metadata=base_metadata
        )
    
    def _select_processing_method(self, doc_info: DocumentInfo) -> str:
        """処理方法の選択"""
        file_type = doc_info.file_type
        
        if file_type in {".xlsx", ".xls", ".xlsm", ".xlsb"} and EXCEL_PROCESSING_AVAILABLE:
            return "excel_processing"
        elif file_type == ".pdf" and PDF_PROCESSING_AVAILABLE:
            return "pdf_processing"
        elif file_type in {".docx", ".pptx"} and MULTIMODAL_PROCESSING_AVAILABLE:
            return "multimodal_processing"
        elif file_type in {".txt", ".md"}:
            return "text_processing"
        else:
            return "unknown_processing"
    
    def _execute_processing(self, doc_info: DocumentInfo, processing_method: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """処理の実行"""
        chunks: List[Dict[str, Any]] = []
        metadata = doc_info.metadata.copy()
        
        if processing_method == "excel_processing" and EXCEL_PROCESSING_AVAILABLE:
            chunks = _process_excel_file(Path(doc_info.file_path), metadata)
            metadata.update({
                "processing_method": "excel_processing",
                "document_type": doc_info.document_type,
                "detailed_type": doc_info.detailed_type
            })
            
        elif processing_method == "pdf_processing" and PDF_PROCESSING_AVAILABLE:
            pdf_processor = PDFProcessor()
            result = pdf_processor.process_pdf(doc_info.file_path)
            if result.success:
                # PDF結果をチャンクに変換
                chunks = self._convert_pdf_result_to_chunks(result)
                metadata.update({
                    "processing_method": "pdf_processing",
                    "document_type": doc_info.document_type,
                    "detailed_type": doc_info.detailed_type,
                    "quality_score": result.quality_score
                })
            
        elif processing_method == "multimodal_processing" and MULTIMODAL_PROCESSING_AVAILABLE:
            multimodal_processor = MultimodalProcessor()
            result = multimodal_processor.process_document_with_images(doc_info.file_path, ProcessingMode.BALANCED)
            if result.success and result.document:
                # マルチモーダル結果をチャンクに変換
                chunks = self._convert_multimodal_result_to_chunks(result)
                metadata.update({
                    "processing_method": "multimodal_processing",
                    "document_type": doc_info.document_type,
                    "detailed_type": doc_info.detailed_type
                })
            
        elif processing_method == "text_processing":
            # テキストファイルの処理
            chunks = self._process_text_file(doc_info.file_path, metadata)
            metadata.update({
                "processing_method": "text_processing",
                "document_type": doc_info.document_type,
                "detailed_type": doc_info.detailed_type
            })
        
        return chunks, metadata
    
    def _convert_pdf_result_to_chunks(self, result) -> List[Dict[str, Any]]:
        """PDF結果をチャンクに変換"""
        chunks = []
        
        # テキストブロックを意味単位に統合
        text_chunks = self._group_pdf_text_blocks(result.text_blocks)
        chunks.extend(text_chunks)
        
        # 表
        for table in result.tables:
            if table.rows:
                page_index = int(getattr(table, "page_number", 0) or 0)
                table_text = f"[表 {table.table_id}]\n"
                if table.headers:
                    table_text += " | ".join(table.headers) + "\n"
                    table_text += " | ".join(["---"] * len(table.headers)) + "\n"
                
                for row in table.rows:
                    table_text += " | ".join(str(cell) for cell in row) + "\n"
                
                chunks.append({
                    "text": table_text,
                    "metadata": {
                        "chunk_type": "pdf_table",
                        "table_id": table.table_id,
                        "file_type": "pdf",
                        "page_number": table.page_number,
                        "page_index": page_index,
                        "page_label": page_index + 1,
                        "confidence": table.confidence,
                        "bbox": table.bbox
                    }
                })
        
        # 画像
        for processed_image in result.processed_images:
            if processed_image.ocr_text.strip():
                page_index = int(getattr(processed_image.original_image, "page_number", 0) or 0)
                chunks.append({
                    "text": f"[画像 {processed_image.image_id}]: {processed_image.ocr_text}",
                    "metadata": {
                        "chunk_type": "pdf_image",
                        "file_type": "pdf",
                        "image_id": processed_image.image_id,
                        "image_type": processed_image.image_type,
                        "ocr_confidence": processed_image.ocr_confidence,
                        "page_number": processed_image.original_image.page_number,
                        "page_index": page_index,
                        "page_label": page_index + 1,
                        "bbox": processed_image.original_image.bbox
                    }
                })
        
        if not chunks:
            return chunks

        return self._attach_pdf_reference_links(chunks)

    _PDF_PAGE_PATTERN = re.compile(
        r"(?:p(?:age)?\s*\.\s*|page\s+|pages\s+|pp\.\s*|ページ|頁|P\s*\.)\s*(\d{1,4})",
        re.IGNORECASE,
    )

    def _attach_pdf_reference_links(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """PDFチャンクにページ参照リンクメタデータを付与"""

        page_map: Dict[int, List[int]] = {}

        for idx, chunk in enumerate(chunks):
            metadata = chunk.setdefault("metadata", {})
            page_index_raw = metadata.get("page_index", metadata.get("page_number"))
            try:
                page_index = int(page_index_raw)
            except (TypeError, ValueError):
                page_index = None

            if page_index is not None:
                metadata["page_index"] = page_index
                metadata.setdefault("page_label", page_index + 1)
                page_map.setdefault(page_index, []).append(idx)

        for chunk in chunks:
            text = (
                chunk.get("text")
                or chunk.get("content")
                or (chunk.get("metadata") or {}).get("text_content")
            )
            if not text:
                continue

            metadata = chunk.setdefault("metadata", {})
            found_pages: Dict[int, str] = {}

            for match in self._PDF_PAGE_PATTERN.finditer(text):
                raw_label = match.group(0)
                number = match.group(1)
                try:
                    page_number = int(number)
                except ValueError:
                    continue
                if page_number <= 0:
                    continue
                page_index = page_number - 1
                found_pages.setdefault(page_index, raw_label.strip())

            if not found_pages:
                continue

            reference_links: List[Dict[str, Any]] = metadata.setdefault("reference_links", [])

            for page_index, label in found_pages.items():
                entry = {
                    "type": "page",
                    "label": label,
                    "page_number": page_index + 1,
                    "page_index": page_index,
                    "resolved": False,
                    "status": "pending",
                }
                reference_links.append(entry)

            metadata["ref_total_count"] = len(reference_links)
            metadata.setdefault("stage3_business", {})["reference_links"] = reference_links
            metadata.setdefault("stage4_search", {})["reference_links"] = reference_links

        return chunks

    def _group_pdf_text_blocks(
        self,
        text_blocks: List[Any],
        *,
        max_chars: int = 1200,
        min_chars: int = 200,
        gap_threshold: float = 25.0,
    ) -> List[Dict[str, Any]]:
        """PDFのテキストブロックをレイアウトベースでまとめる"""

        if not text_blocks:
            return []

        from collections import defaultdict

        grouped: Dict[int, List[Any]] = defaultdict(list)
        for idx, block in enumerate(text_blocks):
            if not getattr(block, "text", "").strip():
                continue
            page_number = int(getattr(block, "page_number", 0) or 0)
            grouped[page_number].append((idx, block))

        chunk_records: List[Dict[str, Any]] = []

        for page_index, page_blocks in grouped.items():
            page_blocks.sort(key=lambda item: (self._safe_bbox_top(item[1]), self._safe_bbox_left(item[1])))

            current_text_parts: List[str] = []
            current_indices: List[int] = []
            current_bbox = None
            current_length = 0
            previous_bottom = None

            def flush_chunk(force: bool = False) -> None:
                nonlocal current_text_parts, current_indices, current_bbox, current_length
                if not current_text_parts:
                    return

                chunk_text = "\n".join(current_text_parts).strip()
                if not chunk_text:
                    current_text_parts = []
                    current_indices = []
                    current_bbox = None
                    current_length = 0
                    return

                bbox = current_bbox or (0.0, 0.0, 0.0, 0.0)
                chunk_metadata = {
                    "chunk_type": "pdf_text",
                    "file_type": "pdf",
                    "page_number": page_index,
                    "page_index": page_index,
                    "page_label": page_index + 1,
                    "block_indices": list(current_indices),
                    "block_count": len(current_indices),
                    "bbox": bbox,
                }

                chunk_records.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                })

                current_text_parts = []
                current_indices = []
                current_bbox = None
                current_length = 0

            for block_index, block in page_blocks:
                block_text = getattr(block, "text", "")
                if not block_text.strip():
                    continue

                bbox = getattr(block, "bbox", None)
                block_top = self._safe_bbox_top(block)
                block_bottom = self._safe_bbox_bottom(block)

                spatial_gap = None
                if previous_bottom is not None and block_top is not None:
                    spatial_gap = block_top - previous_bottom

                if current_text_parts:
                    if spatial_gap is not None and spatial_gap > gap_threshold:
                        flush_chunk()
                    elif current_length >= max_chars and current_length >= min_chars:
                        flush_chunk()

                if not current_text_parts and spatial_gap is not None and spatial_gap > gap_threshold * 2:
                    flush_chunk()

                if not current_text_parts:
                    current_bbox = bbox
                else:
                    current_bbox = self._merge_bbox(current_bbox, bbox)

                current_text_parts.append(block_text.strip())
                current_indices.append(block_index)
                current_length += len(block_text)
                previous_bottom = block_bottom

                if current_length >= max_chars:
                    flush_chunk()

            flush_chunk()

        return chunk_records

    @staticmethod
    def _safe_bbox_top(block: Any) -> float:
        bbox = getattr(block, "bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            return float(bbox[1])
        return 0.0

    @staticmethod
    def _safe_bbox_bottom(block: Any) -> float:
        bbox = getattr(block, "bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return float(bbox[3])
        return 0.0

    @staticmethod
    def _safe_bbox_left(block: Any) -> float:
        bbox = getattr(block, "bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 1:
            return float(bbox[0])
        return 0.0

    @staticmethod
    def _merge_bbox(bbox_a: Optional[Tuple[float, float, float, float]], bbox_b: Optional[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        if bbox_a is None:
            return bbox_b or (0.0, 0.0, 0.0, 0.0)
        if bbox_b is None:
            return bbox_a
        x0 = min(bbox_a[0], bbox_b[0])
        y0 = min(bbox_a[1], bbox_b[1])
        x1 = max(bbox_a[2], bbox_b[2])
        y1 = max(bbox_a[3], bbox_b[3])
        return (x0, y0, x1, y1)
    
    def _convert_multimodal_result_to_chunks(self, result) -> List[Dict[str, Any]]:
        """マルチモーダル結果をチャンクに変換"""
        chunks = []
        
        # 基本テキストチャンク
        if result.document.text_content:
            chunks.append({
                "text": result.document.text_content,
                "metadata": {
                    "chunk_type": "multimodal_text",
                    "processing_method": "multimodal_processing",
                    "total_images": result.document.total_images,
                    "ocr_success_rate": result.document.ocr_success_rate
                }
            })
        
        # 画像チャンク
        for image in result.document.images:
            if image.ocr_text:
                chunks.append({
                    "text": f"[画像 {image.image_id}]: {image.ocr_text}",
                    "metadata": {
                        "chunk_type": "multimodal_image",
                        "image_id": image.image_id,
                        "image_type": image.image_type.value,
                        "ocr_confidence": image.ocr_confidence,
                        "quality_score": image.quality_score
                    }
                })
        
        return chunks
    
    def _process_text_file(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """テキストファイルの処理"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp932', errors='ignore') as f:
                content = f.read()
        
        return [{
            "text": content,
            "metadata": metadata
        }]
    
    def _calculate_quality_score(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> float:
        """品質スコアの計算"""
        if not chunks:
            return 0.0
        
        # 基本的な品質指標
        total_text_length = sum(len(chunk.get("text", "")) for chunk in chunks)
        avg_chunk_length = total_text_length / len(chunks) if chunks else 0
        
        # チャンク数の適切性
        chunk_count_score = min(len(chunks) / 10, 1.0)  # 10チャンクで満点
        
        # テキスト長の適切性
        length_score = min(avg_chunk_length / 500, 1.0)  # 500文字で満点
        
        # メタデータの充実度
        metadata_score = len(metadata) / 10  # 10個のメタデータで満点
        
        # 総合スコア
        quality_score = (chunk_count_score * 0.4 + length_score * 0.4 + metadata_score * 0.2)
        
        return min(quality_score, 1.0)
    
    def _update_stats(self, success: bool, chunk_count: int, processing_time: float, quality_score: float):
        """統計の更新"""
        self.processing_stats["total_files"] += 1
        if success:
            self.processing_stats["successful_files"] += 1
        else:
            self.processing_stats["failed_files"] += 1
        
        self.processing_stats["total_chunks"] += chunk_count
        self.processing_stats["total_processing_time"] += processing_time
        
        # 平均品質スコアの更新
        if self.processing_stats["successful_files"] > 0:
            current_avg = self.processing_stats["average_quality_score"]
            new_avg = (current_avg * (self.processing_stats["successful_files"] - 1) + quality_score) / self.processing_stats["successful_files"]
            self.processing_stats["average_quality_score"] = new_avg
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """処理統計の取得"""
        stats = self.processing_stats.copy()
        
        if stats["total_files"] > 0:
            stats["success_rate"] = stats["successful_files"] / stats["total_files"]
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_files"]
            stats["average_chunks_per_file"] = stats["total_chunks"] / stats["total_files"]
        else:
            stats["success_rate"] = 0.0
            stats["average_processing_time"] = 0.0
            stats["average_chunks_per_file"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """統計のリセット"""
        self.processing_stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0
        }
    
    def export_processing_report(self, output_path: str) -> bool:
        """処理レポートのエクスポート"""
        try:
            report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_stats": self.get_processing_stats(),
                "phase4_available": PHASE4_AVAILABLE,
                "modules_available": {
                    "excel_processing": EXCEL_PROCESSING_AVAILABLE,
                    "multimodal_processing": MULTIMODAL_PROCESSING_AVAILABLE,
                    "pdf_processing": PDF_PROCESSING_AVAILABLE
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"レポートエクスポートエラー: {e}")
            return False
