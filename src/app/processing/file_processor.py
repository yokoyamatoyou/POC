"""
ファイル処理機能
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from src.app.config import (
    FileProcessingError,
    SUPPORTED_FILE_TYPES,
    ERROR_MESSAGES,
    FILE_SIZE_LIMITS_MB,
)
from src.app.utils.logging import get_logger
from src.app.utils.metrics import metrics_collector
from src.multimodal import MultimodalProcessor, ProcessingMode, ContentType
from src.processing_router import ProcessingRouter, ProcessingMethod
from src.integrated_processor import IntegratedProcessor

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {
    ext
    for extensions in SUPPORTED_FILE_TYPES.values()
    for ext in extensions
}

_TEXT_EXTENSIONS = {".txt", ".md"}
_SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}
_WORD_EXTENSIONS = {".docx", ".doc"}
_PDF_EXTENSIONS = {".pdf"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

_DEFAULT_CONTENT_TYPES = {
    **{ext: ContentType.TEXT_ONLY for ext in _TEXT_EXTENSIONS},
    **{ext: ContentType.MIXED for ext in _SPREADSHEET_EXTENSIONS | _WORD_EXTENSIONS | _PDF_EXTENSIONS},
    **{ext: ContentType.IMAGE_RICH for ext in _IMAGE_EXTENSIONS},
    ".csv": ContentType.TEXT_ONLY,
    ".tsv": ContentType.TEXT_ONLY,
    ".json": ContentType.TEXT_ONLY,
    ".yaml": ContentType.TEXT_ONLY,
    ".yml": ContentType.TEXT_ONLY,
    ".xml": ContentType.TEXT_ONLY,
    ".py": ContentType.TEXT_ONLY,
    ".java": ContentType.TEXT_ONLY,
    ".bas": ContentType.TEXT_ONLY,
    ".cls": ContentType.TEXT_ONLY,
    ".vba": ContentType.TEXT_ONLY,
}

_PROCESSING_MODE_MAP = {
    ProcessingMethod.TEXT_ONLY: ProcessingMode.FAST,
    ProcessingMethod.MULTIMODAL_BASIC: ProcessingMode.BALANCED,
    ProcessingMethod.MULTIMODAL_ADVANCED: ProcessingMode.ACCURATE,
    ProcessingMethod.MULTIMODAL_HEAVY: ProcessingMode.ACCURATE,
    ProcessingMethod.IMAGE_HEAVY: ProcessingMode.ACCURATE,
    ProcessingMethod.OCR_FOCUSED: ProcessingMode.ACCURATE,
}


def _determine_content_type(
    file_extension: str,
    processing_parameters: Dict[str, Any],
) -> ContentType:
    """処理ルートと拡張子からコンテンツタイプを推定"""
    if file_extension in _DEFAULT_CONTENT_TYPES:
        return _DEFAULT_CONTENT_TYPES[file_extension]

    image_count = processing_parameters.get("image_count", 0)
    image_ratio = processing_parameters.get("image_ratio", 0.0)

    if image_count == 0:
        return ContentType.TEXT_ONLY

    if image_ratio >= 0.6:
        return ContentType.IMAGE_RICH

    return ContentType.MIXED


def _build_chunk_metadata(
    base_metadata: Dict[str, Any],
    result_metadata: Dict[str, Any],
    chunk_metadata: Optional[Dict[str, Any]],
    route,
    file_path: Path,
    file_size: int,
    processing_mode: ProcessingMode,
    processing_parameters: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """チャンク単位のメタデータを統合"""

    merged_metadata: Dict[str, Any] = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_extension": file_path.suffix.lower(),
        "file_size": file_size,
        "chunk_index": index,
        "processing_mode": processing_mode.value,
        "processing_method": route.selected_method.value,
        "ocr_method": route.ocr_method.value,
        "route_confidence": route.confidence,
        "route_estimated_time": route.estimated_time,
        "processing_parameters": dict(processing_parameters),
        "processing_route": {
            "method": route.selected_method.value,
            "ocr_method": route.ocr_method.value,
            "confidence": route.confidence,
            "estimated_time": route.estimated_time,
        },
    }

    merged_metadata.update(result_metadata)

    if chunk_metadata:
        merged_metadata.update(chunk_metadata)

    for key, value in base_metadata.items():
        merged_metadata.setdefault(key, value)

    return merged_metadata

def process_file_async(
    file_path: Path, 
    base_meta: Dict[str, Any], 
    client: OpenAI, 
    temp_dir: Path
) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    """ファイルを非同期で処理"""

    operation_name = f"process_file_{file_path.name}"
    metrics_collector.start_timer(operation_name)

    try:
        logger.info(f"ファイル処理開始: {file_path}")

        temp_dir.mkdir(parents=True, exist_ok=True)

        if not file_path.exists() or not file_path.is_file():
            raise FileProcessingError(ERROR_MESSAGES.get("file_not_found", "ファイルが見つかりません"))

        file_extension = file_path.suffix.lower()
        file_size = file_path.stat().st_size

        if file_size == 0:
            raise FileProcessingError("ファイルが空です")

        if file_extension not in _SUPPORTED_EXTENSIONS:
            raise FileProcessingError(ERROR_MESSAGES.get("invalid_file_type", "サポート対象外のファイル形式です"))

        size_limit_mb = FILE_SIZE_LIMITS_MB.get(file_extension)
        if size_limit_mb and file_size > size_limit_mb * 1024 * 1024:
            raise FileProcessingError(
                f"ファイルサイズが上限({size_limit_mb}MB)を超えています"
            )


        router = ProcessingRouter()
        route = router.route_document(file_path)
        processing_parameters = dict(route.processing_parameters or {})

        # 画像単体の誤検知に備えて処理モードを補正
        if file_extension in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        }:
            processing_parameters.setdefault("enable_image_analysis", True)
            processing_parameters.setdefault("enable_ocr", True)
            processing_parameters["method"] = ProcessingMethod.IMAGE_HEAVY.value
            processing_parameters.setdefault("chunk_size", 100)
            processing_parameters.setdefault("overlap", 10)

        config_defaults = _get_processing_config(file_path)
        chunk_size = int(processing_parameters.get("chunk_size", config_defaults["chunk_size"]))
        chunk_overlap = int(
            processing_parameters.get("overlap", config_defaults["chunk_overlap"])
        )

        processing_mode = _PROCESSING_MODE_MAP.get(route.selected_method, ProcessingMode.BALANCED)
        content_type = _determine_content_type(file_extension, processing_parameters)

        processor = MultimodalProcessor(processing_mode=processing_mode)
        result = processor.process_file(
            str(file_path),
            mode=processing_mode.value,
            content_type=content_type.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not result.success:
            error_message = result.error_message or ERROR_MESSAGES.get("processing_failed", "処理に失敗しました")
            raise FileProcessingError(error_message)

        result_metadata = dict(result.metadata or {})

        error_message = result.error_message or result_metadata.get("error")
        if error_message:
            raise FileProcessingError(str(error_message))

        if result_metadata.get("file_type") == "unknown" and file_extension not in _DEFAULT_CONTENT_TYPES:
            raise FileProcessingError(ERROR_MESSAGES.get("invalid_file_type", "サポート対象外のファイル形式です"))

        processed_chunks: List[Dict[str, Any]] = []
        base_metadata = dict(base_meta or {})

        for index, chunk in enumerate(result.chunks):
            chunk_text = chunk.get("content") or chunk.get("text") or ""
            chunk_metadata = _build_chunk_metadata(
                base_metadata,
                result_metadata,
                chunk.get("metadata"),
                route,
                file_path,
                file_size,
                processing_mode,
                processing_parameters,
                index,
            )

            processed_chunks.append(
                {
                    "content": chunk_text,
                    "text": chunk_text,
                    "chunk_index": index,
                    "file_name": file_path.name,
                    "file_type": chunk_metadata.get("file_type", file_extension.lstrip(".")),
                    "metadata": chunk_metadata,
                }
            )

        if not processed_chunks:
            logger.warning(f"処理結果が空です: {file_path}")
            raise FileProcessingError("チャンク生成に失敗しました")

        # Phase4: LLMによる要約・メタ強化
        if client is not None:
            try:
                integrated_processor = IntegratedProcessor(client=client)
                processed_chunks = integrated_processor.enhance_chunks(processed_chunks, result_metadata)
            except Exception as exc:
                logger.warning(f"チャンク強化処理（LLM要約）で例外が発生しました: {exc}")

        logger.info(
            "ファイル処理完了: %s (chunks=%d, mode=%s, method=%s)",
            file_path,
            len(processed_chunks),
            processing_mode.value,
            route.selected_method.value,
        )

        return processed_chunks, None

    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"ファイル処理エラー: {file_path} - {e}")
        raise FileProcessingError(f"{ERROR_MESSAGES.get('processing_failed', '処理に失敗しました')}: {e}") from e
    
    finally:
        metrics_collector.end_timer(operation_name)

def process_files_batch(
    files: List[Tuple[Path, Dict[str, Any]]], 
    base_meta: Dict[str, Any], 
    client: OpenAI, 
    temp_dir: Path, 
    max_workers: int = 4
) -> List[Tuple[Path, List[Dict[str, Any]], Optional[Path], Optional[str]]]:
    """ファイルをバッチで処理

    Returns:
        List[Tuple[Path, List[Dict[str, Any]], Optional[Path], Optional[str]]]:
            (処理対象パス, 生成チャンク, 一時ファイル, エラーメッセージ)
    """
    
    operation_name = "process_files_batch"
    metrics_collector.start_timer(operation_name)
    
    try:
        results: List[Tuple[Path, List[Dict[str, Any]], Optional[Path], Optional[str]]] = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 各ファイルの処理を非同期で実行
            future_to_file = {
                executor.submit(process_file_async, file_path, base_meta, client, temp_dir): file_path
                for file_path, _ in files
            }
            
            # 完了したタスクを収集
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    chunks, temp_file = future.result()
                    results.append((file_path, chunks, temp_file, None))
                except FileProcessingError as e:
                    logger.error(f"バッチ処理エラー: {file_path} - {e}")
                    results.append((file_path, [], None, str(e)))
                except Exception as e:
                    logger.error(f"バッチ処理エラー: {file_path} - {e}")
                    results.append((file_path, [], None, str(e)))
        
        logger.info(f"バッチ処理完了: {len(results)} files")
        return results
        
    except Exception as e:
        logger.error(f"バッチ処理エラー: {e}")
        raise FileProcessingError(f"バッチ処理に失敗しました: {e}")
    
    finally:
        metrics_collector.end_timer(operation_name)

def _process_text_file(file_path: Path, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """テキストファイルを処理"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本的なチャンク分割
        chunks = _split_text_into_chunks(content, 1000, 200)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                **base_meta,
                "content": chunk,
                "chunk_index": i,
                "file_type": "text",
                "file_name": file_path.name
            })
        
        return processed_chunks
        
    except Exception as e:
        logger.error(f"テキストファイル処理エラー: {e}")
        return []

def _process_excel_file(file_path: Path, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Excelファイルを処理（MultimodalProcessorに委譲）"""
    try:
        processor = MultimodalProcessor(processing_mode=ProcessingMode.ACCURATE)
        result = processor.process_excel(str(file_path), "")

        if result.metadata.get("error"):
            raise FileProcessingError(result.metadata.get("error"))

        chunks: List[Dict[str, Any]] = []
        excel_chunks = result.metadata.get("excel_chunks", [])

        for chunk_meta in excel_chunks:
            content = chunk_meta.get("content", "")
            metadata = chunk_meta.get("metadata", {})
            if not content.strip():
                continue
            chunk_obj = {
                "content": content,
                "text": content,
                "chunk_index": len(chunks),
                "file_name": file_path.name,
                "file_type": metadata.get("file_type", "excel"),
                "metadata": {
                    **base_meta,
                    **metadata,
                    "processing_mode": result.metadata.get("processing_mode"),
                    "processing_time": result.processing_time,
                },
            }
            chunks.append(chunk_obj)

        if not chunks:
            raise FileProcessingError("Excelから有効なデータを抽出できませんでした")

        return chunks
    except Exception as exc:
        logger.error(f"Excelファイル処理エラー: {exc}")
        return []

def _process_pdf_file(file_path: Path, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """PDFファイルを処理"""
    # 既存のPDFProcessorを使用
    try:
        from src.pdf_processor import PDFProcessor
        processor = PDFProcessor()
        result = processor.process_pdf(str(file_path))
        
        processed_chunks = []
        for i, block in enumerate(result.text_blocks):
            processed_chunks.append({
                **base_meta,
                "content": block.text,
                "chunk_index": i,
                "file_type": "pdf",
                "file_name": file_path.name,
                "page_number": block.page_number,
                "block_type": block.block_type
            })
        
        return processed_chunks
        
    except Exception as e:
        logger.error(f"PDFファイル処理エラー: {e}")
        return []

def _process_word_file(file_path: Path, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Wordファイルを処理"""
    # 既存のWordProcessorを使用
    try:
        from src.word_processor import WordProcessor
        processor = WordProcessor()
        result = processor.process_word(str(file_path))
        
        processed_chunks = []
        for i, chunk in enumerate(result.chunks):
            processed_chunks.append({
                **base_meta,
                "content": chunk.content,
                "chunk_index": i,
                "file_type": "word",
                "file_name": file_path.name,
                "paragraph_index": chunk.paragraph_index
            })
        
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Wordファイル処理エラー: {e}")
        return []

def _get_processing_config(file_path: Path) -> Dict[str, Any]:
    """ファイル形式に応じた処理設定を取得"""
    file_extension = file_path.suffix.lower()
    
    # ファイル形式別の最適なチャンク分けルール
    configs = {
        # テキストファイル
        '.txt': {
            'mode': 'text',
            'content_type': 'text',
            'file_type': 'text',
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        '.md': {
            'mode': 'text',
            'content_type': 'markdown',
            'file_type': 'markdown',
            'chunk_size': 800,
            'chunk_overlap': 150
        },
        # Excelファイル
        '.xlsx': {
            'mode': 'excel',
            'content_type': 'spreadsheet',
            'file_type': 'excel',
            'chunk_size': 500,
            'chunk_overlap': 100
        },
        '.xls': {
            'mode': 'excel',
            'content_type': 'spreadsheet',
            'file_type': 'excel',
            'chunk_size': 500,
            'chunk_overlap': 100
        },
        # PDFファイル
        '.pdf': {
            'mode': 'pdf',
            'content_type': 'document',
            'file_type': 'pdf',
            'chunk_size': 1200,
            'chunk_overlap': 250
        },
        # Wordファイル
        '.docx': {
            'mode': 'docx',
            'content_type': 'document',
            'file_type': 'docx',
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        '.doc': {
            'mode': 'docx',
            'content_type': 'document',
            'file_type': 'docx',
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        # 画像ファイル
        '.jpg': {
            'mode': 'image',
            'content_type': 'image',
            'file_type': 'image',
            'chunk_size': 1,
            'chunk_overlap': 0
        },
        '.png': {
            'mode': 'image',
            'content_type': 'image',
            'file_type': 'image',
            'chunk_size': 1,
            'chunk_overlap': 0
        },
        # JSONファイル
        '.json': {
            'mode': 'json',
            'content_type': 'structured',
            'file_type': 'json',
            'chunk_size': 800,
            'chunk_overlap': 100
        },
        # CSVファイル
        '.csv': {
            'mode': 'csv',
            'content_type': 'tabular',
            'file_type': 'csv',
            'chunk_size': 600,
            'chunk_overlap': 100
        },
        '.tsv': {
            'mode': 'csv',
            'content_type': 'tabular',
            'file_type': 'tsv',
            'chunk_size': 600,
            'chunk_overlap': 100
        },
        '.json': {
            'mode': 'json',
            'content_type': 'structured',
            'file_type': 'json',
            'chunk_size': 800,
            'chunk_overlap': 100
        },
        '.yaml': {
            'mode': 'yaml',
            'content_type': 'structured',
            'file_type': 'yaml',
            'chunk_size': 800,
            'chunk_overlap': 100
        },
        '.yml': {
            'mode': 'yaml',
            'content_type': 'structured',
            'file_type': 'yaml',
            'chunk_size': 800,
            'chunk_overlap': 100
        },
        '.xml': {
            'mode': 'xml',
            'content_type': 'structured',
            'file_type': 'xml',
            'chunk_size': 800,
            'chunk_overlap': 100
        },
        '.py': {
            'mode': 'code',
            'content_type': 'text',
            'file_type': 'code',
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        '.java': {
            'mode': 'code',
            'content_type': 'text',
            'file_type': 'code',
            'chunk_size': 1200,
            'chunk_overlap': 200
        },
        '.bas': {
            'mode': 'code',
            'content_type': 'text',
            'file_type': 'code',
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        '.cls': {
            'mode': 'code',
            'content_type': 'text',
            'file_type': 'code',
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        '.vba': {
            'mode': 'code',
            'content_type': 'text',
            'file_type': 'code',
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
    }
    
    return configs.get(file_extension, {
        'mode': 'text',
        'content_type': 'text',
        'file_type': 'unknown',
        'chunk_size': 1000,
        'chunk_overlap': 200
    })

def _split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """テキストをチャンクに分割"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # オーバーラップを考慮してチャンクを取得
        if end < len(text):
            # 文の境界で分割を試行
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            
            if last_period > start:
                end = last_period + 1
            elif last_newline > start:
                end = last_newline + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks







