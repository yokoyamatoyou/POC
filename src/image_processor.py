"""
画像ファイルをMultimodalProcessor経由で解析し、統一チャンクを生成するコンポーネント。
"""

from typing import List, Dict

from src.multimodal.processor import MultimodalProcessor, ProcessingMode


class ImageProcessor:
    """画像ファイルをMultimodalProcessorで処理しチャンク配列を返す"""

    def __init__(self, processing_mode: ProcessingMode = ProcessingMode.BALANCED) -> None:
        self.processor = MultimodalProcessor(processing_mode=processing_mode)

    def process(self, file_path: str) -> List[Dict]:
        processed = self.processor.process_image(file_path, "")
        base_meta = {
            k: v
            for k, v in processed.metadata.items()
            if k not in {"prechunked_chunks", "image_chunks", "structured_chunks"}
        }

        chunks: List[Dict] = []
        prechunked = processed.metadata.get("prechunked_chunks") or []

        for index, chunk in enumerate(prechunked):
            content = chunk.get("content") or processed.text
            if not content:
                continue

            chunk_metadata = dict(base_meta)
            chunk_metadata.update(chunk.get("metadata", {}))
            chunk_metadata.setdefault("chunk_index", index)
            chunk_metadata.setdefault("content_type", "image")

            chunks.append({
                "content": content,
                "metadata": chunk_metadata,
            })

        if not chunks and processed.text:
            fallback_meta = dict(base_meta)
            fallback_meta.setdefault("content_type", "image")
            chunks.append({
                "content": processed.text,
                "metadata": fallback_meta,
            })

        return chunks
