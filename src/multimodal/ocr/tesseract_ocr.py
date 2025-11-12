"""
Tesseract OCR処理
"""
import logging
import pytesseract
from typing import Any, Dict, List
from PIL import Image
import io


class TesseractOCR:
    """Tesseract OCR処理クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def perform_ocr(self, image_bytes: bytes) -> str:
        """TesseractでOCR実行"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang='jpn+eng')
            return text.strip()
        except Exception as e:
            self.logger.error(f"Tesseract OCRエラー: {e}")
            return ""

    def generate_image_metadata(self, image_bytes: bytes, ocr_text: str,
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """画像メタデータ生成"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size

            # 基本的な画像情報
            image_metadata = {
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height > 0 else 0,
                "ocr_text_length": len(ocr_text),
                "has_text": bool(ocr_text.strip()),
                "image_type": self._classify_image_type(ocr_text, width, height),
                "text_regions": self._detect_text_regions(ocr_text),
                "language": self._detect_language(ocr_text),
                "confidence": metadata.get("confidence", 0.0)
            }

            return image_metadata
        except Exception as e:
            self.logger.error(f"画像メタデータ生成エラー: {e}")
            return {}

    def _classify_image_type(self, ocr_text: str, width: int, height: int) -> str:
        """画像タイプ分類"""
        if not ocr_text.strip():
            return "no_text"

        # アスペクト比による分類
        aspect_ratio = width / height if height > 0 else 0

        if aspect_ratio > 2.0:
            return "wide_document"
        elif aspect_ratio < 0.5:
            return "tall_document"
        elif 0.8 <= aspect_ratio <= 1.2:
            return "square_document"
        else:
            return "standard_document"

    def _detect_text_regions(self, ocr_text: str) -> List[Dict[str, Any]]:
        """テキスト領域検出"""
        if not ocr_text.strip():
            return []

        lines = ocr_text.split('\n')
        regions = []

        for i, line in enumerate(lines):
            if line.strip():
                regions.append({
                    "line_number": i,
                    "text": line.strip(),
                    "length": len(line.strip()),
                    "has_numbers": any(c.isdigit() for c in line),
                    "has_kanji": any('\u4e00' <= c <= '\u9faf' for c in line)
                })

        return regions

    def _detect_language(self, ocr_text: str) -> str:
        """言語検出"""
        if not ocr_text.strip():
            return "unknown"

        # 簡単な言語検出
        has_kanji = any('\u4e00' <= c <= '\u9faf' for c in ocr_text)
        has_hiragana = any('\u3040' <= c <= '\u309f' for c in ocr_text)
        has_katakana = any('\u30a0' <= c <= '\u30ff' for c in ocr_text)

        if has_kanji or has_hiragana or has_katakana:
            return "japanese"
        elif any(c.isalpha() for c in ocr_text):
            return "english"
        else:
            return "mixed"

