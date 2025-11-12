"""
EasyOCR処理
"""
import logging
import easyocr
import numpy as np
import cv2
import os
import time
from pathlib import Path
from typing import Any, Dict, List


class EasyOCR:
    """EasyOCR処理クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reader = None
        self._model_dir = Path(os.environ.get("EASYOCR_MODEL_DIR", "model_cache/easyocr")).resolve()
        self._download_enabled = os.environ.get("EASYOCR_DISABLE_DOWNLOAD", "0") not in {"1", "true", "TRUE"}
        self._initialize()

    def _initialize(self):
        """EasyOCRの初期化（遅延読み込み）"""
        if self.reader is None:
            # EasyOCRの言語は環境変数で指定可能（カンマ区切り）
            langs_env = os.environ.get("EASYOCR_LANGS", "ja,en")
            try:
                langs = [l.strip() for l in langs_env.split(",") if l.strip()]
                model_dir = self._model_dir
                if self._download_enabled:
                    model_dir.mkdir(parents=True, exist_ok=True)
                self.reader = easyocr.Reader(
                    langs,
                    download_enabled=self._download_enabled,
                    model_storage_directory=str(model_dir),
                    user_network_directory=str(model_dir / "user_networks"),
                )
            except Exception:
                # 初期化失敗時はデフォルトでjaのみを試す
                try:
                    model_dir = self._model_dir
                    if self._download_enabled:
                        model_dir.mkdir(parents=True, exist_ok=True)
                    self.reader = easyocr.Reader(
                        ['ja'],
                        download_enabled=self._download_enabled,
                        model_storage_directory=str(model_dir),
                        user_network_directory=str(model_dir / "user_networks"),
                    )
                except Exception:
                    self.reader = None

    def perform_ocr(self, image_bytes: bytes) -> str:
        """EasyOCRでテキスト抽出"""
        try:
            start_time = time.perf_counter()
            if self.reader is None:
                self._initialize()
                if self.reader is None:
                    return ""

            # 画像をnumpy配列に変換
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # OCR実行
            results = self.reader.readtext(image)

            # 結果をテキストに結合（信頼度0.7以上）
            text_parts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.7:  # 信頼度フィルタ（0.5→0.7に変更）
                    text_parts.append(text)

            elapsed = time.perf_counter() - start_time
            self.logger.debug(
                "EasyOCR processed image (lang=%s, download_enabled=%s) in %.3fs, results=%d",
                os.environ.get("EASYOCR_LANGS", "ja,en"),
                self._download_enabled,
                elapsed,
                len(text_parts),
            )

            return " ".join(text_parts)

        except Exception as e:
            self.logger.error(f"EasyOCRエラー: {e}")
            return ""

