"""
LLM補正処理
"""
import logging
from typing import Any, Dict, List


class LLMCorrection:
    """LLM補正処理クラス"""

    def __init__(self, openai_client_manager=None):
        self.logger = logging.getLogger(__name__)
        self.openai_client_manager = openai_client_manager

    def should_use_llm_correction(self, ocr_text: str) -> bool:
        """LLM補正を使うかどうかの判定"""
        if not ocr_text.strip():
            return False

        # 短すぎるテキストは補正不要
        if len(ocr_text.strip()) < 10:
            return False

        # 明らかな誤認識パターンをチェック
        suspicious_patterns = [
            # 数字・文字の混同
            'O', '0', 'o',  # 大文字O、数字0、小文字o
            'I', 'l', '1', '|',  # 大文字I、小文字l、数字1、縦線
        ]

        # 複数の疑わしいパターンが含まれている場合に補正使用
        suspicious_count = 0
        for pattern in suspicious_patterns:
            suspicious_count += ocr_text.count(pattern)

        return suspicious_count >= 3  # 3つ以上の疑わしいパターンがあれば補正使用

    def correct_ocr_with_llm(self, ocr_text: str) -> Dict[str, Any]:
        """LLMでOCR結果を文脈的に補正"""
        try:
            if not self.openai_client_manager:
                return {
                    "original_text": ocr_text,
                    "corrected_text": ocr_text,
                    "corrections_applied": False,
                    "error": "OpenAI client not available"
                }

            prompt = f"""
以下のOCR結果を文脈的に分析し、明らかな誤認識を修正してください。

OCR結果:
{ocr_text}

修正ルール:
1. 数字と文字の混同（Oと0、Iと1、lと1など）を文脈から判断して修正
2. 日本語のよくあるOCR誤認識（釦→金、口→日など）を修正
3. 記号の全角・半角統一（括弧、カンマ、ピリオドなど）
4. 文脈から判断できる明らかな誤りを修正
5. 不確実な場合は元のテキストを保持
6. 修正した箇所は [修正前→修正後] の形式でコメント

修正結果:
"""

            # 同期クライアントを使用
            client = self.openai_client_manager.get_client()

            response = client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは日本語文書のOCR結果を文脈的に分析し、誤認識を修正する専門家です。数字と文字の混同、日本語の誤認識、記号の統一を正確に行ってください。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            corrected_text = response.choices[0].message.content.strip()

            # 修正が適用されたかチェック
            corrections_applied = ocr_text != corrected_text

            return {
                "original_text": ocr_text,
                "corrected_text": corrected_text,
                "corrections_applied": corrections_applied,
                "confidence": 0.8 if corrections_applied else 0.9
            }

        except Exception as e:
            self.logger.error(f"LLM補正エラー: {e}")
            return {
                "original_text": ocr_text,
                "corrected_text": ocr_text,
                "corrections_applied": False,
                "error": str(e)
            }
