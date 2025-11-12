"""
OpenAI Vision処理
"""
import logging
import asyncio
from typing import Any, Dict, Optional


class OpenAIVisionProcessor:
    """OpenAI Vision処理クラス"""

    def __init__(self, openai_client_manager=None):
        self.logger = logging.getLogger(__name__)
        self.client_manager = openai_client_manager

    async def analyze_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Dict[str, Any]:
        """画像を GPT-4o-mini に渡して内容を分析"""
        if not self.client_manager:
            return {"error": "OpenAI client not available"}

        try:
            if prompt is None:
                prompt = """
この画像を詳細に分析し、以下の情報をJSON形式で返してください：
1. text_content: 画像内のテキストを全て抽出
2. description: 画像の内容を日本語で詳細に説明
3. category: 画像の種類（document, chart, diagram, photo, screenshotなど）
4. confidence: 分析の信頼度（0.0-1.0）
5. elements: 画像内の主要要素のリスト
"""

            client = self.client_manager.get_client()

            # 画像をbase64にエンコード
            import base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            response = await client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )

            result_text = response.choices[0].message.content

            return {
                "text": result_text,
                "raw_response": response,
                "model": "gpt-4.1-mini-2025-04-14",
                "success": True
            }

        except Exception as e:
            self.logger.error(f"OpenAI Vision解析エラー: {e}")
            return {
                "error": str(e),
                "success": False
            }

    def analyze_image_sync(self, image_bytes: bytes, prompt: Optional[str] = None) -> Dict[str, Any]:
        """同期版の画像分析"""
        try:
            # 非同期関数を同期的に実行
            return asyncio.run(self.analyze_image(image_bytes, prompt))
        except Exception as e:
            self.logger.error(f"同期画像分析エラー: {e}")
            return {
                "error": str(e),
                "success": False
            }
