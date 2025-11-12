"""OpenAIクライアントとマルチモーダル推論のヘルパー"""

import base64
import logging
import os
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover - optional dependency
    AsyncOpenAI = None  # type: ignore

from src.retry_utils import APIRetryHandler, RetryConfig


class OpenAIClientManager:
    """OpenAIクライアントの取得とキャッシュ"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._client: Optional[AsyncOpenAI] = None

    def get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client

        if AsyncOpenAI is None:
            raise RuntimeError("openai パッケージがインストールされていません")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY が設定されていません")

        self._client = AsyncOpenAI(api_key=api_key)
        self.logger.info("AsyncOpenAI client initialized")
        return self._client


class OpenAIVisionHelper:
    """GPT-4.1-mini（マルチモーダル）で画像内容を取得"""

    def __init__(self, client_manager: Optional[OpenAIClientManager] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.client_manager = client_manager or OpenAIClientManager()

        # リトライ設定
        self.retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=1.0,  # 1秒から開始
            max_delay=30.0,     # 最大30秒
            backoff_factor=2.0,  # 指数バックオフ
            jitter=True         # ジッター有効
        )
        self.retry_handler = APIRetryHandler(self.retry_config)

    async def analyze_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Dict[str, Any]:
        """画像を GPT-4.1-mini に渡して内容を要約（リトライ付き）"""

        async def _single_api_call() -> Dict[str, Any]:
            """単一のAPI呼び出し"""
            client = self.client_manager.get_client()

            base_prompt = (
                "以下の画像はExcelシートをレンダリングしたものです。表の構造、注釈、フロー図"
                "、条件分岐など人間が読み取るべき要点を箇条書きで説明してください。"
                "主要なセルのラベルや矢印の意味も含めてください。"
            )

            final_prompt = prompt or base_prompt

            # 画像をbase64エンコード
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            try:
                response = await client.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": final_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1
                )

                text_content = response.choices[0].message.content.strip()

                return {
                    "text": text_content,
                    "raw_response": response,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }

            except Exception as e:
                self.logger.warning(f"OpenAI API call failed: {e}")
                raise e

        # リトライ付きで実行
        try:
            result = await self.retry_handler.execute_with_retry_async(_single_api_call)
            return result
        except Exception as e:
            # 最終的な失敗
            self.logger.error(f"OpenAI Vision API failed after retries: {e}")
            return {
                "text": "",
                "error": str(e),
                "raw_response": None,
                "tokens_used": 0
            }

    async def analyze_batch_async(self, images: List[bytes], prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """画像のバッチ処理を非同期で実行（リトライ付き）"""
        results: List[Dict[str, Any]] = []
        for image_bytes in images:
            try:
                result = await self.analyze_image(image_bytes, prompt=prompt)
                results.append(result)
            except Exception as exc:
                self.logger.warning("OpenAI vision call failed: %s", exc)
                results.append({"text": "", "error": str(exc), "raw_response": None, "tokens_used": 0})
        return results

    def analyze_batch(self, images: List[bytes], prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """画像のバッチ処理を同期で実行（後方互換性のため）"""
        import asyncio

        async def _run_batch():
            return await self.analyze_batch_async(images, prompt)

        try:
            # 新しいイベントループを作成して同期的に実行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_run_batch())
        finally:
            loop.close()


def run_coroutine_sync(coro):
    """Run an async coroutine in a synchronous context and return its result.

    This helper handles the case where an asyncio event loop is already running
    (e.g., invoked from an async environment). In that case the coroutine is
    executed in a separate thread using ``asyncio.run`` to avoid event-loop
    collisions. Otherwise it uses ``asyncio.run`` directly.
    """
    import asyncio
    import threading

    try:
        # If there is a running loop in this thread, run the coroutine in a
        # separate thread to avoid "asyncio.run() cannot be called from a
        # running event loop" errors.
        try:
            loop = asyncio.get_event_loop()
            running = loop.is_running()
        except RuntimeError:
            running = False

        if running:
            result_container = {}
            exc_container = {}

            def _target():
                try:
                    res = asyncio.run(coro)
                    result_container['res'] = res
                except Exception as e:  # capture and re-raise in caller thread
                    exc_container['e'] = e

            t = threading.Thread(target=_target)
            t.start()
            t.join()

            if 'e' in exc_container:
                raise exc_container['e']
            return result_container.get('res')

        # No running loop in this thread: safe to run directly
        return asyncio.run(coro)

    except Exception:
        # Best-effort: if anything goes wrong fall back to creating a fresh loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.close()
            except Exception:
                pass


