"""
認証機能（軽量版 - POCモード用、Firebase依存なし）
"""
from typing import Optional
from src.app.config import AuthenticationError
from src.app.utils.logging import get_logger

logger = get_logger(__name__)

def initialize_firebase_app():
    """Firebaseアプリケーションを初期化（軽量版ではスキップ）"""
    # 軽量版ではFirebase認証を使用しない
    logger.info("軽量版: Firebase認証をスキップします（POCモード）")
    pass

def verify_user(email: str, password: str) -> Optional[dict]:
    """ユーザー認証を実行（軽量版では常に成功）"""
    # 軽量版では常に認証成功として扱う
    logger.info(f"軽量版: ユーザー認証をスキップします（POCモード）: {email}")
    return {
        "uid": "demo",
        "email": email,
        "display_name": "Demo User",
        "email_verified": True,
        "disabled": False,
    }

def get_user_info(user: dict) -> dict:
    """ユーザー情報を取得"""
    return user








