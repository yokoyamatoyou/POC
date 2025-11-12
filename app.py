"""
軽量版メインアプリケーション（POC用）
"""
import os
import sys
from pathlib import Path
import streamlit as st
from openai import OpenAI

# パッケージ解決用に親ディレクトリをPythonパスへ追加
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 設定とユーティリティのインポート
from src.app.config import get_app_config
from src.app.utils.logging import setup_logging, get_logger
from src.app.auth import initialize_firebase_app, verify_user
from src.app.ui import main_app_ui

# ログ設定
logger = setup_logging(log_level="INFO", log_file=None)

# Streamlitのページ設定
st.set_page_config(
    layout="wide",
    page_title="統合ナレッジ検索システム（軽量版）",
    page_icon=None,
)


def _apply_user_context_to_vector_store(vector_store) -> None:
    if not vector_store:
        return

    context = st.session_state.get('user_context') or {}
    tenant_id = context.get('tenant_id', 'tenant_001')
    departments = context.get('departments', [])
    if isinstance(departments, str):
        departments = [dept.strip() for dept in departments.split(',') if dept.strip()]

    role_level = context.get('role_level', 0)
    role_label = context.get('role_label', '一般')

    user_info = st.session_state.get('user_info') or {}
    user_id = user_info.get('uid') or st.session_state.get('user_email') or 'demo'

    try:
        vector_store.set_user_context(
            tenant_id,
            departments,
            role_level,
            role_label,
            user_id=user_id,
        )
    except Exception as exc:
        logger.warning(f"ユーザーコンテキストの適用に失敗しました: {exc}")


def initialize_app():
    """アプリケーションを初期化"""
    try:
        # 設定を取得
        config = get_app_config()
        
        # POCモード: Firebase認証をスキップ
        if config.get('poc_mode', True):
            pass
        else:
            # Firebase初期化（通常モード）
            initialize_firebase_app()
        
        # OpenAIクライアント初期化
        if not config.get('openai_api_key'):
            st.error("OpenAI APIキーが設定されていません。環境変数OPENAI_API_KEYを設定してください。")
            st.stop()
        
        client = OpenAI(api_key=config['openai_api_key'])
        
        # セッション状態の初期化
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.user_email = None
            st.session_state.user_info = None
        
        return client, config
        
    except Exception as e:
        st.error(f"アプリケーション初期化エラー: {e}")
        logger.error(f"アプリケーション初期化エラー: {e}")
        st.stop()


def get_vector_store(client: OpenAI):
    """ベクトルストアを取得"""
    try:
        from src.vector_store import VectorStoreManager
        return VectorStoreManager(client)
    except ImportError as e:
        st.error(f"ベクトルストアモジュールのインポートに失敗しました: {e}")
        logger.error(f"ベクトルストアインポートエラー: {e}")
        return None


def get_rag_engine(vector_store, client: OpenAI):
    """RAGエンジンを取得"""
    try:
        from src.rag_engine import RAGEngine
        return RAGEngine(vector_store, client)
    except ImportError as e:
        st.error(f"RAGエンジンモジュールのインポートに失敗しました: {e}")
        logger.error(f"RAGエンジンインポートエラー: {e}")
        return None


def main():
    """メイン関数"""
    try:
        # アプリケーション初期化
        client, config = initialize_app()
        
        # POCモード: 認証をスキップしてデモユーザーで実行
        if config.get('poc_mode', True):
            st.session_state["poc_mode_active"] = True
            if 'initialized' not in st.session_state:
                st.session_state.initialized = True
            if not st.session_state.get('user_email'):
                st.session_state.user_email = "poc_demo@example.com"
                st.session_state.user_info = {"uid": "poc_demo", "email": st.session_state.user_email}
                st.session_state.user_context = {
                    "tenant_id": "tenant_001",
                    "departments": [],
                    "role_level": 0,
                    "role_label": "一般"
                }
        else:
            st.session_state["poc_mode_active"] = False
            # ログイン状態の確認（通常モード）
            if not st.session_state.get('user_email'):
                st.info("通常モードでは認証が必要です。POCモードを使用する場合は環境変数POC_MODE=trueを設定してください。")
                return
        
        # OpenAIクライアントと設定をセッションに保持
        st.session_state.openai_client = client
        st.session_state.app_config = config

        # ベクトルストアを初期化（セッション状態を優先）
        vector_store = st.session_state.get('vector_store')
        if vector_store is None:
            vector_store = get_vector_store(client)
            if not vector_store:
                st.error("ベクトルストアの初期化に失敗しました。")
                return
            st.session_state.vector_store = vector_store
        else:
            # 常に最新のOpenAIクライアントを反映
            vector_store.client = client
            st.session_state.vector_store = vector_store

        _apply_user_context_to_vector_store(vector_store)

        # RAGエンジンを初期化（セッション状態を優先）
        rag_engine = st.session_state.get('rag_engine')
        if rag_engine is None:
            rag_engine = get_rag_engine(vector_store, client)
            if not rag_engine:
                st.error("RAGエンジンの初期化に失敗しました。")
                return
            st.session_state.rag_engine = rag_engine
        else:
            rag_engine.vector_store = vector_store
            rag_engine.client = client
            st.session_state.rag_engine = rag_engine
        
        _apply_user_context_to_vector_store(vector_store)
        
        # メインUI
        main_app_ui()
        
    except Exception as e:
        st.error(f"アプリケーションエラー: {e}")
        logger.error(f"アプリケーションエラー: {e}")


if __name__ == "__main__":
    main()








