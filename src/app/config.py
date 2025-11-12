"""
アプリケーション設定・定数・例外クラス（軽量版）
"""
from typing import Dict, Any
import os
from pathlib import Path

# --- 定数 ---
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_WORKERS = 4
DEFAULT_TEMP_DIR = Path("temp_processing")
DEFAULT_PDF_TEMP_DIR = Path("temp_pdf")

# --- UI/UX定数（視覚的設計の統一） ---
UI_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e", 
    "success": "#2ca02c",
    "warning": "#d62728",
    "info": "#9467bd",
    "light": "#f8f9fa",
    "dark": "#343a40"
}

UI_ICONS = {
    "search": "",
    "upload": "",
    "chat": "",
    "knowledge": "",
    "settings": "",
    "dashboard": "",
    "pdf": "",
    "excel": "",
    "word": "",
    "image": "",
    "error": "[error]",
    "success": "[ok]",
    "warning": "[!]",
    "info": "[i]",
    "admin": "",
}

# --- ファイルタイプ定数 ---
SUPPORTED_FILE_TYPES = {
    "text": [".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".html"],
    "excel": [".xlsx", ".xls", ".xlsm", ".xlsb"],
    "word": [".docx", ".doc"],
    "pdf": [".pdf"],
    "image": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"],
    "code": [".py", ".java", ".bas", ".cls", ".vba"],
    "archive": [".rar", ".zip", ".7z"]
}

# --- エラーメッセージ定数 ---
ERROR_MESSAGES = {
    "file_not_found": "ファイルが見つかりません",
    "invalid_file_type": "サポートされていないファイルタイプです",
    "processing_failed": "ファイル処理に失敗しました",
    "authentication_failed": "認証に失敗しました",
    "network_error": "ネットワークエラーが発生しました",
    "unknown_error": "不明なエラーが発生しました"
}

# --- カスタム例外クラス ---
class RAGApplicationError(Exception):
    """RAGアプリケーションの基底例外クラス"""
    pass

class FileProcessingError(RAGApplicationError):
    """ファイル処理関連の例外"""
    pass

class AuthenticationError(RAGApplicationError):
    """認証関連の例外"""
    pass

class ConfigurationError(RAGApplicationError):
    """設定関連の例外"""
    pass

class NetworkError(RAGApplicationError):
    """ネットワーク関連の例外"""
    pass

# --- ファイル処理容量制限（本番対応） ---
FILE_SIZE_LIMITS_MB = {
    ".docx": 50,    # Word: 50 MB
    ".doc": 50,
    ".xlsx": 100,   # Excel: 100 MB
    ".xls": 100,
    ".pdf": 200,    # PDF: 200 MB
}

EXCEL_ROW_LIMITS = {
    "warning": 10000,   # 警告: 10K行以上
    "error": 50000,     # エラー: 50K行以上
}

PDF_PAGE_LIMITS = {
    "warning": 500,     # 警告: 500ページ以上
    "error": 1000,      # エラー: 1000ページ以上
    "ocr_max": 100,     # OCR処理上限: 100ページ
}

DOCX_CHAR_LIMITS = {
    "max": 1000000,     # 最大: 100万文字
}

DOCX_IMAGE_LIMITS = {
    "max": 100,         # 最大処理画像数: 100個
}

# --- 設定取得関数 ---
def get_app_config() -> Dict[str, Any]:
    """アプリケーション設定を取得"""
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
        "max_workers": int(os.getenv("MAX_WORKERS", DEFAULT_MAX_WORKERS)),
        "temp_dir": Path(os.getenv("TEMP_DIR", DEFAULT_TEMP_DIR)),
        "pdf_temp_dir": Path(os.getenv("PDF_TEMP_DIR", DEFAULT_PDF_TEMP_DIR)),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "firebase_config": os.getenv("FIREBASE_CONFIG"),
        "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
        # POC専用モード / デフォルト無認証: デフォルトで認証を無効にしてトップ画面を表示します。
        # 環境変数で明示的に制御する場合は POC_MODE を使用してください。
        "poc_mode": os.getenv("POC_MODE", "true").lower() == "true"
    }








