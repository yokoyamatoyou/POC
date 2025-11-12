"""
ログ設定・エラーハンドリング（軽量版）
"""
import logging
import os
from typing import Optional
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """ログ設定を初期化"""
    
    # ログレベルを設定
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # ログフォーマットを設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ルートロガーを設定
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 既存のハンドラーをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # コンソールハンドラーを追加
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラーを追加（指定された場合）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """指定された名前のロガーを取得"""
    return logging.getLogger(name)

# デフォルトロガー
logger = get_logger(__name__)








