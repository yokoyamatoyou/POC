"""アクセス制御 (ACL) に関する共通ヘルパー"""

from __future__ import annotations

import os
from dataclasses import dataclass


def is_poc_mode() -> bool:
    """POCモードかどうかを判定"""
    return os.getenv("POC_MODE", "true").lower() == "true"


def acl_mode_message() -> str:
    """POCモード/本番モードに応じた案内文を返す"""
    if is_poc_mode():
        return "POCモード: 現在は誰でもACLを編集できます（本番では管理者限定予定）。"
    return "本番モード: ACL編集は管理者のみ許可されています。"








