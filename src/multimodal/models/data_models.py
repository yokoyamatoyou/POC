"""
MultimodalProcessor データモデル定義
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ContentType(Enum):
    """コンテンツタイプ"""
    TEXT_ONLY = "text_only"
    IMAGE_RICH = "image_rich"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProcessingMode(Enum):
    """処理モード"""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class ProcessedContent:
    """処理済みコンテンツ

    互換性対応: 既存コードベースでは `text_content`, `image_texts`, `content_type`,
    `token_count`, `processing_mode` などのキーワードで生成している箇所があるため、
    それらを受け取って新しいフィールドにマッピングする互換コンストラクタを提供します。
    """

    def __init__(
        self,
        text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        processing_time: float = 0.0,
        confidence: float = 0.0,
        *,
        # 互換キーワード（旧コード）
        text_content: Optional[str] = None,
        image_texts: Optional[List[Any]] = None,
        content_type: Optional[ContentType] = None,
        token_count: Optional[int] = None,
        processing_mode: Optional[ProcessingMode] = None,
        **kwargs: Any,
    ) -> None:
        # 新しい主要フィールド
        self.text: str = text_content if text_content is not None else text
        self.metadata: Dict[str, Any] = dict(metadata or {})
        # 互換情報をメタデータに保存
        if image_texts is not None:
            self.metadata.setdefault("image_texts", image_texts)
        if content_type is not None:
            # store enum value for serialization
            try:
                self.metadata.setdefault("content_type", content_type.value)
            except Exception:
                self.metadata.setdefault("content_type", str(content_type))
        if token_count is not None:
            self.metadata.setdefault("token_count", token_count)
        if processing_mode is not None:
            try:
                self.metadata.setdefault("processing_mode", processing_mode.value)
            except Exception:
                self.metadata.setdefault("processing_mode", str(processing_mode))

        # 既存の時間/信頼度フィールド
        self.processing_time: float = processing_time
        self.confidence: float = confidence
        # 互換プロパティ
        self._content_type: Optional[ContentType] = content_type
        self._token_count: Optional[int] = token_count
        self._processing_mode: Optional[ProcessingMode] = processing_mode

    @property
    def text_content(self) -> str:
        return self.text

    @property
    def image_texts(self) -> List[Any]:
        return list(self.metadata.get("image_texts", []))

    @property
    def content_type(self) -> Optional[ContentType]:
        return self._content_type

    @property
    def token_count(self) -> Optional[int]:
        return self._token_count

    @property
    def processing_mode(self) -> Optional[ProcessingMode]:
        return self._processing_mode


@dataclass
class MultimodalDocument:
    """マルチモーダル文書"""
    document_id: str
    content_type: ContentType
    text_content: str
    images: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalProcessingResult:
    """マルチモーダル処理結果"""
    success: bool = True
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    content: Optional[ProcessedContent] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

