"""
メタデータ分離・永続化システム
RAG日本語検索アルゴリズムガイドに基づくメタデータ管理
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import sqlite3
import json
import logging
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class InternalMetadata:
    """データ内部メタデータ（文書内容に基づく情報）"""
    paragraph_index: Optional[int] = None
    style: Optional[str] = None
    is_caption: bool = False
    has_figure_ref: bool = False
    ref_solved_ratio: float = 0.0
    ocr_confidence_avg: float = 0.0
    text_extraction_ratio: float = 0.0
    total_images: int = 0
    total_tables: int = 0
    total_pages: int = 0
    font_info: Dict[str, Any] = field(default_factory=dict)
    outline: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ExternalMetadata:
    """データ外部メタデータ（ファイルシステム/処理情報）"""
    file_created_date: Optional[datetime] = None
    file_modified_date: Optional[datetime] = None
    file_size: int = 0
    file_path: str = ""
    processing_timestamp: Optional[datetime] = None
    tenant_id: str = ""
    department: str = ""
    importance_level: str = "normal"  # low, normal, high, urgent
    user_tags: List[str] = field(default_factory=list)
    access_acl: List[str] = field(default_factory=list)  # アクセス制御リスト

@dataclass
class PriorityMetadata:
    """ランキング/検索用優先メタデータ（検索エンジン用）"""
    document_type: str = ""
    file_modified_date: Optional[datetime] = None
    importance_level: str = "normal"
    ocr_confidence_avg: float = 0.0
    ref_solved_ratio: float = 0.0
    access_acl: List[str] = field(default_factory=list)
    department: str = ""
    tenant_id: str = ""

@dataclass
class SeparatedMetadata:
    """分離されたメタデータ統合クラス"""
    document_id: str
    chunk_id: str = ""
    internal: InternalMetadata = field(default_factory=InternalMetadata)
    external: ExternalMetadata = field(default_factory=ExternalMetadata)
    priority: PriorityMetadata = field(default_factory=PriorityMetadata)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict representation of the separated metadata."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "internal": {
                "paragraph_index": self.internal.paragraph_index,
                "style": self.internal.style,
                "is_caption": self.internal.is_caption,
                "has_figure_ref": self.internal.has_figure_ref,
                "ref_solved_ratio": self.internal.ref_solved_ratio,
                "ocr_confidence_avg": self.internal.ocr_confidence_avg,
                "text_extraction_ratio": self.internal.text_extraction_ratio,
                "total_images": self.internal.total_images,
                "total_tables": self.internal.total_tables,
                "total_pages": self.internal.total_pages,
                "font_info": self.internal.font_info,
                "outline": self.internal.outline,
                "annotations": self.internal.annotations,
            },
            "external": {
                "file_created_date": self.external.file_created_date.isoformat() if self.external.file_created_date else None,
                "file_modified_date": self.external.file_modified_date.isoformat() if self.external.file_modified_date else None,
                "file_size": self.external.file_size,
                "file_path": self.external.file_path,
                "processing_timestamp": self.external.processing_timestamp.isoformat() if self.external.processing_timestamp else None,
                "tenant_id": self.external.tenant_id,
                "department": self.external.department,
                "importance_level": self.external.importance_level,
                "user_tags": self.external.user_tags,
                "access_acl": self.external.access_acl,
            },
            "priority": {
                "document_type": self.priority.document_type,
                "file_modified_date": self.priority.file_modified_date.isoformat() if self.priority.file_modified_date else None,
                "importance_level": self.priority.importance_level,
                "ocr_confidence_avg": self.priority.ocr_confidence_avg,
                "ref_solved_ratio": self.priority.ref_solved_ratio,
                "access_acl": self.priority.access_acl,
                "department": self.priority.department,
                "tenant_id": self.priority.tenant_id,
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class MetadataSeparator:
    """メタデータ分離クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def separate_metadata(self, raw_metadata: Dict[str, Any], document_id: str, chunk_id: str = "") -> SeparatedMetadata:
        """
        生メタデータを分離して構造化

        Args:
            raw_metadata: 生メタデータ辞書
            document_id: 文書ID
            chunk_id: チャンクID（オプション）

        Returns:
            SeparatedMetadata: 分離されたメタデータ
        """
        try:
            # Internal Metadataの抽出
            internal = InternalMetadata(
                paragraph_index=raw_metadata.get("paragraph_index"),
                style=raw_metadata.get("style"),
                is_caption=raw_metadata.get("is_caption", False),
                has_figure_ref=raw_metadata.get("has_figure_ref", False),
                ref_solved_ratio=raw_metadata.get("ref_solved_ratio", 0.0),
                ocr_confidence_avg=raw_metadata.get("ocr_confidence_avg", 0.0),
                text_extraction_ratio=raw_metadata.get("text_extraction_ratio", 0.0),
                total_images=raw_metadata.get("total_images", 0),
                total_tables=raw_metadata.get("total_tables", 0),
                total_pages=raw_metadata.get("total_pages", 0),
                font_info=raw_metadata.get("font_info", {}),
                outline=raw_metadata.get("outline", []),
                annotations=raw_metadata.get("annotations", [])
            )

            # External Metadataの抽出
            external = ExternalMetadata(
                file_created_date=self._parse_datetime(raw_metadata.get("file_created_date")),
                file_modified_date=self._parse_datetime(raw_metadata.get("file_modified_date")),
                file_size=raw_metadata.get("file_size", 0),
                file_path=raw_metadata.get("file_path", ""),
                processing_timestamp=self._parse_datetime(raw_metadata.get("processing_timestamp")),
                tenant_id=raw_metadata.get("tenant_id", ""),
                department=raw_metadata.get("department", ""),
                importance_level=raw_metadata.get("importance_level", "normal"),
                user_tags=raw_metadata.get("user_tags", []),
                access_acl=raw_metadata.get("access_acl", [])
            )

            # Priority Metadataの抽出（検索エンジン用）
            priority = PriorityMetadata(
                document_type=raw_metadata.get("document_type", ""),
                file_modified_date=self._parse_datetime(raw_metadata.get("file_modified_date")),
                importance_level=raw_metadata.get("importance_level", "normal"),
                ocr_confidence_avg=raw_metadata.get("ocr_confidence_avg", 0.0),
                ref_solved_ratio=raw_metadata.get("ref_solved_ratio", 0.0),
                access_acl=raw_metadata.get("access_acl", []),
                department=raw_metadata.get("department", ""),
                tenant_id=raw_metadata.get("tenant_id", "")
            )

            return SeparatedMetadata(
                document_id=document_id,
                chunk_id=chunk_id,
                internal=internal,
                external=external,
                priority=priority
            )

        except Exception as e:
            self.logger.error(f"メタデータ分離エラー: {e}")
            # エラー時は空のメタデータを返す
            return SeparatedMetadata(
                document_id=document_id,
                chunk_id=chunk_id
            )

    def _parse_datetime(self, datetime_str: Any) -> Optional[datetime]:
        """日時文字列をdatetimeオブジェクトに変換"""
        if isinstance(datetime_str, datetime):
            return datetime_str
        elif isinstance(datetime_str, str):
            try:
                return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            except ValueError:
                return None
        return None

class MetadataPersistenceManager:
    """メタデータ永続化マネージャー"""

    def __init__(self, db_path: str = "metadata.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Internal Metadataテーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS internal_metadata (
                    document_id TEXT,
                    chunk_id TEXT,
                    paragraph_index INTEGER,
                    style TEXT,
                    is_caption BOOLEAN,
                    has_figure_ref BOOLEAN,
                    ref_solved_ratio REAL,
                    ocr_confidence_avg REAL,
                    text_extraction_ratio REAL,
                    total_images INTEGER,
                    total_tables INTEGER,
                    total_pages INTEGER,
                    font_info TEXT,
                    outline TEXT,
                    annotations TEXT,
                    PRIMARY KEY (document_id, chunk_id)
                )
            ''')

            # External Metadataテーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS external_metadata (
                    document_id TEXT,
                    chunk_id TEXT,
                    file_created_date TEXT,
                    file_modified_date TEXT,
                    file_size INTEGER,
                    file_path TEXT,
                    processing_timestamp TEXT,
                    tenant_id TEXT,
                    department TEXT,
                    importance_level TEXT,
                    user_tags TEXT,
                    access_acl TEXT,
                    PRIMARY KEY (document_id, chunk_id)
                )
            ''')

            # Priority Metadataテーブル（検索エンジン用）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS priority_metadata (
                    document_id TEXT,
                    chunk_id TEXT,
                    document_type TEXT,
                    file_modified_date TEXT,
                    importance_level TEXT,
                    ocr_confidence_avg REAL,
                    ref_solved_ratio REAL,
                    access_acl TEXT,
                    department TEXT,
                    tenant_id TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (document_id, chunk_id)
                )
            ''')

            # Metadata Indexテーブル（メタデータ管理用）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata_index (
                    document_id TEXT PRIMARY KEY,
                    total_chunks INTEGER,
                    last_updated TEXT,
                    metadata_version TEXT
                )
            ''')

            conn.commit()

    def save_metadata(self, metadata: SeparatedMetadata) -> bool:
        """
        分離されたメタデータを保存

        Args:
            metadata: 保存するメタデータ

        Returns:
            bool: 保存成功フラグ
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    # Internal Metadata保存
                    cursor.execute('''
                        INSERT OR REPLACE INTO internal_metadata
                        (document_id, chunk_id, paragraph_index, style, is_caption, has_figure_ref,
                         ref_solved_ratio, ocr_confidence_avg, text_extraction_ratio, total_images,
                         total_tables, total_pages, font_info, outline, annotations)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metadata.document_id, metadata.chunk_id,
                        metadata.internal.paragraph_index, metadata.internal.style,
                        metadata.internal.is_caption, metadata.internal.has_figure_ref,
                        metadata.internal.ref_solved_ratio, metadata.internal.ocr_confidence_avg,
                        metadata.internal.text_extraction_ratio, metadata.internal.total_images,
                        metadata.internal.total_tables, metadata.internal.total_pages,
                        json.dumps(metadata.internal.font_info, ensure_ascii=False),
                        json.dumps(metadata.internal.outline, ensure_ascii=False),
                        json.dumps(metadata.internal.annotations, ensure_ascii=False)
                    ))

                    # External Metadata保存
                    cursor.execute('''
                        INSERT OR REPLACE INTO external_metadata
                        (document_id, chunk_id, file_created_date, file_modified_date, file_size,
                         file_path, processing_timestamp, tenant_id, department, importance_level,
                         user_tags, access_acl)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metadata.document_id, metadata.chunk_id,
                        metadata.external.file_created_date.isoformat() if metadata.external.file_created_date else None,
                        metadata.external.file_modified_date.isoformat() if metadata.external.file_modified_date else None,
                        metadata.external.file_size, metadata.external.file_path,
                        metadata.external.processing_timestamp.isoformat() if metadata.external.processing_timestamp else None,
                        metadata.external.tenant_id, metadata.external.department,
                        metadata.external.importance_level,
                        json.dumps(metadata.external.user_tags, ensure_ascii=False),
                        json.dumps(metadata.external.access_acl, ensure_ascii=False)
                    ))

                    # Priority Metadata保存
                    cursor.execute('''
                        INSERT OR REPLACE INTO priority_metadata
                        (document_id, chunk_id, document_type, file_modified_date, importance_level,
                         ocr_confidence_avg, ref_solved_ratio, access_acl, department, tenant_id,
                         created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metadata.document_id, metadata.chunk_id,
                        metadata.priority.document_type,
                        metadata.priority.file_modified_date.isoformat() if metadata.priority.file_modified_date else None,
                        metadata.priority.importance_level,
                        metadata.priority.ocr_confidence_avg, metadata.priority.ref_solved_ratio,
                        json.dumps(metadata.priority.access_acl, ensure_ascii=False),
                        metadata.priority.department, metadata.priority.tenant_id,
                        metadata.created_at.isoformat(), metadata.updated_at.isoformat()
                    ))

                    # Metadata Index更新
                    cursor.execute('''
                        INSERT OR REPLACE INTO metadata_index
                        (document_id, total_chunks, last_updated, metadata_version)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        metadata.document_id, 1, metadata.updated_at.isoformat(), "1.0"
                    ))

                    conn.commit()
                    self.logger.info(f"メタデータ保存完了: {metadata.document_id}")
                    return True

            except sqlite3.Error as e:
                self.logger.warning(f"メタデータ保存エラー (試行 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.logger.error("メタデータ保存失敗 - すべてのリトライが失敗しました")
                    return False
                continue

            except Exception as e:
                self.logger.error(f"予期せぬエラー: {e}")
                return False

    def load_priority_metadata(self, document_id: str, chunk_id: str = "") -> Optional[Dict[str, Any]]:
        """
        検索用の優先メタデータをロード

        Args:
            document_id: 文書ID
            chunk_id: チャンクID（オプション）

        Returns:
            検索用メタデータ辞書
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if chunk_id:
                    cursor.execute('''
                        SELECT document_type, file_modified_date, importance_level,
                               ocr_confidence_avg, ref_solved_ratio, access_acl,
                               department, tenant_id
                        FROM priority_metadata
                        WHERE document_id = ? AND chunk_id = ?
                    ''', (document_id, chunk_id))
                else:
                    # 文書全体の優先メタデータを取得
                    cursor.execute('''
                        SELECT document_type, file_modified_date, importance_level,
                               ocr_confidence_avg, ref_solved_ratio, access_acl,
                               department, tenant_id
                        FROM priority_metadata
                        WHERE document_id = ?
                        ORDER BY updated_at DESC LIMIT 1
                    ''', (document_id,))

                row = cursor.fetchone()
                if row:
                    return {
                        "document_type": row[0],
                        "file_modified_date": row[1],
                        "importance_level": row[2],
                        "ocr_confidence_avg": row[3],
                        "ref_solved_ratio": row[4],
                        "access_acl": json.loads(row[5]) if row[5] else [],
                        "department": row[6],
                        "tenant_id": row[7]
                    }
                return None

        except Exception as e:
            self.logger.error(f"優先メタデータロードエラー: {e}")
            return None

    def load_all_metadata(self) -> List[SeparatedMetadata]:
        """Load all separated metadata entries from the DB and return as SeparatedMetadata objects."""
        results: List[SeparatedMetadata] = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT document_id FROM metadata_index')
                rows = cursor.fetchall()
                for (doc_id,) in rows:
                    # load internal/external/priority by joining tables
                    cursor.execute('''
                        SELECT i.chunk_id, i.paragraph_index, i.style, i.is_caption, i.has_figure_ref,
                               i.ref_solved_ratio, i.ocr_confidence_avg, i.text_extraction_ratio, i.total_images,
                               i.total_tables, i.total_pages, i.font_info, i.outline, i.annotations,
                               e.file_created_date, e.file_modified_date, e.file_size, e.file_path, e.processing_timestamp,
                               e.tenant_id, e.department, e.importance_level, e.user_tags, e.access_acl,
                               p.document_type, p.file_modified_date, p.importance_level, p.ocr_confidence_avg, p.ref_solved_ratio,
                               p.access_acl, p.department, p.tenant_id, p.created_at, p.updated_at
                        FROM internal_metadata i
                        LEFT JOIN external_metadata e ON i.document_id = e.document_id AND i.chunk_id = e.chunk_id
                        LEFT JOIN priority_metadata p ON i.document_id = p.document_id AND i.chunk_id = p.chunk_id
                        WHERE i.document_id = ?
                    ''', (doc_id,))
                    for row in cursor.fetchall():
                        chunk_id = row[0]
                        # parse fields into SeparatedMetadata
                        internal = InternalMetadata(
                            paragraph_index=row[1],
                            style=row[2],
                            is_caption=bool(row[3]),
                            has_figure_ref=bool(row[4]),
                            ref_solved_ratio=row[5] or 0.0,
                            ocr_confidence_avg=row[6] or 0.0,
                            text_extraction_ratio=row[7] or 0.0,
                            total_images=row[8] or 0,
                            total_tables=row[9] or 0,
                            total_pages=row[10] or 0,
                            font_info=json.loads(row[11]) if row[11] else {},
                            outline=json.loads(row[12]) if row[12] else [],
                            annotations=json.loads(row[13]) if row[13] else [],
                        )
                        external = ExternalMetadata(
                            file_created_date=self._parse_datetime(row[14]),
                            file_modified_date=self._parse_datetime(row[15]),
                            file_size=row[16] or 0,
                            file_path=row[17] or "",
                            processing_timestamp=self._parse_datetime(row[18]),
                            tenant_id=row[19] or "",
                            department=row[20] or "",
                            importance_level=row[21] or "normal",
                            user_tags=json.loads(row[22]) if row[22] else [],
                            access_acl=json.loads(row[23]) if row[23] else [],
                        )
                        priority = PriorityMetadata(
                            document_type=row[24] or "",
                            file_modified_date=self._parse_datetime(row[25]),
                            importance_level=row[26] or "normal",
                            ocr_confidence_avg=row[27] or 0.0,
                            ref_solved_ratio=row[28] or 0.0,
                            access_acl=json.loads(row[29]) if row[29] else [],
                            department=row[30] or "",
                            tenant_id=row[31] or "",
                        )

                        sm = SeparatedMetadata(document_id=doc_id, chunk_id=chunk_id, internal=internal, external=external, priority=priority)
                        results.append(sm)
            return results
        except Exception as e:
            self.logger.error(f"全メタデータ読み込みエラー: {e}")
            return []

    def get_document_ids(self) -> List[str]:
        """保存されている文書IDの一覧を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT document_id FROM metadata_index')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"文書ID取得エラー: {e}")
            return []

    def delete_metadata(self, document_id: str) -> bool:
        """指定文書のメタデータを削除"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # すべてのテーブルから削除
                for table in ['internal_metadata', 'external_metadata', 'priority_metadata']:
                    cursor.execute(f'DELETE FROM {table} WHERE document_id = ?', (document_id,))

                cursor.execute('DELETE FROM metadata_index WHERE document_id = ?', (document_id,))

                conn.commit()
                self.logger.info(f"メタデータ削除完了: {document_id}")
                return True

        except Exception as e:
            self.logger.error(f"メタデータ削除エラー: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """メタデータ統計情報を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 文書数を取得
                cursor.execute('SELECT COUNT(*) FROM metadata_index')
                document_count = cursor.fetchone()[0]

                # チャンク総数を取得
                cursor.execute('SELECT SUM(total_chunks) FROM metadata_index')
                total_chunks = cursor.fetchone()[0] or 0

                # 各テーブルのレコード数を取得
                stats = {
                    "total_documents": document_count,
                    "total_chunks": total_chunks,
                    "db_size_bytes": Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
                }

                return stats

        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {}

    def _parse_datetime(self, datetime_str: Any) -> Optional[datetime]:
        """日時文字列をdatetimeオブジェクトに変換"""
        if isinstance(datetime_str, datetime):
            return datetime_str
        elif isinstance(datetime_str, str):
            try:
                return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            except ValueError:
                return None
        return None








