"""
ナレッジベースUI
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.app.config import SUPPORTED_FILE_TYPES
from src.app.utils.access_control import acl_mode_message
from src.app.utils.logging import get_logger
from src.app.ui.table_utils import render_stable_dataframe

logger = get_logger(__name__)

ROLE_OPTIONS = ["一般", "係長以上", "課長以上", "部長以上", "役員以上"]
ROLE_LEVEL_MAP = {
    "一般": 1,
    "係長以上": 2,
    "課長以上": 3,
    "部長以上": 4,
    "役員以上": 5,
}
ACCESS_SCOPE_OPTIONS = [
    ("全員（テナント内）", "tenant"),
    ("部門限定", "department"),
    ("機密", "confidential"),
]
ACCESS_SCOPE_LABEL_MAP = {value: label for label, value in ACCESS_SCOPE_OPTIONS}


def _parse_csv_input(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def _apply_acl_to_chunks(
    chunks: List[Dict[str, Any]],
    acl_payload: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """ACL情報をチャンクに適用"""
    if not acl_payload:
        return chunks

    tenant_id = acl_payload.get("tenant_id")
    department = acl_payload.get("department")
    access_scope = acl_payload.get("access_scope")
    min_role_level = acl_payload.get("min_role_level")
    min_role_label = acl_payload.get("min_role_label")
    allowed_departments = _normalize_list(acl_payload.get("allowed_departments") or [])
    allowed_users = _normalize_list(acl_payload.get("allowed_users") or [])

    for chunk in chunks:
        metadata = chunk.setdefault("metadata", {})

        # externalアクセス制御
        external_meta = metadata.setdefault("external", {})
        access_acl = external_meta.setdefault("access_acl", {})
        access_acl.update(
            {
                "tenant_id": tenant_id,
                "department": department,
                "access_scope": access_scope,
                "min_role_level": min_role_level,
                "min_role_label": min_role_label,
                "allowed_departments": allowed_departments,
                "allowed_users": allowed_users,
            }
        )

        # priorityアクセス制御（高速フィルタ用）
        priority_meta = metadata.setdefault("priority", {})
        priority_acl = priority_meta.setdefault("access_acl", {})
        priority_acl.update(
            {
                "tenant_id": tenant_id,
                "access_scope": access_scope,
                "min_role_level": min_role_level,
            }
        )

        # 直接参照用メタデータ
        metadata["tenant_id"] = tenant_id
        metadata["department"] = department
        metadata["access_scope"] = access_scope
        metadata["min_role_level"] = min_role_level
        metadata["allowed_departments"] = allowed_departments
        metadata["allowed_users"] = allowed_users

    return chunks


def _get_default_department(user_context: Dict[str, Any]) -> str:
    departments = user_context.get("departments") or []
    if isinstance(departments, list) and departments:
        return str(departments[0])
    if isinstance(departments, str) and departments.strip():
        return departments.strip()
    return "未設定"


def _extract_acl_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    metadata = doc.get("metadata") or {}

    external_acl = (
        (metadata.get("external") or {}).get("access_acl")
        or (metadata.get("raw") or {}).get("external", {}).get("access_acl")
        or {}
    )
    priority_acl = (
        (metadata.get("priority") or {}).get("access_acl")
        or (metadata.get("raw") or {}).get("priority", {}).get("access_acl")
        or {}
    )

    tenant_id = metadata.get("tenant_id") or external_acl.get("tenant_id") or doc.get("tenant_id") or ""
    department = metadata.get("department") or external_acl.get("department") or "未設定"
    access_scope = metadata.get("access_scope") or external_acl.get("access_scope") or "tenant"
    min_role_level = metadata.get("min_role_level") or external_acl.get("min_role_level") or 1
    min_role_label = metadata.get("min_role_label") or external_acl.get("min_role_label")
    allowed_departments = metadata.get("allowed_departments") or external_acl.get("allowed_departments") or []
    allowed_users = metadata.get("allowed_users") or external_acl.get("allowed_users") or []

    if not min_role_label:
        min_role_label = next((label for label, level in ROLE_LEVEL_MAP.items() if level == min_role_level), "一般")

    return {
        "tenant_id": tenant_id,
        "department": department,
        "access_scope": access_scope,
        "min_role_level": min_role_level,
        "min_role_label": min_role_label,
        "allowed_departments": _normalize_list(allowed_departments),
        "allowed_users": _normalize_list(allowed_users),
        "priority": priority_acl,
    }


def _render_acl_editor(vector_store, doc: Dict[str, Any], acl_info: Optional[Dict[str, Any]] = None) -> None:
    doc_id = doc.get("doc_id") or doc.get("metadata", {}).get("stage1_basic", {}).get("doc_id")
    if not doc_id or not vector_store:
        st.info("この文書のACL情報を取得できませんでした。")
        return

    acl_info = acl_info or _extract_acl_metadata(doc)

    st.caption("POCモード: ACLは設定タブで編集可能です。将来的には管理者のみ編集できるように制限します。")

    col1, col2 = st.columns(2)
    with col1:
        tenant_id = st.text_input(
            "テナントID",
            value=acl_info["tenant_id"],
            key=f"acl_tenant_{doc_id}"
        )
        department = st.text_input(
            "主要部門",
            value=acl_info["department"],
            key=f"acl_department_{doc_id}"
        )

    with col2:
        access_scope_label = ACCESS_SCOPE_LABEL_MAP.get(
            acl_info["access_scope"],
            ACCESS_SCOPE_OPTIONS[0][0]
        )
        access_scope_label = st.selectbox(
            "アクセス範囲",
            [label for label, _ in ACCESS_SCOPE_OPTIONS],
            index=[label for label, _ in ACCESS_SCOPE_OPTIONS].index(access_scope_label),
            key=f"acl_scope_{doc_id}"
        )
        access_scope_value = next(value for label, value in ACCESS_SCOPE_OPTIONS if label == access_scope_label)

        min_role_label = st.selectbox(
            "最小役職",
            ROLE_OPTIONS,
            index=ROLE_OPTIONS.index(acl_info["min_role_label"]) if acl_info["min_role_label"] in ROLE_OPTIONS else 0,
            key=f"acl_role_{doc_id}"
        )
        min_role_level = ROLE_LEVEL_MAP[min_role_label]

    col3, col4 = st.columns(2)
    with col3:
        allowed_departments = st.text_input(
            "許可部門 (カンマ区切り)",
            value=", ".join(acl_info["allowed_departments"]),
            key=f"acl_allowed_departments_{doc_id}"
        )
    with col4:
        allowed_users = st.text_input(
            "許可ユーザー (カンマ区切り)",
            value=", ".join(acl_info["allowed_users"]),
            key=f"acl_allowed_users_{doc_id}"
        )

    if st.button("ACLを更新", key=f"update_acl_{doc_id}"):
        if not tenant_id.strip():
            st.error("テナントIDは必須です。")
            return

        updated_acl = {
            "tenant_id": tenant_id.strip(),
            "department": department.strip() or "未設定",
            "access_scope": access_scope_value,
            "min_role_level": min_role_level,
            "min_role_label": min_role_label,
            "allowed_departments": _parse_csv_input(allowed_departments),
            "allowed_users": _parse_csv_input(allowed_users),
        }

        if vector_store.update_document_acl(doc_id, updated_acl):
            st.success("ACLを更新しました")
            st.rerun()
        else:
            st.error("ACLの更新に失敗しました")

def display_knowledge_base(vector_store):
    """ナレッジベースの内容を表示"""
    st.subheader("ナレッジベース内容")
    
    if not vector_store:
        st.info("ベクトルストアが初期化されていません。")
        return

    try:
        # ベクトルストアから文書を取得
        documents = vector_store.get_all_documents()
        
        if not documents:
            st.info("ナレッジベースに文書が登録されていません。")
            return
        
        # 文書一覧を表示
        st.write(f"登録文書数: {len(documents)}")
        
        # 文書タイプ別の統計
        doc_types = {}
        for doc in documents:
            stage1 = (doc.get('metadata') or {}).get('stage1_basic', {})
            doc_type = doc.get('file_type') or stage1.get('file_type') or 'unknown'
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        if doc_types:
            st.subheader("文書タイプ別統計")
            df_types = pd.DataFrame(list(doc_types.items()), columns=['文書タイプ', '件数'])
            render_stable_dataframe(
                df_types,
                column_config={
                    "文書タイプ": st.column_config.TextColumn(width="medium"),
                    "件数": st.column_config.NumberColumn(format="%d", width="small"),
                },
                min_height=132,
                max_height=280,
            )
        
        # 文書一覧をテーブルで表示
        st.subheader("文書一覧")
        
        # 表示用のデータを準備
        display_data = []
        for i, doc in enumerate(documents):
            stage1 = (doc.get('metadata') or {}).get('stage1_basic', {})
            raw_size = stage1.get('file_size', doc.get('file_size'))
            normalized_size = None
            if isinstance(raw_size, (int, float)):
                normalized_size = int(raw_size)
            elif isinstance(raw_size, str):
                stripped = raw_size.replace(',', '').strip()
                if stripped.isdigit():
                    normalized_size = int(stripped)
            display_data.append({
                'ID': i + 1,
                'ファイル名': doc.get('file_name', 'N/A'),
                '文書タイプ': doc.get('file_type', stage1.get('file_type', 'N/A')),
                'チャンク数': len(doc.get('chunks', [])),
                '作成日': stage1.get('created_at', doc.get('created_at', 'N/A')),
                'サイズ (bytes)': normalized_size if normalized_size is not None else None,
            })
        
        if display_data:
            df = pd.DataFrame(display_data)
            render_stable_dataframe(
                df,
                column_config={
                    'ID': st.column_config.NumberColumn(width="small"),
                    'ファイル名': st.column_config.TextColumn(width="large"),
                    '文書タイプ': st.column_config.TextColumn(width="small"),
                    'チャンク数': st.column_config.NumberColumn(format="%d", width="small"),
                    '作成日': st.column_config.TextColumn(width="medium"),
                    'サイズ (bytes)': st.column_config.NumberColumn(format="%,d", width="small"),
                },
                min_height=192,
                max_height=440,
            )
            
            # 個別文書の詳細表示
            if st.checkbox("個別文書の詳細を表示"):
                selected_doc_index = st.selectbox(
                    "表示する文書を選択",
                    range(len(documents)),
                    format_func=lambda x: f"{x+1}: {documents[x].get('file_name', 'N/A')}"
                )
                
                if selected_doc_index is not None:
                    doc = documents[selected_doc_index]
                    _display_document_details(vector_store, doc)
        else:
            st.info("表示する文書がありません。")
            
    except Exception as e:
        st.error(f"ナレッジベースの表示中にエラーが発生しました: {e}")
        logger.error(f"ナレッジベース表示エラー: {e}")

def _display_document_details(vector_store, doc: Dict[str, Any]):
    """個別文書の詳細を表示"""
    st.subheader("文書詳細")
    
    col1, col2 = st.columns(2)
    
    stage1 = (doc.get('metadata') or {}).get('stage1_basic', {})
    doc_id = doc.get('doc_id') or stage1.get('doc_id')

    with col1:
        st.write(f"**ファイル名**: {doc.get('file_name', 'N/A')}")
        st.write(f"**文書タイプ**: {doc.get('file_type', stage1.get('file_type', 'N/A'))}")
        file_size = stage1.get('file_size', doc.get('file_size', 'N/A'))
        st.write(f"**ファイルサイズ**: {file_size if file_size is not None else 'N/A'} bytes")
    
    with col2:
        st.write(f"**作成日**: {stage1.get('created_at', doc.get('created_at', 'N/A'))}")
        st.write(f"**更新日**: {doc.get('updated_at', 'N/A')}")
        st.write(f"**チャンク数**: {len(doc.get('chunks', []))}")
    
    if doc_id:
        with st.expander("文書を削除", expanded=False):
            st.warning("この操作は元に戻せません。関連するチャンクやインデックスも削除されます。")
            confirm_key = f"confirm_delete_{doc_id}"
            confirmed = st.checkbox("削除してよいことを確認しました", key=confirm_key)
            if st.button("文書を削除する", key=f"delete_doc_{doc_id}"):
                if not confirmed:
                    st.error("削除を実行するには確認チェックが必要です。")
                else:
                    try:
                        vector_store.delete_document(doc_id)
                        st.success("文書を削除しました。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"文書の削除に失敗しました: {exc}")
                        logger.error(f"ナレッジ削除エラー({doc_id}): {exc}")

    # メタデータ（編集可能）
    if doc.get('metadata'):
        st.subheader("メタデータ")
        
        # 段階的メタデータの表示・編集
        if isinstance(doc['metadata'], dict) and 'stage1_basic' in doc['metadata']:
            # 段階的メタデータの場合
            with st.expander("基本情報", expanded=True):
                st.json(doc['metadata']['stage1_basic'])
            
            with st.expander("処理結果", expanded=False):
                st.json(doc['metadata']['stage2_processing'])
            
            with st.expander("ビジネス情報（編集可能）", expanded=False):
                business_meta = doc['metadata'].get('stage3_business', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    department = st.selectbox(
                        "部門",
                        ["営業", "マーケティング", "開発", "人事", "財務", "未設定"],
                        index=["営業", "マーケティング", "開発", "人事", "財務", "未設定"].index(
                            business_meta.get("department", "未設定")
                        ),
                        key=f"dept_{doc.get('doc_id', doc.get('file_name', ''))}"
                    )
                    
                    importance = st.selectbox(
                        "重要度",
                        ["高", "中", "低"],
                        index=["高", "中", "低"].index(
                            business_meta.get("importance_level", "中")
                        ),
                        key=f"importance_{doc.get('doc_id', doc.get('file_name', ''))}"
                    )
                
                with col2:
                    access_level = st.selectbox(
                        "アクセスレベル",
                        ["一般", "制限", "機密"],
                        index=["一般", "制限", "機密"].index(
                            business_meta.get("access_level", "一般")
                        ),
                        key=f"access_{doc.get('doc_id', doc.get('file_name', ''))}"
                    )
                
                user_tags = st.text_input(
                    "タグ（カンマ区切り）",
                    value=", ".join(business_meta.get("user_tags", [])),
                    key=f"tags_{doc.get('doc_id', doc.get('file_name', ''))}"
                )
                
                if st.button("ビジネス情報を更新", key=f"update_business_{doc.get('doc_id', doc.get('file_name', ''))}"):
                    # メタデータ更新処理
                    updated_metadata = {
                        "department": department,
                        "importance_level": importance,
                        "access_level": access_level,
                        "user_tags": [tag.strip() for tag in user_tags.split(",") if tag.strip()]
                    }
                    
                    # ベクトルストアのメタデータを更新
                    if _update_document_metadata(vector_store, doc.get('doc_id'), 'stage3_business', updated_metadata):
                        st.success("ビジネス情報を更新しました")
                        st.rerun()  # UIを更新
                    else:
                        st.error("メタデータの更新に失敗しました")
            
            with st.expander("検索最適化", expanded=False):
                search_meta = doc['metadata'].get('stage4_search', {})
                
                keywords = st.text_input(
                    "キーワード（カンマ区切り）",
                    value=", ".join(search_meta.get("keywords", [])),
                    key=f"keywords_{doc.get('doc_id', doc.get('file_name', ''))}"
                )
                
                category = st.selectbox(
                    "カテゴリ",
                    ["技術文書", "営業資料", "財務報告", "人事資料", "未分類"],
                    index=["技術文書", "営業資料", "財務報告", "人事資料", "未分類"].index(
                        search_meta.get("category", "未分類")
                    ),
                    key=f"category_{doc.get('doc_id', doc.get('file_name', ''))}"
                )
                
                if st.button("検索情報を更新", key=f"update_search_{doc.get('doc_id', doc.get('file_name', ''))}"):
                    # メタデータ更新処理
                    updated_metadata = {
                        "keywords": [keyword.strip() for keyword in keywords.split(",") if keyword.strip()],
                        "category": category,
                        "search_priority": 1.0,  # 将来実装
                        "last_accessed": None
                    }
                    
                    # ベクトルストアのメタデータを更新
                    if _update_document_metadata(vector_store, doc.get('doc_id'), 'stage4_search', updated_metadata):
                        st.success("検索情報を更新しました")
                        st.rerun()  # UIを更新
                    else:
                        st.error("メタデータの更新に失敗しました")

            with st.expander("アクセス制御 (ACL)", expanded=False):
                _render_acl_editor(vector_store, doc)
        else:
            # 従来のメタデータ
            st.json(doc['metadata'])
    
    # チャンク内容
    chunks = doc.get('chunks', [])
    if chunks:
        st.subheader("チャンク内容")
        
        for i, chunk in enumerate(chunks[:5]):  # 最初の5チャンクのみ表示
            with st.expander(f"チャンク {i+1}"):
                st.write(chunk.get('content', 'N/A'))
                
                if chunk.get('metadata'):
                    st.write("**メタデータ**:")
                    st.json(chunk['metadata'])
        
        if len(chunks) > 5:
            st.info(f"他に {len(chunks) - 5} 個のチャンクがあります")

def _update_document_metadata(vector_store, doc_id: Optional[str], stage: str, updated_metadata: Dict[str, Any]) -> bool:
    """文書のメタデータを更新"""
    try:
        if not vector_store or not doc_id:
            return False

        success = vector_store.update_document_metadata(doc_id, stage, updated_metadata)
        if success:
            logger.info(f"メタデータ更新完了: {doc_id} - {stage}")
        return success

    except Exception as e:
        logger.error(f"メタデータ更新エラー: {e}")
        return False


def _get_user_context() -> Dict[str, Any]:
    context = st.session_state.get("user_context") or {}
    tenant_id = context.get("tenant_id") or ""
    departments = context.get("departments") or []
    role_level = context.get("role_level") or 1
    role_label = context.get("role_label") or "一般"

    if isinstance(departments, str):
        departments = [dept.strip() for dept in departments.split(",") if dept.strip()]

    return {
        "tenant_id": tenant_id,
        "departments": departments,
        "role_level": role_level,
        "role_label": role_label,
    }


def knowledge_registration_ui(client, vector_store):
    """ナレッジベース登録UI"""
    st.subheader("ナレッジ登録")
    st.caption("ファイルをアップロードすると、自動でチャンク化・メタデータ生成・インデックス登録まで実行します。")

    form_col, guide_col = st.columns([0.58, 0.42])

    uploaded_files: List[Any] = []
    tenant_id = ""
    department = ""
    access_scope_value = "tenant"
    min_role_label = ROLE_OPTIONS[0]
    min_role_level = ROLE_LEVEL_MAP[min_role_label]
    allowed_departments: List[str] = []
    allowed_users: List[str] = []
    errors: List[str] = []

    with form_col:
        st.markdown("#### 1. ファイルを選択")
        uploaded_files = st.file_uploader(
            "登録するファイル",
            type=sum(SUPPORTED_FILE_TYPES.values(), []),
            accept_multiple_files=True,
            help="対応形式: " + ", ".join(sorted(set(sum(SUPPORTED_FILE_TYPES.values(), []))))
        )

        if uploaded_files:
            st.markdown("#### 2. アクセス制御")
            user_context = _get_user_context()
            tenant_id = user_context.get("tenant_id") or st.session_state.get("user_tenant_id") or ""
            department = _get_default_department(user_context)

            tenant_id = st.text_input("テナントID", value=tenant_id, help="例: 001")
            department = st.text_input("主要部門", value=department or "未設定")

            access_scope_label = st.selectbox(
                "アクセス範囲",
                options=[label for label, _ in ACCESS_SCOPE_OPTIONS],
                index=0,
            )
            access_scope_value = next(value for label, value in ACCESS_SCOPE_OPTIONS if label == access_scope_label)

            min_role_label = st.selectbox("最小役職", options=ROLE_OPTIONS, index=0)
            min_role_level = ROLE_LEVEL_MAP[min_role_label]

            allowed_departments_input = st.text_input("許可部門 (任意)", value=", ".join(user_context.get("departments", [])))
            allowed_users_input = st.text_input("許可ユーザー (任意)")

            allowed_departments = _parse_csv_input(allowed_departments_input)
            allowed_users = _parse_csv_input(allowed_users_input)

            errors = []
            if not tenant_id.strip():
                errors.append("テナントIDを入力してください。")
            if access_scope_value == "confidential" and not allowed_users:
                errors.append("機密設定時は許可ユーザーを1件以上指定してください。")
            for message in errors:
                st.error(message)

            st.markdown("#### 3. 処理オプション")
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider(
                    "チャンクサイズ",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    key="register_chunk_size",
                )
                chunk_overlap = st.slider(
                    "チャンクオーバーラップ",
                    min_value=0,
                    max_value=500,
                    value=200,
                    key="register_chunk_overlap",
                )
            with col2:
                max_workers = st.slider(
                    "最大ワーカー数",
                    min_value=1,
                    max_value=8,
                    value=4,
                    key="register_max_workers",
                )
                auto_classify = st.checkbox("自動文書分類", value=True, key="register_auto_classify")

            if st.button("登録を開始", type="primary"):
                if errors:
                    return
                if min_role_level <= 0:
                    st.error("最小役職の設定を確認してください。")
                    return
                _process_and_register_files(
                    uploaded_files,
                    vector_store,
                    client,
                    chunk_size,
                    chunk_overlap,
                    max_workers,
                    auto_classify,
                    acl_payload={
                        "tenant_id": tenant_id.strip(),
                        "department": department.strip() or "未設定",
                        "access_scope": access_scope_value,
                        "min_role_level": min_role_level,
                        "min_role_label": min_role_label,
                        "allowed_departments": allowed_departments,
                        "allowed_users": allowed_users,
                    },
                )

    with guide_col:
        st.markdown("#### 操作ガイド")
        st.markdown(
            """
            1. ファイルを選択
            2. アクセス制御を設定
            3. オプションを確認して登録
            """
        )
        st.caption(acl_mode_message())

        if uploaded_files:
            file_info = [
                {
                    "ファイル名": file.name,
                    "サイズ": f"{file.size:,} bytes",
                    "タイプ": file.type,
                }
                for file in uploaded_files
            ]
            st.markdown("#### 選択中のファイル")
            df = pd.DataFrame(file_info)
            render_stable_dataframe(
                df,
                column_config={
                    "ファイル名": st.column_config.TextColumn(width="large"),
                    "サイズ": st.column_config.TextColumn(width="medium"),
                    "タイプ": st.column_config.TextColumn(width="small"),
                },
                min_height=150,
                max_height=210,
            )

def _process_and_register_files(
    uploaded_files,
    vector_store,
    client,
    chunk_size: int,
    chunk_overlap: int,
    max_workers: int,
    auto_classify: bool,
    acl_payload: Optional[Dict[str, Any]] = None,
):
    """ファイルを処理してナレッジベースに登録"""
    
    import uuid
    request_id = str(uuid.uuid4())[:8]  # 8文字の短縮UUID
    
    phase_steps = [
        ("prepare", "ファイル保存"),
        ("analyze", "解析"),
        ("register", "登録"),
        ("complete", "完了"),
    ]
    phase_state = {key: "pending" for key, _ in phase_steps}
    timeline_placeholder = st.empty()
    status_text = st.empty()
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0.0)
    event_placeholder = st.empty()

    def _render_timeline() -> None:
        markers = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "done": "[✓]",
            "error": "[x]",
        }
        segments = []
        for key, label in phase_steps:
            marker = markers.get(phase_state.get(key, "pending"), "[ ]")
            segments.append(f"{marker} {label}")
        timeline_placeholder.markdown("**処理フロー**  " + " ─ ".join(segments))

    def _update_phase(key: str, state: str) -> None:
        if key in phase_state:
            phase_state[key] = state
        _render_timeline()

    def _set_event(message: str) -> None:
        from datetime import datetime as _dt

        timestamp = _dt.now().strftime("%H:%M:%S")
        event_placeholder.markdown(f"> {timestamp} {message}")

    _render_timeline()
    _update_phase("prepare", "in_progress")
    _set_event("一時領域にファイルを保存しています")
    
    logger.info(f"[{request_id}] ファイル処理開始: {len(uploaded_files)}ファイル")
    
    try:
        from src.app.processing.file_processor import process_files_batch
        from pathlib import Path
        import tempfile
        
        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # ファイルを一時保存
            file_paths = []
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = temp_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append((file_path, {}))
                _set_event(f"保存完了: {uploaded_file.name}")

            _update_phase("prepare", "done")
            _update_phase("analyze", "in_progress")
            status_text.text("解析中...")
            _set_event("ファイル解析と要約を実行しています")
            logger.info(f"[{request_id}] ファイル処理開始")
            results = process_files_batch(
                file_paths,
                {},
                client,
                temp_path,
                max_workers
            )
            logger.info(f"[{request_id}] ファイル処理完了: {len(results)}ファイル")
            _set_event(f"解析完了: {len(results)}ファイル")
            _update_phase("analyze", "done")

            # ベクトルストアに登録
            status_text.text("登録処理を実行しています...")
            _update_phase("register", "in_progress")
            _set_event("チャンク登録と埋め込みを更新しています")
            registered_count = 0
            failure_records: List[Dict[str, Any]] = []

            total_results = max(len(results), 1)
            total_chunks = 0

            for index, (file_path, chunks, temp_file, error) in enumerate(results, start=1):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if error:
                    failure_records.append({
                        "timestamp": timestamp,
                        "file_name": file_path.name,
                        "reason": str(error),
                    })
                    logger.warning(f"[{request_id}] 登録スキップ: {file_path.name} - {error}")
                    _set_event(f"{file_path.name}: 登録をスキップしました")

                if chunks:
                    logger.info(f"[{request_id}] ベクトルストア登録: {file_path.name} ({len(chunks)}チャンク)")
                    chunk_count = len(chunks)
                    total_chunks += chunk_count
                    _set_event(f"{file_path.name}: {chunk_count}チャンクを登録中")
                    result = vector_store.add_document(
                        file_path,
                        _apply_acl_to_chunks(chunks, acl_payload),
                        None,
                    )
                    if result == "success":
                        registered_count += 1
                        logger.info(f"[{request_id}] 登録成功: {file_path.name}")
                        _set_event(f"{file_path.name}: 登録完了")
                    else:
                        failure_records.append({
                            "timestamp": timestamp,
                            "file_name": file_path.name,
                            "reason": f"ベクトル登録失敗 ({result})",
                        })
                        logger.warning(f"[{request_id}] 登録失敗: {file_path.name} ({result})")
                        _set_event(f"{file_path.name}: 登録に失敗しました ({result})")

                progress_bar.progress(min(index / total_results, 1.0))

            # 結果表示
            _update_phase("register", "done")
            logger.info(f"[{request_id}] 処理完了: {registered_count}/{len(uploaded_files)} ファイル登録")
            st.success(f"処理完了: {registered_count}/{len(uploaded_files)} ファイルが登録されました")

            if failure_records:
                st.error("登録できなかったファイル:\n- " + "\n- ".join(f"{item['file_name']}: {item['reason']}" for item in failure_records))

            # 統計情報
            st.info(f"総チャンク数: {total_chunks}")
            logger.info(f"[{request_id}] 総チャンク数: {total_chunks}")
            _set_event("処理が完了しました")
            _update_phase("complete", "done")

            # セッション状態に登録結果を保存
            summary = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": len(uploaded_files),
                "registered_count": registered_count,
                "total_chunks": total_chunks,
            }
            st.session_state["knowledge_registration_summary"] = summary

            if failure_records:
                error_log = st.session_state.setdefault("knowledge_registration_errors", [])
                error_log.extend(failure_records)
                # 最新50件のみ保持
                st.session_state["knowledge_registration_errors"] = error_log[-50:]
            else:
                st.session_state.setdefault("knowledge_registration_errors", [])
            
    except Exception as e:
        logger.error(f"[{request_id}] ファイル処理エラー: {e}")
        st.error(f"ファイル処理中にエラーが発生しました: {e}")
        _set_event(f"エラーが発生しました: {e}")
        _update_phase("complete", "error")
    
    finally:
        progress_placeholder.empty()
        status_text.empty()
        logger.info(f"[{request_id}] 処理終了")
