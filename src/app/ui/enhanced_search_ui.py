"""
拡張検索結果表示UI
ファイルタイプ別の高度な表示機能を統合
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from src.file_type_display import SearchResultsDisplay, create_display_config
from src.app.utils.logging import get_logger

logger = get_logger(__name__)

def enhanced_search_results_ui(
    results: List[Dict[str, Any]],
    query: str,
    show_file_type_summary: bool = True,
    max_results: int = 10,
    show_thumbnail: bool = True,
    show_preview: bool = True
) -> None:
    """
    拡張された検索結果表示UI
    
    Args:
        results: 検索結果リスト
        query: 検索クエリ
        show_file_type_summary: ファイルタイプ別サマリーを表示するか
        max_results: 最大表示件数
        show_thumbnail: サムネイルを表示するか
        show_preview: プレビューを表示するか
    """
    
    # 表示設定を作成
    display_config = create_display_config(
        show_thumbnail=show_thumbnail,
        show_preview=show_preview,
        max_preview_size=500,
        thumbnail_size=(200, 150),
        enable_interactive=True
    )
    
    # 検索結果表示管理クラス
    display_manager = SearchResultsDisplay(display_config)
    
    # ファイルタイプ別サマリー表示
    if show_file_type_summary and results:
        display_manager.display_file_type_summary(results)
        st.divider()
    
    # 検索結果表示
    display_manager.display_search_results(results, query, max_results)

def search_ui_with_filters(
    results: List[Dict[str, Any]],
    query: str,
    enable_filters: bool = True
) -> None:
    """
    フィルタ機能付き検索UI
    
    Args:
        results: 検索結果リスト
        query: 検索クエリ
        enable_filters: フィルタ機能を有効にするか
    """
    
    st.markdown(f"### 検索結果: '{query}'")
    st.caption("列を整理し、必要に応じてプレビューやフィルターを展開できるようにしてあります。")
    st.markdown(f"**{len(results)}件**の結果が見つかりました")
    
    if not results:
        st.info("検索結果が見つかりませんでした")
        return
    
    # フィルタ機能
    if enable_filters:
        with st.expander("表示フィルタ", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_thumbnail = st.checkbox("サムネイル表示", value=True, help="画像やPDFではマウスオーバーで拡大ヒントが表示されます。")
                show_preview = st.checkbox("プレビュー表示", value=True, help="テキスト/表の抜粋を展開表示します。")
            
            with col2:
                max_results = st.slider("表示件数", 1, 50, 10, help="表示件数が多いと一覧性が下がるため、必要に応じて調整してください。")
                sort_by = st.selectbox(
                    "並び順",
                    ["関連度", "ファイルタイプ", "ファイルサイズ", "作成日"],
                    help="Microsoftデザインガイドに合わせ、主要指標から選択できるようにしています。"
                )
            
            with col3:
                file_type_filter = st.multiselect(
                    "ファイルタイプ",
                    ["image", "pdf", "excel", "word", "text"],
                    default=[],
                    help="特定の文書タイプだけに絞り込みたい場合に選択してください。"
                )
        
        # フィルタ適用
        filtered_results = apply_filters(results, sort_by, file_type_filter)
    else:
        filtered_results = results
        show_thumbnail = True
        show_preview = True
        max_results = 10
    
    # 拡張検索結果表示
    enhanced_search_results_ui(
        results=filtered_results,
        query=query,
        show_file_type_summary=False,  # フィルタUIで表示済み
        max_results=max_results,
        show_thumbnail=show_thumbnail,
        show_preview=show_preview
    )

def apply_filters(
    results: List[Dict[str, Any]], 
    sort_by: str,
    file_type_filter: List[str]
) -> List[Dict[str, Any]]:
    """検索結果にフィルタを適用"""
    
    filtered_results = results.copy()
    
    # ファイルタイプフィルタ
    if file_type_filter:
        filtered_results = [
            result for result in filtered_results
            if any(file_type in str(result.get('metadata', {}).get('file_type', '')).lower() 
                   for file_type in file_type_filter)
        ]
    
    # ソート
    if sort_by == "関連度":
        filtered_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    elif sort_by == "ファイルタイプ":
        filtered_results.sort(key=lambda x: x.get('metadata', {}).get('file_type', ''))
    elif sort_by == "ファイルサイズ":
        filtered_results.sort(
            key=lambda x: x.get('metadata', {}).get('file_size', 0), 
            reverse=True
        )
    elif sort_by == "作成日":
        filtered_results.sort(
            key=lambda x: x.get('metadata', {}).get('creation_date', ''), 
            reverse=True
        )
    
    return filtered_results

def interactive_search_ui(
    search_function,
    default_query: str = "",
    placeholder: str = "検索クエリを入力してください..."
) -> None:
    """
    インタラクティブ検索UI
    
    Args:
        search_function: 検索関数 (query: str) -> List[Dict[str, Any]]
        default_query: デフォルトクエリ
        placeholder: プレースホルダーテキスト
    """
    
    # 検索クエリ入力
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "検索クエリ",
            value=default_query,
            placeholder=placeholder,
            key="search_query"
        )
    
    with col2:
        search_button = st.button("検索", type="primary")
    
    # 検索実行
    if search_button and query.strip():
        with st.spinner("検索中..."):
            try:
                results = search_function(query.strip())
                st.session_state.search_results = results
                st.session_state.last_query = query.strip()
            except Exception as e:
                st.error(f"検索エラー: {e}")
                logger.error(f"検索エラー: {e}")
                return
    
    # 検索結果表示
    if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
        search_ui_with_filters(
            results=st.session_state.search_results,
            query=st.session_state.get('last_query', ''),
            enable_filters=True
        )

def quick_search_ui(
    results: List[Dict[str, Any]],
    query: str,
    compact_mode: bool = False
) -> None:
    """
    クイック検索結果表示UI（コンパクト版）
    
    Args:
        results: 検索結果リスト
        query: 検索クエリ
        compact_mode: コンパクトモード
    """
    
    st.markdown(f"**'{query}'** の検索結果 ({len(results)}件)")
    
    if not results:
        st.info("結果なし")
        return
    
    # コンパクトモード
    if compact_mode:
        for i, result in enumerate(results[:5], 1):
            score = result.get('score', 0.0)
            text = result.get('text', result.get('content', ''))[:100]
            file_type = result.get('metadata', {}).get('file_type', 'unknown')
            
            st.markdown(f"{i}. **{file_type}** (スコア: {score:.3f}) - {text}...")
    else:
        # 通常表示
        enhanced_search_results_ui(
            results=results,
            query=query,
            show_file_type_summary=False,
            max_results=5,
            show_thumbnail=False,
            show_preview=False
        )

def search_history_ui() -> None:
    """検索履歴表示UI"""
    
    if not hasattr(st.session_state, 'search_history'):
        st.session_state.search_history = []
    
    st.markdown("### 検索履歴")
    
    if not st.session_state.search_history:
        st.info("検索履歴がありません")
        return
    
    # 履歴表示
    for i, (query, timestamp, result_count) in enumerate(st.session_state.search_history[-10:]):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{query}**")
        
        with col2:
            st.markdown(f"{result_count}件")
        
        with col3:
            if st.button("再検索", key=f"rerun_{i}"):
                # 再検索実行
                st.session_state.search_query = query
                st.rerun()

def add_to_search_history(query: str, result_count: int) -> None:
    """検索履歴に追加"""
    if not hasattr(st.session_state, 'search_history'):
        st.session_state.search_history = []
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M")
    
    st.session_state.search_history.append((query, timestamp, result_count))
    
    # 履歴を最新10件に制限
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[-10:]

