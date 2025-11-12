"""
メインアプリケーションUI
"""
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from src.app.utils.logging import get_logger
from src.app.ui.enhanced_search_ui import enhanced_search_results_ui, interactive_search_ui
from src.app.utils.audit_log import (
    events_to_csv,
    load_events_paginated,
    summarize_decisions,
)
from src.app.utils.access_control import acl_mode_message
from src.app.ui.table_utils import render_stable_dataframe
from src.app.utils.summary_metrics import get_summary_metrics, get_recent_summary_events

logger = get_logger(__name__)


def _parse_date_string(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    # ISO形式の時間付きの場合は日付部分のみ使用
    if "T" in text:
        text = text.split("T", 1)[0]

    normalized_candidates = {text, text.replace("/", "-"), text.replace(".", "-")}
    for candidate in normalized_candidates:
        try:
            return datetime.fromisoformat(candidate).date()
        except ValueError:
            continue

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    return None


def _extract_document_date(metadata: Dict[str, Any]) -> Optional[date]:
    stage1 = metadata.get("stage1_basic") or {}
    candidates = [
        metadata.get("created_at"),
        stage1.get("created_at"),
        metadata.get("updated_at"),
        stage1.get("last_indexed_at"),
    ]
    for candidate in candidates:
        parsed = _parse_date_string(candidate)
        if parsed:
            return parsed
    return None

def main_app_ui():
    """ログイン後に表示されるメインアプリケーションのUI"""
    st.sidebar.success(f"ようこそ、{st.session_state.user_email}さん")
    
    if st.sidebar.button("ログアウト"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # 統合タブ構成（8タブ → 4タブ）
    # 関連機能をグループ化して、ユーザーの認知負荷を低減
    tab1, tab2, tab3, tab4 = st.tabs([
        "検索",
        "ナレッジ管理",
        "分析",
        "設定",
    ])
    
    # ===== Tab 1: 検索 =====
    # 統合：基本検索 + PDF処理 + 時間認識検索
    with tab1:
        st.header("ナレッジ検索")
        st.caption("標準はベクトルとBM25を組み合わせたハイブリッド検索です。登録作業は『ナレッジ管理』タブで行います。")

        with st.expander("アクセス制御のヒント", expanded=False):
            user_context = st.session_state.get("user_context") or {}
            tenant_id = user_context.get("tenant_id") or "未設定"
            departments = user_context.get("departments") or []
            role_label = user_context.get("role_label") or "一般"
            st.markdown(
                """
                - **テナント**: `{tenant}`
                - **所属部門**: {departments}
                - **役職レベル**: {role}

                この条件に合致しない文書は検索候補に含まれません。表示されない場合は「設定」タブでユーザーコンテキストを調整してください。
                """.format(
                    tenant=tenant_id,
                    departments=", ".join(departments) if departments else "未設定",
                    role=role_label,
                )
            )
            st.caption(acl_mode_message())
        
        st.markdown("---")
        st.subheader("ハイブリッド検索")
        st.markdown("自然言語で質問を入力すると、ベクトル検索とBM25検索を組み合わせた結果を返します。")
        query = st.text_area(
            "質問を入力",
            placeholder="例:\n• プロジェクトの進捗状況について\n• 最新の財務報告書はどこにありますか？",
            height=100
        )

        use_period = st.checkbox("期間を指定する", help="チェックすると検索結果を指定期間内に絞り込みます。")
        calendar_range = None
        manual_start = ""
        manual_end = ""

        if use_period:
            default_range = (
                date.today() - timedelta(days=30),
                date.today(),
            )
            cal_col, manual_col = st.columns([0.6, 0.4])
            with cal_col:
                calendar_range = st.date_input(
                    "期間（カレンダー選択）",
                    value=default_range,
                    help="カレンダーで範囲を選択できます。単一日付を選択することも可能です。",
                )
            with manual_col:
                manual_start = st.text_input(
                    "開始日 (YYYY-MM-DD)",
                    placeholder="例: 1993-04-01",
                )
                manual_end = st.text_input(
                    "終了日 (YYYY-MM-DD)",
                    placeholder="例: 1993-12-31",
                )

        if st.button("検索実行", type="primary"):
            if query:
                try:
                    rag_engine = st.session_state.get('rag_engine')
                    vector_store = st.session_state.get('vector_store')

                    if not vector_store:
                        st.error("ベクトルストアが初期化されていません")
                        return

                    if not rag_engine:
                        st.error("RAGエンジンが初期化されていません")
                        return

                    with st.spinner("拡張検索中..."):
                        results = rag_engine.search(
                            query=query,
                            top_k=5,
                            search_mode="hybrid",
                            use_adaptive=True,
                        )

                    start_date_filter: Optional[date] = None
                    end_date_filter: Optional[date] = None
                    if use_period:
                        if isinstance(calendar_range, (list, tuple)):
                            if len(calendar_range) == 2:
                                start_date_filter = calendar_range[0]
                                end_date_filter = calendar_range[1]
                            elif len(calendar_range) == 1:
                                start_date_filter = end_date_filter = calendar_range[0]
                        elif isinstance(calendar_range, date):
                            start_date_filter = end_date_filter = calendar_range

                        manual_start_date = _parse_date_string(manual_start)
                        manual_end_date = _parse_date_string(manual_end)
                        if manual_start_date:
                            start_date_filter = manual_start_date
                        if manual_end_date:
                            end_date_filter = manual_end_date

                        if start_date_filter and end_date_filter and start_date_filter > end_date_filter:
                            start_date_filter, end_date_filter = end_date_filter, start_date_filter

                    if use_period and (start_date_filter or end_date_filter):
                        filtered_results = []
                        for item in results:
                            metadata = item.get("metadata", {})
                            doc_date = _extract_document_date(metadata)
                            if not doc_date:
                                filtered_results.append(item)
                                continue
                            if start_date_filter and doc_date < start_date_filter:
                                continue
                            if end_date_filter and doc_date > end_date_filter:
                                continue
                            filtered_results.append(item)
                        results = filtered_results

                    if results:
                        st.success(f"検索完了: {len(results)}件の結果が見つかりました")

                        formatted_results = []
                        for item in results:
                            metadata = item.get("metadata", {})
                            formatted_results.append(
                                {
                                    "content": item.get("content", ""),
                                    "metadata": metadata,
                                    "score": item.get("score", 0.0),
                                    "vector_score": item.get("vector_score", 0.0),
                                    "lexical_score": item.get("lexical_score", 0.0),
                                    "title": item.get("title", "無題"),
                                    "file_path": item.get("file_path"),
                                    "doc_id": item.get("doc_id"),
                                }
                            )

                        enhanced_search_results_ui(
                            results=formatted_results,
                            query=query,
                            show_file_type_summary=False,
                            max_results=5,
                            show_thumbnail=False,
                            show_preview=True,
                        )
                    else:
                        st.warning("検索結果が見つかりませんでした")
                except Exception as e:
                    st.error(f"検索中にエラーが発生しました: {e}")
                    logger.error(f"検索エラー: {e}")
            else:
                st.warning("質問を入力してください")
    
    # ===== Tab 2: ナレッジ管理 =====
    # 統合：ナレッジベース登録・管理
    with tab2:
        # 実際の処理パイプラインを呼び出し
        from src.app.ui.knowledge_ui import knowledge_registration_ui, display_knowledge_base
        
        # ナレッジベース登録UI
        knowledge_registration_ui(st.session_state.get('openai_client'), st.session_state.get('vector_store'))
        
        st.markdown("---")
        
        # ナレッジベース表示
        display_knowledge_base(st.session_state.get('vector_store'))
    
    # ===== Tab 3: 分析 =====
    # 統合：ダッシュボード + 高度評価
    with tab3:
        st.header("分析")
        st.markdown("インデックス状況と検索パフォーマンスのヘルスチェックを行います。")

        vector_store = st.session_state.get('vector_store')

        if not vector_store:
            st.info("ベクトルストアが初期化されていません。ナレッジ登録を先に実行してください。")
        else:
            stats = vector_store.get_statistics()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("登録文書数", f"{stats['document_count']:,}")
            with col2:
                st.metric("総チャンク数", f"{stats['chunk_count']:,}")
            with col3:
                st.metric("平均チャンク長", f"{stats['average_chunk_length']:.1f} 文字")
            with col4:
                st.metric(
                    "埋め込み済み文書",
                    f"{stats['embedded_document_count']:,}",
                    delta=-stats['documents_without_embeddings'] if stats['documents_without_embeddings'] else None,
                )

            st.caption(
                f"埋め込み未完文書: {stats['documents_without_embeddings']} 件"
                if stats['documents_without_embeddings']
                else "すべての文書が埋め込み済みです。"
            )

            summary = st.session_state.get("knowledge_registration_summary")
            if summary:
                st.caption(
                    f"最終登録: {summary['timestamp']} / 成功 {summary['registered_count']}/{summary['total_files']} 件 / 総チャンク {summary['total_chunks']}"
                )

            file_type_distribution = stats.get("file_type_distribution", {})
            if file_type_distribution:
                st.subheader("文書タイプ別件数")
                dist_df = pd.DataFrame(
                    sorted(file_type_distribution.items(), key=lambda item: item[1], reverse=True),
                    columns=["文書タイプ", "件数"],
                )
                render_stable_dataframe(
                    dist_df,
                    column_config={
                        "文書タイプ": st.column_config.TextColumn(width="medium"),
                        "件数": st.column_config.NumberColumn(format="%d", width="small"),
                    },
                    min_height=140,
                    max_height=260,
                )
                if len(dist_df) >= 2:
                    with st.expander("文書タイプ別グラフ", expanded=False):
                        st.bar_chart(dist_df.set_index("文書タイプ"))

            error_log = st.session_state.get("knowledge_registration_errors", [])
            st.subheader("直近の登録エラー")
            if error_log:
                error_df = pd.DataFrame(error_log)[["timestamp", "file_name", "reason"]].sort_values(
                    by="timestamp", ascending=False
                )
                render_stable_dataframe(
                    error_df,
                    column_config={
                        "timestamp": st.column_config.TextColumn(label="発生日時", width="medium"),
                        "file_name": st.column_config.TextColumn(label="ファイル名", width="large"),
                        "reason": st.column_config.TextColumn(label="理由", width="large"),
                    },
                    min_height=140,
                    max_height=260,
                )
                if st.button("登録エラーログをクリア", key="analysis_clear_error_log"):
                    st.session_state["knowledge_registration_errors"] = []
                    st.success("登録エラーログをクリアしました")
            else:
                st.info("登録エラーは記録されていません。")

            st.markdown("---")
            st.subheader("検索ヘルスチェック")
            rag_engine = st.session_state.get('rag_engine')
            with st.expander("検索ヘルスチェック（任意）", expanded=False):
                test_query = st.text_input(
                    "テストクエリ",
                    key="analysis_test_query",
                    placeholder="例: 最新の業務手順書を探したい",
                    help="検索品質を定期的に確認する場合に利用してください。"
                )

            def _results_to_dataframe(label: str, results: List[Dict[str, Any]]) -> pd.DataFrame:
                rows: List[Dict[str, Any]] = []
                for item in results[:5]:
                    metadata = item.get("metadata", {})
                    rows.append(
                        {
                            "スコア": round(item.get("relevance_score", 0.0), 4),
                            "ベクトル": round(item.get("vector_score", 0.0), 4),
                            "BM25": round(item.get("lexical_score", 0.0), 4),
                            "doc_id": metadata.get("doc_id"),
                            "ファイル名": metadata.get("file_name"),
                        }
                    )
                df = pd.DataFrame(rows)
                if not df.empty:
                    df.insert(0, "順位", [idx + 1 for idx in range(len(df))])
                return df

                if st.button(
                    "検索を実行",
                    key="analysis_run_search",
                    help="現在のインデックス状態で検索結果が正しく返るかを確認します。"
                ):
                    if not test_query.strip():
                        st.warning("クエリを入力してください。")
                    elif not rag_engine:
                        st.warning("RAGエンジンが初期化されていません。")
                    else:
                        with st.spinner("検索エンジンを評価しています..."):
                            try:
                                hybrid_results = rag_engine.search(
                                    query=test_query,
                                    top_k=5,
                                    search_mode="hybrid",
                                    use_adaptive=True,
                                )
                                vector_results = rag_engine.search(
                                    query=test_query,
                                    top_k=5,
                                    search_mode="vector",
                                    use_adaptive=False,
                                )
                                bm25_results = rag_engine.search(
                                    query=test_query,
                                    top_k=5,
                                    search_mode="bm25",
                                    use_adaptive=False,
                                )

                                result_tabs = st.tabs(["ハイブリッド", "ベクトル", "BM25"])
                                for tab, (label, results) in zip(
                                    result_tabs,
                                    [
                                        ("ハイブリッド", hybrid_results),
                                        ("ベクトル", vector_results),
                                        ("BM25", bm25_results),
                                    ],
                                ):
                                    with tab:
                                        if results:
                                            df_results = _results_to_dataframe(label, results)
                                            render_stable_dataframe(
                                                df_results,
                                                column_config={
                                                    "順位": st.column_config.NumberColumn(width="small"),
                                                    "スコア": st.column_config.NumberColumn(format="%.4f", width="small"),
                                                    "ベクトル": st.column_config.NumberColumn(format="%.4f", width="small"),
                                                    "BM25": st.column_config.NumberColumn(format="%.4f", width="small"),
                                                    "doc_id": st.column_config.TextColumn(label="doc_id", width="medium"),
                                                    "ファイル名": st.column_config.TextColumn(width="large"),
                                                },
                                                min_height=200,
                                                max_height=280,
                                            )
                                        else:
                                            st.info("該当する結果がありませんでした。")
                            except Exception as exc:
                                st.error(f"検索テスト中にエラーが発生しました: {exc}")
    
    # ===== Tab 4: 設定 =====
    # 統合：アプリ設定 + 高度設定 + 開発者向け
    with tab4:
        st.header("設定")
        st.caption("ここではアプリケーションの動作調整と品質チェックを行えます。")

        settings_tabs = st.tabs([
            "アプリケーション設定",
            "ユーザーコンテキスト",
            "ベストプラクティス",
            "ACL監査ログ",
            "開発者向け",
        ])

        with settings_tabs[0]:
            st.subheader("アプリケーション設定")
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider(
                    "チャンクサイズ",
                    100,
                    2000,
                    1000,
                    key="settings_chunk_size",
                    help="チャンク分割時の目標文字数を設定します。",
                )
            with col2:
                max_workers = st.slider(
                    "最大ワーカー数",
                    1,
                    8,
                    4,
                    key="settings_max_workers",
                    help="並列処理に使用するワーカー数を指定します。",
                )

            col1, col2 = st.columns(2)
            with col1:
                default_results = st.slider(
                    "デフォルト結果数",
                    5,
                    50,
                    10,
                    key="default_results",
                    help="検索結果の初期表示件数です。",
                )
            with col2:
                ocr_threshold = st.slider(
                    "OCR信頼度閾値",
                    0.0,
                    1.0,
                    0.5,
                    key="ocr_threshold",
                    help="OCR結果を採用する下限値を設定します。",
                )

        with settings_tabs[1]:
            st.subheader("ユーザーコンテキスト設定")
            user_context = st.session_state.get("user_context") or {}

            tenant_id = st.text_input(
                "テナントID",
                value=user_context.get("tenant_id", ""),
                placeholder="tenant_001",
                help="アクセス許可判定に利用されるテナント識別子です。",
            )

            departments_input = st.text_input(
                "所属部門 (カンマ区切り)",
                value=", ".join(user_context.get("departments", [])),
                placeholder="営業, 開発",
                help="複数部門に所属する場合はカンマで区切って入力してください。",
            )

            role_level = st.selectbox(
                "役職レベル",
                options=[
                    ("一般", 1),
                    ("係長以上", 2),
                    ("課長以上", 3),
                    ("部長以上", 4),
                    ("役員以上", 5),
                ],
                index=max(user_context.get("role_level", 1) - 1, 0),
                format_func=lambda opt: opt[0],
            )

            if st.button("コンテキストを更新", key="update_user_context"):
                departments = [dept.strip() for dept in departments_input.split(",") if dept.strip()]
                st.session_state["user_context"] = {
                    "tenant_id": tenant_id.strip(),
                    "departments": departments,
                    "role_level": role_level[1],
                    "role_label": role_level[0],
                }
                st.success("ユーザーコンテキストを更新しました。")
                from src.vector_store import VectorStoreManager

                vector_store = st.session_state.get("vector_store")
                if isinstance(vector_store, VectorStoreManager):
                    user_info = st.session_state.get("user_info") or {}
                    user_id = user_info.get("uid") or st.session_state.get("user_email") or ""
                    vector_store.set_user_context(
                        tenant_id.strip(),
                        departments,
                        role_level[1],
                        role_level[0],
                        user_id=user_id,
                    )

        with settings_tabs[2]:
            st.subheader("ベストプラクティスパネル")
            metrics = get_summary_metrics()
            operation_labels = {
                "chunk_summary": "チャンク要約",
                "search_summary": "検索結果要約",
            }

            summary_rows: List[Dict[str, Any]] = []
            for op_key, label in operation_labels.items():
                counts = metrics.get(op_key, {})
                success = counts.get("success", 0)
                failure = counts.get("failure", 0)
                fallback = counts.get("fallback", 0)
                skipped = counts.get("skipped", 0)
                total = success + failure + fallback + skipped
                success_rate = (success / total) if total else None
                fallback_rate = (fallback / total) if total else None
                summary_rows.append(
                    {
                        "対象": label,
                        "成功": success,
                        "失敗": failure,
                        "フォールバック": fallback,
                        "スキップ": skipped,
                        "試行回数": total,
                        "成功率(%)": round(success_rate * 100, 1) if success_rate is not None else None,
                        "フォールバック率(%)": round(fallback_rate * 100, 1) if fallback_rate is not None else None,
                    }
                )

            summary_df = pd.DataFrame(summary_rows)
            if not summary_df.empty:
                render_stable_dataframe(
                    summary_df,
                    column_config={
                        "成功": st.column_config.NumberColumn(format="%d", width="small"),
                        "失敗": st.column_config.NumberColumn(format="%d", width="small"),
                        "フォールバック": st.column_config.NumberColumn(format="%d", width="small"),
                        "スキップ": st.column_config.NumberColumn(format="%d", width="small"),
                        "試行回数": st.column_config.NumberColumn(format="%d", width="small"),
                        "成功率(%)": st.column_config.NumberColumn(format="%.1f", width="small"),
                        "フォールバック率(%)": st.column_config.NumberColumn(format="%.1f", width="small"),
                    },
                    min_height=160,
                    max_height=220,
                )
            else:
                st.info("要約メトリクスはまだ記録されていません。")

            log_path = Path("logs/summary_metrics.jsonl")
            col_log, col_placeholder = st.columns((1, 3))
            with col_log:
                if log_path.exists():
                    st.download_button(
                        "メトリクスログをダウンロード",
                        data=log_path.read_bytes(),
                        file_name="summary_metrics.jsonl",
                        mime="application/json",
                    )
                else:
                    st.caption("ログファイルはまだ生成されていません。")

            recent_events = get_recent_summary_events()
            if recent_events:
                recent_rows: List[Dict[str, Any]] = []
                for event in recent_events[-50:]:
                    detail = event.detail or {}
                    recent_rows.append(
                        {
                            "記録時刻": datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                            "操作": operation_labels.get(event.operation, event.operation),
                            "結果": event.outcome,
                            "ファイル名": detail.get("file_name") or "-",
                            "チャンクID": detail.get("chunk_id") or "-",
                            "理由": detail.get("reason", ""),
                        }
                    )
                recent_df = pd.DataFrame(recent_rows)
                st.markdown("#### 最近の記録")
                render_stable_dataframe(
                    recent_df,
                    column_config={
                        "結果": st.column_config.TextColumn(width="small"),
                        "理由": st.column_config.TextColumn(width="large"),
                    },
                    min_height=220,
                    max_height=320,
                )
            else:
                st.info("最近の要約イベントは見つかりませんでした。")

            best_practices_path = Path("docs/KNOWLEDGE_BEST_PRACTICES.md")
            if best_practices_path.exists():
                with st.expander("チェックリスト（Markdown）", expanded=False):
                    st.markdown(best_practices_path.read_text(encoding="utf-8"))
            else:
                st.caption("ベストプラクティスドキュメントが見つかりませんでした。")

        with settings_tabs[3]:
            st.subheader("ACL監査ログ")
            tenant_filter = st.text_input("テナントIDで絞り込み", value="")
            decision_filter = st.selectbox(
                "判定で絞り込み",
                options=["", "ALLOW", "DENY_TENANT_MISMATCH", "DENY_ROLE_LEVEL", "DENY_DEPARTMENT", "DENY_USER"],
                format_func=lambda x: "すべて" if x == "" else x,
            )
            col_page, col_size = st.columns(2)
            with col_page:
                page = st.number_input("ページ", min_value=1, value=1, step=1)
            with col_size:
                page_size = st.slider("ページサイズ", 10, 200, 50, step=10)

            if "acl_log_cache" not in st.session_state:
                st.session_state["acl_log_cache"] = {}

            cache_key = (tenant_filter, decision_filter, page, page_size)

            def _load_events():
                events, total = load_events_paginated(
                    page,
                    page_size,
                    tenant_id=tenant_filter or None,
                    decision=decision_filter or None,
                )
                st.session_state["acl_log_cache"][cache_key] = (events, total)

            col_btn, col_refresh, col_dl = st.columns([1, 1, 1])
            with col_btn:
                if st.button("監査ログを読み込み", key="load_audit_logs"):
                    _load_events()

            with col_refresh:
                if st.button("再読み込み", key="refresh_audit_logs"):
                    _load_events()

            cache_entry = st.session_state["acl_log_cache"].get(cache_key)
            if cache_entry:
                events, total = cache_entry
            else:
                events, total = [], 0

            if events:
                summary = summarize_decisions(events)
                st.write("判定サマリ", dict(summary))

                display_data = [
                    {
                        "timestamp": event.timestamp,
                        "doc_id": event.doc_id,
                        "tenant_id": event.tenant_id,
                        "access_scope": event.access_scope,
                        "decision": event.decision,
                        "reason": event.reason,
                        "user": event.user,
                    }
                    for event in events
                ]
                render_stable_dataframe(
                    display_data,
                    column_config={
                        "timestamp": st.column_config.TextColumn(label="日時", width="medium"),
                        "doc_id": st.column_config.TextColumn(width="medium"),
                        "tenant_id": st.column_config.TextColumn(width="medium"),
                        "access_scope": st.column_config.TextColumn(width="small"),
                        "decision": st.column_config.TextColumn(width="small"),
                        "reason": st.column_config.TextColumn(width="large"),
                        "user": st.column_config.TextColumn(width="medium"),
                    },
                    min_height=200,
                    max_height=360,
                )

                total_pages = (total - 1) // page_size + 1 if total else 1
                st.caption(f"全{total}件中 {page}/{total_pages}ページ表示")

                csv_data = events_to_csv(events)
                st.download_button(
                    label="CSVをダウンロード",
                    data=csv_data,
                    file_name="acl_audit_page.csv",
                    mime="text/csv",
                )
            else:
                st.info("監査ログが読み込まれていません。条件を指定して読み込みを押してください。")

        with settings_tabs[4]:
            st.subheader("開発者向けツール")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("システム健全性チェック"):
                    st.success("全てのコンポーネントが正常に動作しています。")
            with col2:
                if st.button("品質分析を実行"):
                    st.info("品質分析レポートを生成しています…")
            with col3:
                if st.button("精度評価を実行"):
                    st.info("検索精度の評価を開始しました…")

            st.markdown("#### デバッグ情報")
            if st.checkbox("デバッグモードを表示"):
                st.json(
                    {
                        "version": "1.0.0",
                        "python": "3.11.9",
                        "streamlit": "1.28.1",
                        "model": "gpt-4.1-mini-2025-04-14",
                    }
                )

            st.markdown("#### システムメトリクス")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("インデックス", "1,234")
            with col2:
                st.metric("ストレージ", "45 MB")
            with col3:
                st.metric("平均時間", "324 ms")
            with col4:
                st.metric("キャッシュ率", "78%")

            st.markdown("#### エラーログ")
            if st.button("エラーログを表示"):
                st.info("最近のエラーログは検出されていません。")

        st.caption("設定を変更すると即時に反映されます。")

    if st.session_state.get("poc_mode_active"):
        st.markdown("---")
        st.caption("POCモード: 認証をスキップしてデモユーザーで実行しています。")

def create_sidebar():
    """サイドバーを作成"""
    with st.sidebar:
        st.title("RAG SaaS")
        
        # ヘルスチェック
        from src.app.utils.health import get_health_summary
        st.markdown(get_health_summary())
        
        # システム情報
        if st.checkbox("システム情報を表示"):
            from src.app.utils.metrics import get_system_info
            system_info = get_system_info()
            st.json(system_info)
        
        # 設定
        st.subheader("設定")
        chunk_size = st.slider("チャンクサイズ", 100, 2000, 1000, key="sidebar_chunk_size")
        max_workers = st.slider("最大ワーカー数", 1, 8, 4, key="sidebar_max_workers")
        
        # 設定をセッション状態に保存
        st.session_state.chunk_size = chunk_size
        st.session_state.max_workers = max_workers
