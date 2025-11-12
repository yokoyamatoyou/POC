"""
チャットUI
"""
import streamlit as st
from typing import Optional, List, Dict, Any
from src.app.utils.logging import get_logger

logger = get_logger(__name__)

def chat_ui(rag_engine, vector_store):
    """チャットUI"""
    st.subheader("AIチャット")
    
    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # チャット入力
    if prompt := st.chat_input("質問を入力してください"):
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI回答を生成
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                try:
                    # RAGエンジンで回答を生成
                    response = rag_engine.generate_answer(prompt)
                    
                    # 回答を表示
                    st.markdown(response.answer)
                    
                    # 引用情報を表示
                    if hasattr(response, 'citations') and response.citations:
                        with st.expander("引用情報"):
                            for i, citation in enumerate(response.citations, 1):
                                st.write(f"**引用 {i}**: {citation.get('source', 'N/A')}")
                                st.write(f"関連性: {citation.get('relevance', 0):.2f}")
                    
                    # AIメッセージを履歴に追加
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.answer
                    })
                    
                except Exception as e:
                    st.error(f"回答生成中にエラーが発生しました: {e}")
                    logger.error(f"チャット回答生成エラー: {e}")
    
    # チャット履歴の管理
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("履歴をクリア"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("履歴をエクスポート"):
            _export_chat_history()

def create_answer_generation_ui(client, vector_store):
    """回答生成UI"""
    st.subheader("高度な回答生成")
    
    # 質問入力
    question = st.text_area(
        "質問を入力してください",
        height=100,
        help="詳細な質問を入力すると、より精度の高い回答が生成されます"
    )
    
    # 生成オプション
    col1, col2 = st.columns(2)
    
    with col1:
        max_tokens = st.slider(
            "最大トークン数",
            min_value=100,
            max_value=4000,
            value=1000,
            help="生成する回答の最大長"
        )
        
        temperature = st.slider(
            "創造性",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="回答の創造性レベル（0: 保守的、1: 創造的）"
        )
    
    with col2:
        top_k = st.slider(
            "関連文書数",
            min_value=1,
            max_value=20,
            value=5,
            help="検索する関連文書の数"
        )
        
        include_metadata = st.checkbox(
            "メタデータを含める",
            value=True,
            help="回答にメタデータを含める"
        )
    
    # 生成実行
    if st.button("回答を生成", type="primary"):
        if question:
            _generate_advanced_answer(
                question,
                client,
                vector_store,
                max_tokens,
                temperature,
                top_k,
                include_metadata
            )
        else:
            st.warning("質問を入力してください")

def _generate_advanced_answer(
    question: str,
    client,
    vector_store,
    max_tokens: int,
    temperature: float,
    top_k: int,
    include_metadata: bool
):
    """高度な回答を生成"""
    
    with st.spinner("高度な回答を生成中..."):
        try:
            # 関連文書を検索
            relevant_docs = vector_store.search(question, top_k=top_k)
            
            # コンテキストを構築
            context = "\n\n".join([doc.get('content', '') for doc in relevant_docs])
            
            # プロンプトを構築
            prompt = f"""
以下のコンテキストに基づいて、質問に回答してください。

コンテキスト:
{context}

質問: {question}

回答:
"""
            
            # OpenAI APIで回答を生成
            response = client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[
                    {"role": "system", "content": "あなたは専門的な知識を持つアシスタントです。提供されたコンテキストに基づいて、正確で有用な回答を提供してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            
            # 回答を表示
            st.subheader("生成された回答")
            st.markdown(answer)
            
            # 関連文書を表示
            if relevant_docs:
                st.subheader("参考文書")
                for i, doc in enumerate(relevant_docs, 1):
                    with st.expander(f"文書 {i}: {doc.get('file_name', 'N/A')}"):
                        st.write(f"**関連性**: {doc.get('relevance_score', 0):.2f}")
                        st.write(f"**内容**: {doc.get('content', 'N/A')[:500]}...")
                        
                        if include_metadata and doc.get('metadata'):
                            st.write("**メタデータ**:")
                            st.json(doc['metadata'])
            
        except Exception as e:
            st.error(f"回答生成中にエラーが発生しました: {e}")
            logger.error(f"高度回答生成エラー: {e}")

def display_generated_answer(answer):
    """生成された回答の表示"""
    # 信頼度スコアの表示
    if hasattr(answer, 'confidence_score') and answer.confidence_score > 0:
        confidence_color = "green" if answer.confidence_score > 0.7 else "orange" if answer.confidence_score > 0.4 else "red"
        st.markdown(f"**信頼度: <span style='color: {confidence_color}'>{answer.confidence_score:.1f}%</span>**", unsafe_allow_html=True)
    
    # 要点の表示
    if hasattr(answer, 'summary') and answer.summary:
        st.subheader("要点")
        st.write(answer.summary)
    
    # 詳細の表示
    if hasattr(answer, 'details') and answer.details:
        st.subheader("詳細")
        st.write(answer.details)
    
    # 手順の表示
    if hasattr(answer, 'procedures') and answer.procedures:
        st.subheader("手順・注意事項")
        st.write(answer.procedures)
    
    # 引用の表示
    if hasattr(answer, 'citations') and answer.citations:
        st.subheader("引用情報")
        
        for i, citation in enumerate(answer.citations, 1):
            with st.expander(f"引用 {i}: {citation.get('source_file', 'N/A')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**ファイル**: {citation.get('source_file', 'N/A')}")
                    if citation.get('section'):
                        st.write(f"**セクション**: {citation['section']}")
                    if citation.get('page'):
                        st.write(f"**ページ**: {citation['page']}")
                    if citation.get('sheet_name'):
                        st.write(f"**シート**: {citation['sheet_name']}")
                
                with col2:
                    if citation.get('version'):
                        st.write(f"**版**: {citation['version']}")
                    if citation.get('last_modified'):
                        st.write(f"**更新日**: {citation['last_modified']}")
                    st.write(f"**関連性**: {citation.get('relevance_score', 0):.2f}")
    
    # メタデータの表示（デバッグ用）
    if hasattr(answer, 'metadata') and st.checkbox("メタデータを表示"):
        st.json(answer.metadata)

def _export_chat_history():
    """チャット履歴をエクスポート"""
    try:
        import json
        from datetime import datetime
        
        # チャット履歴をJSON形式でエクスポート
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "messages": st.session_state.messages
        }
        
        # ダウンロードボタンを表示
        st.download_button(
            label="チャット履歴をダウンロード",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"チャット履歴のエクスポート中にエラーが発生しました: {e}")
        logger.error(f"チャット履歴エクスポートエラー: {e}")
