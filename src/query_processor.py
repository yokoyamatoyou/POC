"""
Phase 4: クエリ処理エンジン
自然言語クエリの処理・拡張・最適化
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import json

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spacyが利用できません")

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """クエリタイプ"""
    FACTUAL = "factual"          # 事実確認
    PROCEDURAL = "procedural"    # 手順・方法
    COMPARATIVE = "comparative"  # 比較
    ANALYTICAL = "analytical"    # 分析
    DESCRIPTIVE = "descriptive"  # 説明
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """クエリ複雑度"""
    SIMPLE = "simple"      # 単純
    MODERATE = "moderate"  # 中程度
    COMPLEX = "complex"    # 複雑


@dataclass
class ProcessedQuery:
    """処理済みクエリ"""
    original_query: str
    processed_query: str
    query_type: QueryType
    complexity: QueryComplexity
    keywords: List[str]
    entities: List[str]
    intent: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class QueryExpansion:
    """クエリ拡張"""
    synonyms: List[str]
    related_terms: List[str]
    context_terms: List[str]
    technical_terms: List[str]


class QueryProcessor:
    """クエリ処理エンジン"""
    
    def __init__(self):
        """初期化"""
        self.nlp = None
        self.query_patterns = self._load_query_patterns()
        self.technical_terms = self._load_technical_terms()
        
        # モデルの初期化
        self._initialize_models()
        
        # 処理統計
        self.processing_stats = {
            "total_queries": 0,
            "average_processing_time": 0.0,
            "query_type_distribution": {},
            "complexity_distribution": {},
            "average_confidence": 0.0
        }
    
    def _initialize_models(self):
        """モデルの初期化"""
        try:
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("ja_core_news_sm")
                    logger.info("spaCy日本語モデルを初期化")
                except OSError:
                    try:
                        self.nlp = spacy.load("ja_core_news_md")
                        logger.info("spaCy日本語モデル（中）を初期化")
                    except OSError:
                        logger.warning("spaCy日本語モデルが利用できません。")
        except Exception as e:
            logger.error(f"モデル初期化エラー: {e}")
    
    def _load_query_patterns(self) -> Dict[str, List[str]]:
        """クエリパターンの読み込み"""
        return {
            "factual": [
                r"(.+)とは何か",
                r"(.+)の定義",
                r"(.+)について教えて",
                r"(.+)とは",
                r"(.+)の意味"
            ],
            "procedural": [
                r"(.+)の方法",
                r"(.+)の手順",
                r"(.+)のやり方",
                r"(.+)を(.+)するには",
                r"(.+)の(.+)方法"
            ],
            "comparative": [
                r"(.+)と(.+)の違い",
                r"(.+)と(.+)の比較",
                r"(.+)と(.+)どちらが",
                r"(.+)と(.+)の関係"
            ],
            "analytical": [
                r"(.+)の原因",
                r"(.+)の理由",
                r"(.+)の影響",
                r"(.+)の効果",
                r"(.+)の分析"
            ],
            "descriptive": [
                r"(.+)の特徴",
                r"(.+)の性質",
                r"(.+)の説明",
                r"(.+)について"
            ]
        }
    
    def _load_technical_terms(self) -> Dict[str, List[str]]:
        """技術用語辞書の読み込み"""
        return {
            "技術": ["API", "データベース", "アルゴリズム", "システム", "ソフトウェア"],
            "ビジネス": ["売上", "利益", "コスト", "マーケティング", "戦略"],
            "法務": ["契約", "規約", "法律", "権利", "義務"],
            "人事": ["採用", "評価", "研修", "給与", "福利厚生"]
        }
    
    def process_query(self, query: str, context: Optional[str] = None) -> ProcessedQuery:
        """
        クエリの処理
        
        Args:
            query: 元のクエリ
            context: コンテキスト情報
            
        Returns:
            ProcessedQuery: 処理済みクエリ
        """
        start_time = time.time()
        
        try:
            # 基本的な前処理
            processed_query = self._preprocess_query(query)
            
            # クエリタイプの判定
            query_type = self._classify_query_type(processed_query)
            
            # 複雑度の判定
            complexity = self._assess_complexity(processed_query)
            
            # キーワードの抽出
            keywords = self._extract_keywords(processed_query)
            
            # エンティティの抽出
            entities = self._extract_entities(processed_query)
            
            # 意図の分析
            intent = self._analyze_intent(processed_query, query_type)
            
            # 信頼度の計算
            confidence = self._calculate_confidence(processed_query, query_type, keywords)
            
            # クエリの拡張
            if context:
                processed_query = self._expand_query_with_context(processed_query, context)
            
            # メタデータの生成
            metadata = {
                "original_length": len(query),
                "processed_length": len(processed_query),
                "keyword_count": len(keywords),
                "entity_count": len(entities),
                "has_context": context is not None
            }
            
            processing_time = time.time() - start_time
            
            # 統計の更新
            self._update_stats(query_type, complexity, confidence, processing_time)
            
            return ProcessedQuery(
                original_query=query,
                processed_query=processed_query,
                query_type=query_type,
                complexity=complexity,
                keywords=keywords,
                entities=entities,
                intent=intent,
                confidence=confidence,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"クエリ処理エラー: {e}")
            return ProcessedQuery(
                original_query=query,
                processed_query=query,
                query_type=QueryType.UNKNOWN,
                complexity=QueryComplexity.SIMPLE,
                keywords=[],
                entities=[],
                intent="unknown",
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={}
            )
    
    def _preprocess_query(self, query: str) -> str:
        """クエリの前処理"""
        # 空白の正規化
        processed = re.sub(r'\s+', ' ', query.strip())
        
        # 特殊文字の処理
        processed = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', ' ', processed)
        
        # 重複語の除去
        words = processed.split()
        unique_words = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        
        return ' '.join(unique_words)
    
    def _classify_query_type(self, query: str) -> QueryType:
        """クエリタイプの分類"""
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return QueryType(query_type)
        
        return QueryType.UNKNOWN
    
    def _assess_complexity(self, query: str) -> QueryComplexity:
        """クエリ複雑度の評価"""
        word_count = len(query.split())
        
        if word_count <= 3:
            return QueryComplexity.SIMPLE
        elif word_count <= 8:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
    
    def _extract_keywords(self, query: str) -> List[str]:
        """キーワードの抽出"""
        if not self.nlp:
            # spaCyが利用できない場合は単純な分割
            return [word for word in query.split() if len(word) > 1]
        
        try:
            doc = self.nlp(query)
            keywords = []
            
            for token in doc:
                # 名詞、動詞、形容詞を抽出
                if token.pos_ in ["NOUN", "VERB", "ADJ"] and not token.is_stop and not token.is_punct:
                    keywords.append(token.lemma_)
            
            return keywords
            
        except Exception as e:
            logger.warning(f"キーワード抽出エラー: {e}")
            return [word for word in query.split() if len(word) > 1]
    
    def _extract_entities(self, query: str) -> List[str]:
        """エンティティの抽出"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(query)
            entities = []
            
            for ent in doc.ents:
                entities.append(ent.text)
            
            return entities
            
        except Exception as e:
            logger.warning(f"エンティティ抽出エラー: {e}")
            return []
    
    def _analyze_intent(self, query: str, query_type: QueryType) -> str:
        """意図の分析"""
        intent_mapping = {
            QueryType.FACTUAL: "事実確認",
            QueryType.PROCEDURAL: "手順確認",
            QueryType.COMPARATIVE: "比較検討",
            QueryType.ANALYTICAL: "分析要求",
            QueryType.DESCRIPTIVE: "説明要求",
            QueryType.UNKNOWN: "不明"
        }
        
        return intent_mapping.get(query_type, "不明")
    
    def _calculate_confidence(self, query: str, query_type: QueryType, keywords: List[str]) -> float:
        """信頼度の計算"""
        confidence = 0.0
        
        # クエリタイプが特定できた場合
        if query_type != QueryType.UNKNOWN:
            confidence += 0.4
        
        # キーワードが適切に抽出できた場合
        if len(keywords) >= 2:
            confidence += 0.3
        elif len(keywords) >= 1:
            confidence += 0.2
        
        # クエリの長さが適切な場合
        word_count = len(query.split())
        if 3 <= word_count <= 10:
            confidence += 0.2
        elif word_count > 10:
            confidence += 0.1
        
        # 技術用語が含まれている場合
        for category, terms in self.technical_terms.items():
            if any(term in query for term in terms):
                confidence += 0.1
                break
        
        return min(confidence, 1.0)
    
    def _expand_query_with_context(self, query: str, context: str) -> str:
        """コンテキストを使ったクエリ拡張"""
        try:
            if not self.nlp:
                return query
            
            # コンテキストから関連語を抽出
            context_doc = self.nlp(context)
            context_keywords = []
            
            for token in context_doc:
                if token.pos_ in ["NOUN", "VERB", "ADJ"] and not token.is_stop:
                    context_keywords.append(token.lemma_)
            
            # クエリにコンテキストキーワードを追加
            if context_keywords:
                expanded_query = f"{query} {' '.join(context_keywords[:3])}"
                return expanded_query
            
        except Exception as e:
            logger.warning(f"クエリ拡張エラー: {e}")
        
        return query
    
    def expand_query(self, query: str) -> QueryExpansion:
        """
        クエリの拡張
        
        Args:
            query: 元のクエリ
            
        Returns:
            QueryExpansion: 拡張結果
        """
        synonyms: list[str] = []
        related_terms: list[str] = []
        context_terms: list[str] = []
        technical_terms: list[str] = []
        
        try:
            if self.nlp:
                doc = self.nlp(query)
                
                # 同義語の抽出（簡易版）
                for token in doc:
                    if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                        # 語幹を取得
                        lemma = token.lemma_
                        if lemma != token.text:
                            synonyms.append(lemma)
                
                # 関連語の抽出
                for token in doc:
                    if token.pos_ == "NOUN":
                        # 同じ品詞の語を関連語として追加
                        related_terms.append(token.text)
            
            # 技術用語の検出
            for category, terms in self.technical_terms.items():
                for term in terms:
                    if term in query:
                        technical_terms.append(term)
            
        except Exception as e:
            logger.warning(f"クエリ拡張エラー: {e}")
        
        return QueryExpansion(
            synonyms=synonyms,
            related_terms=related_terms,
            context_terms=context_terms,
            technical_terms=technical_terms
        )
    
    def optimize_query_for_search(self, processed_query: ProcessedQuery) -> str:
        """
        検索用クエリの最適化
        
        Args:
            processed_query: 処理済みクエリ
            
        Returns:
            str: 最適化されたクエリ
        """
        query = processed_query.processed_query
        
        # クエリタイプに応じた最適化
        if processed_query.query_type == QueryType.FACTUAL:
            # 事実確認クエリはキーワードを強調
            keywords = processed_query.keywords
            if keywords:
                query = f"{' '.join(keywords)} {query}"
        
        elif processed_query.query_type == QueryType.PROCEDURAL:
            # 手順クエリは動詞を強調
            query = f"方法 手順 {query}"
        
        elif processed_query.query_type == QueryType.COMPARATIVE:
            # 比較クエリは比較語を追加
            query = f"比較 違い {query}"
        
        # エンティティがある場合は追加
        if processed_query.entities:
            query = f"{query} {' '.join(processed_query.entities)}"
        
        return query
    
    def _update_stats(self, query_type: QueryType, complexity: QueryComplexity, confidence: float, processing_time: float):
        """統計の更新"""
        self.processing_stats["total_queries"] += 1
        
        # クエリタイプ分布
        type_key = query_type.value
        self.processing_stats["query_type_distribution"][type_key] = \
            self.processing_stats["query_type_distribution"].get(type_key, 0) + 1
        
        # 複雑度分布
        complexity_key = complexity.value
        self.processing_stats["complexity_distribution"][complexity_key] = \
            self.processing_stats["complexity_distribution"].get(complexity_key, 0) + 1
        
        # 平均処理時間の更新
        current_avg = self.processing_stats["average_processing_time"]
        new_avg = (current_avg * (self.processing_stats["total_queries"] - 1) + processing_time) / self.processing_stats["total_queries"]
        self.processing_stats["average_processing_time"] = new_avg
        
        # 平均信頼度の更新
        current_avg_conf = self.processing_stats["average_confidence"]
        new_avg_conf = (current_avg_conf * (self.processing_stats["total_queries"] - 1) + confidence) / self.processing_stats["total_queries"]
        self.processing_stats["average_confidence"] = new_avg_conf
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """処理統計の取得"""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """統計のリセット"""
        self.processing_stats = {
            "total_queries": 0,
            "average_processing_time": 0.0,
            "query_type_distribution": {},
            "complexity_distribution": {},
            "average_confidence": 0.0
        }
    
    def export_query_analysis(self, output_path: str) -> bool:
        """クエリ分析結果のエクスポート"""
        try:
            analysis_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_stats": self.get_processing_stats(),
                "query_patterns": self.query_patterns,
                "technical_terms": self.technical_terms,
                "spacy_available": SPACY_AVAILABLE
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"分析結果エクスポートエラー: {e}")
            return False
