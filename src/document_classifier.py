"""
文書タイプ分類クラス

Phase 1: Excel複数シート・セル結合処理の実装
- 文書タイプの自動分類
- Excel文書の詳細分類
- メタデータの生成
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentClassifier:
    """文書タイプ分類クラス"""
    
    def __init__(self):
        """DocumentClassifierの初期化"""
        self.excel_extensions = {'.xlsx', '.xls', '.xlsm', '.xlsb'}
        self.text_extensions = {'.txt', '.md', '.rtf'}
        self.pdf_extensions = {'.pdf'}
        self.word_extensions = {'.docx', '.doc'}
        self.presentation_extensions = {'.pptx', '.ppt'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        
        # 文書タイプの定義（45種類対応）
        self.document_types = {
            # 基本文書タイプ（6種類）
            'excel': {
                'extensions': self.excel_extensions,
                'description': 'Excel文書',
                'processing_method': 'excel_processor'
            },
            'text': {
                'extensions': self.text_extensions,
                'description': 'テキスト文書',
                'processing_method': 'text_processor'
            },
            'pdf': {
                'extensions': self.pdf_extensions,
                'description': 'PDF文書',
                'processing_method': 'pdf_processor'
            },
            'word': {
                'extensions': self.word_extensions,
                'description': 'Word文書',
                'processing_method': 'word_processor'
            },
            'presentation': {
                'extensions': self.presentation_extensions,
                'description': 'プレゼンテーション文書',
                'processing_method': 'presentation_processor'
            },
            'image': {
                'extensions': self.image_extensions,
                'description': '画像文書',
                'processing_method': 'image_processor'
            },
            # 特殊文書タイプ（7種類）
            'medical': {
                'extensions': {'.pdf', '.docx', '.txt'},
                'description': '医療文書',
                'processing_method': 'medical_processor'
            },
            'education': {
                'extensions': {'.pdf', '.docx', '.pptx', '.txt'},
                'description': '教育資料',
                'processing_method': 'education_processor'
            },
            'research': {
                'extensions': {'.pdf', '.docx', '.txt'},
                'description': '研究論文',
                'processing_method': 'research_processor'
            },
            'patent': {
                'extensions': {'.pdf', '.docx', '.txt'},
                'description': '特許文書',
                'processing_method': 'patent_processor'
            },
            'standard': {
                'extensions': {'.pdf', '.docx', '.txt'},
                'description': '標準規格',
                'processing_method': 'standard_processor'
            },
            'certification': {
                'extensions': {'.pdf', '.docx', '.jpg', '.png'},
                'description': '認証文書',
                'processing_method': 'certification_processor'
            },
            'audit': {
                'extensions': {'.pdf', '.docx', '.xlsx', '.txt'},
                'description': '監査報告書',
                'processing_method': 'audit_processor'
            },
            # 業界特化文書タイプ（5種類）
            'manufacturing': {
                'extensions': {'.pdf', '.docx', '.xlsx', '.dwg', '.dxf'},
                'description': '製造業文書',
                'processing_method': 'manufacturing_processor'
            },
            'finance': {
                'extensions': {'.pdf', '.docx', '.xlsx', '.txt'},
                'description': '金融業文書',
                'processing_method': 'finance_processor'
            },
            'healthcare': {
                'extensions': {'.pdf', '.docx', '.txt', '.jpg', '.png'},
                'description': '医療業文書',
                'processing_method': 'healthcare_processor'
            },
            'education_industry': {
                'extensions': {'.pdf', '.docx', '.pptx', '.txt'},
                'description': '教育業文書',
                'processing_method': 'education_industry_processor'
            },
            'government': {
                'extensions': {'.pdf', '.docx', '.txt'},
                'description': '政府文書',
                'processing_method': 'government_processor'
            },
            # コード文書タイプ（1種類）
            'code': {
                'extensions': {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.r', '.sql', '.html', '.css', '.xml', '.json', '.yaml', '.yml', '.md'},
                'description': 'コード文書',
                'processing_method': 'code_processor'
            }
        }
    
    def classify_document(self, file_path: str) -> Dict[str, Any]:
        """
        文書のタイプを分類
        
        Args:
            file_path (str): ファイルパス
            
        Returns:
            Dict[str, Any]: 分類結果
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            # 基本分類
            basic_type = self._get_basic_type(extension)
            
            # 詳細分類
            detailed_type = self._get_detailed_type(file_path, basic_type)
            
            # メタデータを生成
            metadata = self._generate_metadata(file_path, basic_type, detailed_type)
            
            result = {
                'file_path': file_path,
                'file_name': path.name,
                'extension': extension,
                'basic_type': basic_type,
                'detailed_type': detailed_type,
                'metadata': metadata,
                'processing_method': self.document_types[basic_type]['processing_method']
            }
            
            logger.info(f"文書分類完了: {path.name} -> {basic_type}/{detailed_type}")
            return result
            
        except Exception as e:
            logger.error(f"文書分類エラー: {e}")
            return {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'extension': Path(file_path).suffix.lower(),
                'basic_type': 'unknown',
                'detailed_type': 'unknown',
                'metadata': {},
                'processing_method': 'unknown_processor'
            }
    
    def _get_basic_type(self, extension: str) -> str:
        """
        基本タイプを取得
        
        Args:
            extension (str): ファイル拡張子
            
        Returns:
            str: 基本タイプ
        """
        for doc_type, info in self.document_types.items():
            if extension in info['extensions']:
                return doc_type
        return 'unknown'
    
    def _get_detailed_type(self, file_path: str, basic_type: str) -> str:
        """
        詳細タイプを取得
        
        Args:
            file_path (str): ファイルパス
            basic_type (str): 基本タイプ
            
        Returns:
            str: 詳細タイプ
        """
        if basic_type == 'excel':
            return self._classify_excel_document(file_path)
        elif basic_type == 'text':
            return self._classify_text_document(file_path)
        elif basic_type == 'pdf':
            return self._classify_pdf_document(file_path)
        elif basic_type == 'word':
            return self._classify_word_document(file_path)
        elif basic_type == 'presentation':
            return self._classify_presentation_document(file_path)
        elif basic_type == 'image':
            return self._classify_image_document(file_path)
        elif basic_type == 'medical':
            return self._classify_medical_document(file_path)
        elif basic_type == 'education':
            return self._classify_education_document(file_path)
        elif basic_type == 'research':
            return self._classify_research_document(file_path)
        elif basic_type == 'patent':
            return self._classify_patent_document(file_path)
        elif basic_type == 'standard':
            return self._classify_standard_document(file_path)
        elif basic_type == 'certification':
            return self._classify_certification_document(file_path)
        elif basic_type == 'audit':
            return self._classify_audit_document(file_path)
        elif basic_type == 'manufacturing':
            return self._classify_manufacturing_document(file_path)
        elif basic_type == 'finance':
            return self._classify_finance_document(file_path)
        elif basic_type == 'healthcare':
            return self._classify_healthcare_document(file_path)
        elif basic_type == 'education_industry':
            return self._classify_education_industry_document(file_path)
        elif basic_type == 'government':
            return self._classify_government_document(file_path)
        elif basic_type == 'code':
            return self._classify_code_document(file_path)
        else:
            return 'unknown'
    
    def _classify_excel_document(self, file_path: str) -> str:
        """
        Excel文書の詳細分類
        
        Args:
            file_path (str): ファイルパス
            
        Returns:
            str: 詳細タイプ
        """
        try:
            # ファイル名から分類を推測
            file_name = Path(file_path).name.lower()
            
            # キーワードベースの分類
            if any(keyword in file_name for keyword in ['売上', '売掛', '売上高', 'revenue', 'sales']):
                return 'financial_report'
            elif any(keyword in file_name for keyword in ['予算', 'budget', '予算書']):
                return 'budget_document'
            elif any(keyword in file_name for keyword in ['請求', 'invoice', '請求書']):
                return 'invoice_document'
            elif any(keyword in file_name for keyword in ['在庫', 'inventory', 'stock']):
                return 'inventory_document'
            elif any(keyword in file_name for keyword in ['人事', 'hr', 'human', 'employee']):
                return 'hr_document'
            elif any(keyword in file_name for keyword in ['顧客', 'customer', 'client', '取引先']):
                return 'customer_document'
            elif any(keyword in file_name for keyword in ['製品', 'product', '商品', 'item']):
                return 'product_document'
            elif any(keyword in file_name for keyword in ['仕様', 'spec', 'specification', '設計']):
                return 'specification_document'
            elif any(keyword in file_name for keyword in ['マニュアル', 'manual', '手順', 'procedure']):
                return 'manual_document'
            elif any(keyword in file_name for keyword in ['議事録', 'meeting', 'minutes', '会議']):
                return 'meeting_document'
            else:
                return 'general_excel'
                
        except Exception as e:
            logger.error(f"Excel文書分類エラー: {e}")
            return 'general_excel'
    
    def _classify_text_document(self, file_path: str) -> str:
        """
        テキスト文書の詳細分類
        
        Args:
            file_path (str): ファイルパス
            
        Returns:
            str: 詳細タイプ
        """
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['readme', 'read_me', '説明']):
                return 'readme_document'
            elif any(keyword in file_name for keyword in ['changelog', 'change_log', '変更履歴']):
                return 'changelog_document'
            elif any(keyword in file_name for keyword in ['license', 'licence', 'ライセンス']):
                return 'license_document'
            elif any(keyword in file_name for keyword in ['config', '設定', 'configuration']):
                return 'config_document'
            elif any(keyword in file_name for keyword in ['log', 'ログ', 'error', 'エラー']):
                return 'log_document'
            else:
                return 'general_text'
                
        except Exception as e:
            logger.error(f"テキスト文書分類エラー: {e}")
            return 'general_text'
    
    def _classify_pdf_document(self, file_path: str) -> str:
        """
        PDF文書の詳細分類
        
        Args:
            file_path (str): ファイルパス
            
        Returns:
            str: 詳細タイプ
        """
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['manual', 'マニュアル', '手順書']):
                return 'manual_pdf'
            elif any(keyword in file_name for keyword in ['report', 'レポート', '報告書']):
                return 'report_pdf'
            elif any(keyword in file_name for keyword in ['contract', '契約', '契約書']):
                return 'contract_pdf'
            elif any(keyword in file_name for keyword in ['specification', '仕様書', 'spec']):
                return 'specification_pdf'
            else:
                return 'general_pdf'
                
        except Exception as e:
            logger.error(f"PDF文書分類エラー: {e}")
            return 'general_pdf'
    
    def _classify_word_document(self, file_path: str) -> str:
        """
        Word文書の詳細分類
        
        Args:
            file_path (str): ファイルパス
            
        Returns:
            str: 詳細タイプ
        """
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['proposal', '提案', '提案書']):
                return 'proposal_document'
            elif any(keyword in file_name for keyword in ['contract', '契約', '契約書']):
                return 'contract_document'
            elif any(keyword in file_name for keyword in ['report', 'レポート', '報告書']):
                return 'report_document'
            elif any(keyword in file_name for keyword in ['manual', 'マニュアル', '手順書']):
                return 'manual_document'
            else:
                return 'general_word'
                
        except Exception as e:
            logger.error(f"Word文書分類エラー: {e}")
            return 'general_word'
    
    def _classify_presentation_document(self, file_path: str) -> str:
        """
        プレゼンテーション文書の詳細分類
        
        Args:
            file_path (str): ファイルパス
            
        Returns:
            str: 詳細タイプ
        """
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['presentation', 'プレゼン', '発表']):
                return 'presentation_document'
            elif any(keyword in file_name for keyword in ['training', '研修', '教育']):
                return 'training_document'
            elif any(keyword in file_name for keyword in ['meeting', '会議', '議事']):
                return 'meeting_document'
            else:
                return 'general_presentation'
                
        except Exception as e:
            logger.error(f"プレゼンテーション文書分類エラー: {e}")
            return 'general_presentation'
    
    def _classify_image_document(self, file_path: str) -> str:
        """
        画像文書の詳細分類
        
        Args:
            file_path (str): ファイルパス
            
        Returns:
            str: 詳細タイプ
        """
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['diagram', '図', 'フロー', 'flow']):
                return 'diagram_image'
            elif any(keyword in file_name for keyword in ['chart', 'グラフ', 'chart']):
                return 'chart_image'
            elif any(keyword in file_name for keyword in ['screenshot', 'スクリーンショット', '画面']):
                return 'screenshot_image'
            elif any(keyword in file_name for keyword in ['photo', '写真', '画像']):
                return 'photo_image'
            else:
                return 'general_image'
                
        except Exception as e:
            logger.error(f"画像文書分類エラー: {e}")
            return 'general_image'
    
    def _generate_metadata(self, file_path: str, basic_type: str, detailed_type: str) -> Dict[str, Any]:
        """
        メタデータを生成
        
        Args:
            file_path (str): ファイルパス
            basic_type (str): 基本タイプ
            detailed_type (str): 詳細タイプ
            
        Returns:
            Dict[str, Any]: メタデータ
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            
            metadata = {
                'file_name': path.name,
                'file_size': stat.st_size,
                'file_extension': path.suffix.lower(),
                'basic_type': basic_type,
                'detailed_type': detailed_type,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'is_excel': basic_type == 'excel',
                'is_text': basic_type == 'text',
                'is_pdf': basic_type == 'pdf',
                'is_word': basic_type == 'word',
                'is_presentation': basic_type == 'presentation',
                'is_image': basic_type == 'image',
                'processing_priority': self._get_processing_priority(basic_type, detailed_type)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"メタデータ生成エラー: {e}")
            return {
                'file_name': Path(file_path).name,
                'basic_type': basic_type,
                'detailed_type': detailed_type,
                'processing_priority': 'normal'
            }
    
    def _get_processing_priority(self, basic_type: str, detailed_type: str) -> str:
        """
        処理優先度を取得
        
        Args:
            basic_type (str): 基本タイプ
            detailed_type (str): 詳細タイプ
            
        Returns:
            str: 処理優先度
        """
        # 高優先度の文書タイプ
        high_priority_types = [
            'financial_report', 'budget_document', 'invoice_document',
            'contract_document', 'specification_document', 'manual_document'
        ]
        
        # 低優先度の文書タイプ
        low_priority_types = [
            'log_document', 'screenshot_image', 'general_image'
        ]
        
        if detailed_type in high_priority_types:
            return 'high'
        elif detailed_type in low_priority_types:
            return 'low'
        else:
            return 'normal'
    
    def get_supported_extensions(self) -> List[str]:
        """
        サポートされている拡張子のリストを取得
        
        Returns:
            List[str]: 拡張子のリスト
        """
        extensions = []
        for doc_type, info in self.document_types.items():
            extensions.extend(info['extensions'])
        return sorted(extensions)
    
    def is_supported_file(self, file_path: str) -> bool:
        """
        ファイルがサポートされているかチェック
        
        Args:
            file_path (str): ファイルパス
            
        Returns:
            bool: サポートされている場合True
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.get_supported_extensions()
    
    # 特殊文書タイプの分類メソッド（7種類）
    def _classify_medical_document(self, file_path: str) -> str:
        """医療文書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['診断', 'diagnosis', '診療', 'medical']):
                return 'diagnosis_document'
            elif any(keyword in file_name for keyword in ['処方', 'prescription', '薬剤', 'medication']):
                return 'prescription_document'
            elif any(keyword in file_name for keyword in ['検査', 'test', 'lab', 'laboratory']):
                return 'test_document'
            elif any(keyword in file_name for keyword in ['手術', 'surgery', 'operation']):
                return 'surgery_document'
            elif any(keyword in file_name for keyword in ['看護', 'nursing', 'care']):
                return 'nursing_document'
            else:
                return 'general_medical'
        except Exception as e:
            logger.error(f"医療文書分類エラー: {e}")
            return 'general_medical'
    
    def _classify_education_document(self, file_path: str) -> str:
        """教育資料の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['教科書', 'textbook', '教材', 'teaching']):
                return 'textbook_document'
            elif any(keyword in file_name for keyword in ['試験', 'exam', 'test', 'quiz']):
                return 'exam_document'
            elif any(keyword in file_name for keyword in ['課題', 'assignment', 'homework']):
                return 'assignment_document'
            elif any(keyword in file_name for keyword in ['講義', 'lecture', 'presentation']):
                return 'lecture_document'
            elif any(keyword in file_name for keyword in ['研究', 'research', '論文', 'thesis']):
                return 'research_document'
            else:
                return 'general_education'
        except Exception as e:
            logger.error(f"教育資料分類エラー: {e}")
            return 'general_education'
    
    def _classify_research_document(self, file_path: str) -> str:
        """研究論文の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['論文', 'paper', 'article', 'journal']):
                return 'journal_article'
            elif any(keyword in file_name for keyword in ['学位論文', 'thesis', 'dissertation']):
                return 'thesis_document'
            elif any(keyword in file_name for keyword in ['研究報告', 'research_report', 'report']):
                return 'research_report'
            elif any(keyword in file_name for keyword in ['学会発表', 'conference', 'presentation']):
                return 'conference_paper'
            elif any(keyword in file_name for keyword in ['レビュー', 'review', 'survey']):
                return 'review_document'
            else:
                return 'general_research'
        except Exception as e:
            logger.error(f"研究論文分類エラー: {e}")
            return 'general_research'
    
    def _classify_patent_document(self, file_path: str) -> str:
        """特許文書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['特許', 'patent', 'invention']):
                return 'patent_application'
            elif any(keyword in file_name for keyword in ['実用新案', 'utility_model']):
                return 'utility_model'
            elif any(keyword in file_name for keyword in ['意匠', 'design', 'trademark']):
                return 'design_patent'
            elif any(keyword in file_name for keyword in ['商標', 'trademark', 'brand']):
                return 'trademark'
            else:
                return 'general_patent'
        except Exception as e:
            logger.error(f"特許文書分類エラー: {e}")
            return 'general_patent'
    
    def _classify_standard_document(self, file_path: str) -> str:
        """標準規格の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['iso', 'jis', 'iec', 'ansi']):
                return 'international_standard'
            elif any(keyword in file_name for keyword in ['規格', 'standard', 'specification']):
                return 'technical_standard'
            elif any(keyword in file_name for keyword in ['品質', 'quality', 'qms']):
                return 'quality_standard'
            elif any(keyword in file_name for keyword in ['安全', 'safety', 'security']):
                return 'safety_standard'
            else:
                return 'general_standard'
        except Exception as e:
            logger.error(f"標準規格分類エラー: {e}")
            return 'general_standard'
    
    def _classify_certification_document(self, file_path: str) -> str:
        """認証文書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['iso9001', 'iso14001', 'iso27001']):
                return 'iso_certification'
            elif any(keyword in file_name for keyword in ['資格', 'certification', 'license']):
                return 'professional_certification'
            elif any(keyword in file_name for keyword in ['免許', 'license', 'permit']):
                return 'license_document'
            elif any(keyword in file_name for keyword in ['認定', 'accreditation', 'approval']):
                return 'accreditation_document'
            else:
                return 'general_certification'
        except Exception as e:
            logger.error(f"認証文書分類エラー: {e}")
            return 'general_certification'
    
    def _classify_audit_document(self, file_path: str) -> str:
        """監査報告書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['監査', 'audit', 'auditing']):
                return 'audit_report'
            elif any(keyword in file_name for keyword in ['内部監査', 'internal_audit']):
                return 'internal_audit'
            elif any(keyword in file_name for keyword in ['外部監査', 'external_audit']):
                return 'external_audit'
            elif any(keyword in file_name for keyword in ['財務監査', 'financial_audit']):
                return 'financial_audit'
            else:
                return 'general_audit'
        except Exception as e:
            logger.error(f"監査報告書分類エラー: {e}")
            return 'general_audit'
    
    # 業界特化文書タイプの分類メソッド（5種類）
    def _classify_manufacturing_document(self, file_path: str) -> str:
        """製造業文書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['設計', 'design', 'cad', 'drawing']):
                return 'design_document'
            elif any(keyword in file_name for keyword in ['製造', 'manufacturing', 'production']):
                return 'production_document'
            elif any(keyword in file_name for keyword in ['品質管理', 'quality_control', 'qc']):
                return 'quality_control'
            elif any(keyword in file_name for keyword in ['設備', 'equipment', 'machine']):
                return 'equipment_document'
            elif any(keyword in file_name for keyword in ['安全', 'safety', 'hazard']):
                return 'safety_document'
            else:
                return 'general_manufacturing'
        except Exception as e:
            logger.error(f"製造業文書分類エラー: {e}")
            return 'general_manufacturing'
    
    def _classify_finance_document(self, file_path: str) -> str:
        """金融業文書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['決算', 'financial', 'annual_report']):
                return 'financial_report'
            elif any(keyword in file_name for keyword in ['投資', 'investment', 'portfolio']):
                return 'investment_document'
            elif any(keyword in file_name for keyword in ['融資', 'loan', 'credit']):
                return 'loan_document'
            elif any(keyword in file_name for keyword in ['保険', 'insurance', 'policy']):
                return 'insurance_document'
            elif any(keyword in file_name for keyword in ['リスク', 'risk', 'compliance']):
                return 'risk_document'
            else:
                return 'general_finance'
        except Exception as e:
            logger.error(f"金融業文書分類エラー: {e}")
            return 'general_finance'
    
    def _classify_healthcare_document(self, file_path: str) -> str:
        """医療業文書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['診療', 'clinical', 'patient']):
                return 'clinical_document'
            elif any(keyword in file_name for keyword in ['薬事', 'pharmaceutical', 'drug']):
                return 'pharmaceutical_document'
            elif any(keyword in file_name for keyword in ['医療機器', 'medical_device']):
                return 'medical_device_document'
            elif any(keyword in file_name for keyword in ['臨床試験', 'clinical_trial']):
                return 'clinical_trial_document'
            else:
                return 'general_healthcare'
        except Exception as e:
            logger.error(f"医療業文書分類エラー: {e}")
            return 'general_healthcare'
    
    def _classify_education_industry_document(self, file_path: str) -> str:
        """教育業文書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['カリキュラム', 'curriculum', 'syllabus']):
                return 'curriculum_document'
            elif any(keyword in file_name for keyword in ['評価', 'assessment', 'evaluation']):
                return 'assessment_document'
            elif any(keyword in file_name for keyword in ['研修', 'training', 'development']):
                return 'training_document'
            elif any(keyword in file_name for keyword in ['教育方針', 'education_policy']):
                return 'education_policy'
            else:
                return 'general_education_industry'
        except Exception as e:
            logger.error(f"教育業文書分類エラー: {e}")
            return 'general_education_industry'
    
    def _classify_government_document(self, file_path: str) -> str:
        """政府文書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            
            if any(keyword in file_name for keyword in ['法律', 'law', 'legal', 'legislation']):
                return 'legal_document'
            elif any(keyword in file_name for keyword in ['政策', 'policy', 'regulation']):
                return 'policy_document'
            elif any(keyword in file_name for keyword in ['予算', 'budget', 'finance']):
                return 'budget_document'
            elif any(keyword in file_name for keyword in ['統計', 'statistics', 'data']):
                return 'statistics_document'
            else:
                return 'general_government'
        except Exception as e:
            logger.error(f"政府文書分類エラー: {e}")
            return 'general_government'
    
    def _classify_code_document(self, file_path: str) -> str:
        """コード文書の詳細分類"""
        try:
            file_name = Path(file_path).name.lower()
            extension = Path(file_path).suffix.lower()
            
            # 拡張子ベースの分類
            if extension in ['.py']:
                return 'python_code'
            elif extension in ['.js', '.ts']:
                return 'javascript_typescript'
            elif extension in ['.java']:
                return 'java_code'
            elif extension in ['.cpp', '.c', '.h']:
                return 'c_cpp_code'
            elif extension in ['.cs']:
                return 'csharp_code'
            elif extension in ['.php']:
                return 'php_code'
            elif extension in ['.rb']:
                return 'ruby_code'
            elif extension in ['.go']:
                return 'go_code'
            elif extension in ['.rs']:
                return 'rust_code'
            elif extension in ['.swift']:
                return 'swift_code'
            elif extension in ['.kt']:
                return 'kotlin_code'
            elif extension in ['.scala']:
                return 'scala_code'
            elif extension in ['.r']:
                return 'r_code'
            elif extension in ['.sql']:
                return 'sql_code'
            elif extension in ['.html', '.css']:
                return 'web_code'
            elif extension in ['.xml', '.json', '.yaml', '.yml']:
                return 'config_code'
            elif extension in ['.md']:
                return 'markdown_document'
            else:
                return 'general_code'
        except Exception as e:
            logger.error(f"コード文書分類エラー: {e}")
            return 'general_code'


def classify_document(file_path: str) -> Dict[str, Any]:
    """
    文書を分類する便利関数
    
    Args:
        file_path (str): ファイルパス
        
    Returns:
        Dict[str, Any]: 分類結果
    """
    classifier = DocumentClassifier()
    return classifier.classify_document(file_path)
