"""
アップロード検証・エラーチェック機能
Phase 2の最優先実装項目

機能:
- ファイル形式・サイズの事前検証
- 処理可能性の事前判定
- 詳細なエラーメッセージの生成
- ユーザーフレンドリーなエラー表示
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# ログ設定
logger = logging.getLogger(__name__)


class FileValidationStatus(Enum):
    """ファイル検証ステータス"""
    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    INVALID_SIZE = "invalid_size"
    INVALID_PATH = "invalid_path"
    PROCESSING_ERROR = "processing_error"


class ProcessingCapability(Enum):
    """処理可能性レベル"""
    FULL_SUPPORT = "full_support"  # 完全対応
    PARTIAL_SUPPORT = "partial_support"  # 部分対応
    BASIC_SUPPORT = "basic_support"  # 基本対応
    NO_SUPPORT = "no_support"  # 非対応


@dataclass
class UploadValidationResult:
    """アップロード検証結果"""
    is_valid: bool
    file_size: int
    file_format: str
    processing_capability: ProcessingCapability
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    estimated_processing_time: float = 0.0
    supported_features: List[str] = field(default_factory=list)


class UploadValidator:
    """アップロード検証・エラーチェッククラス"""
    
    def __init__(self, max_file_size: int = 50 * 1024 * 1024):  # 50MB
        """
        初期化
        
        Args:
            max_file_size: 最大ファイルサイズ（バイト）
        """
        self.max_file_size = max_file_size
        self.supported_formats = {
            # 完全対応フォーマット
            '.txt': ProcessingCapability.FULL_SUPPORT,
            '.md': ProcessingCapability.FULL_SUPPORT,
            '.xlsx': ProcessingCapability.FULL_SUPPORT,
            '.xls': ProcessingCapability.FULL_SUPPORT,
            '.xlsm': ProcessingCapability.FULL_SUPPORT,
            '.xlsb': ProcessingCapability.FULL_SUPPORT,
            
            # 部分対応フォーマット（Phase 2で実装予定）
            '.docx': ProcessingCapability.PARTIAL_SUPPORT,
            '.pdf': ProcessingCapability.PARTIAL_SUPPORT,
            '.pptx': ProcessingCapability.PARTIAL_SUPPORT,
            
            # 基本対応フォーマット
            '.html': ProcessingCapability.BASIC_SUPPORT,
            '.xml': ProcessingCapability.BASIC_SUPPORT,
            '.csv': ProcessingCapability.BASIC_SUPPORT,
            
            # 画像フォーマット（Phase 2で実装予定）
            '.jpg': ProcessingCapability.PARTIAL_SUPPORT,
            '.jpeg': ProcessingCapability.PARTIAL_SUPPORT,
            '.png': ProcessingCapability.PARTIAL_SUPPORT,
            '.gif': ProcessingCapability.PARTIAL_SUPPORT,
            '.bmp': ProcessingCapability.PARTIAL_SUPPORT,
            '.tiff': ProcessingCapability.PARTIAL_SUPPORT,
        }
        
        # 処理時間の推定（秒）
        self.processing_time_estimates = {
            ProcessingCapability.FULL_SUPPORT: 1.0,
            ProcessingCapability.PARTIAL_SUPPORT: 3.0,
            ProcessingCapability.BASIC_SUPPORT: 2.0,
            ProcessingCapability.NO_SUPPORT: 0.0,
        }
        
        # サポート機能の定義
        self.supported_features_map = {
            ProcessingCapability.FULL_SUPPORT: [
                "テキスト抽出", "メタデータ生成", "チャンク分割", 
                "検索最適化", "品質評価"
            ],
            ProcessingCapability.PARTIAL_SUPPORT: [
                "テキスト抽出", "メタデータ生成", "基本チャンク分割"
            ],
            ProcessingCapability.BASIC_SUPPORT: [
                "テキスト抽出", "基本メタデータ生成"
            ],
            ProcessingCapability.NO_SUPPORT: []
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> UploadValidationResult:
        """
        ファイルの事前検証
        
        Args:
            file_path: ファイルパス
            
        Returns:
            UploadValidationResult: 検証結果
        """
        try:
            file_path = Path(file_path)
            
            # ファイル存在確認
            if not file_path.exists():
                return self._create_error_result(
                    FileValidationStatus.INVALID_PATH,
                    f"ファイルが見つかりません: {file_path.name}",
                    ["ファイルパスを確認してください", "ファイルが移動・削除されていないか確認してください"]
                )
            
            # ファイルサイズ確認
            file_size = file_path.stat().st_size
            if file_size == 0:
                return self._create_error_result(
                    FileValidationStatus.INVALID_SIZE,
                    "ファイルが空です",
                    ["ファイルが正しく保存されているか確認してください"]
                )
            
            if file_size > self.max_file_size:
                return self._create_error_result(
                    FileValidationStatus.INVALID_SIZE,
                    f"ファイルサイズが大きすぎます ({self._format_file_size(file_size)})",
                    [f"50MB以下のファイルをアップロードしてください", "ファイルを分割してアップロードしてください"]
                )
            
            # ファイル形式確認
            file_ext = file_path.suffix.lower()
            if file_ext not in self.supported_formats:
                return self._create_error_result(
                    FileValidationStatus.INVALID_FORMAT,
                    f"サポートされていないファイル形式です: {file_ext}",
                    [f"サポートされている形式: {', '.join(self.supported_formats.keys())}", 
                     "ファイルをサポートされている形式に変換してください"]
                )
            
            # 処理可能性の判定
            processing_capability = self.supported_formats[file_ext]
            estimated_time = self.processing_time_estimates[processing_capability]
            supported_features = self.supported_features_map[processing_capability]
            
            # 警告の生成
            warnings = self._generate_warnings(file_path, file_size, processing_capability)
            
            # 推奨事項の生成
            recommendations = self._generate_recommendations(file_path, processing_capability)
            
            return UploadValidationResult(
                is_valid=True,
                file_size=file_size,
                file_format=file_ext,
                processing_capability=processing_capability,
                warnings=warnings,
                errors=[],
                recommendations=recommendations,
                estimated_processing_time=estimated_time,
                supported_features=supported_features
            )
            
        except Exception as e:
            logger.error(f"ファイル検証エラー: {e}")
            return self._create_error_result(
                FileValidationStatus.PROCESSING_ERROR,
                f"検証中にエラーが発生しました: {str(e)}",
                ["ファイルが破損していないか確認してください", "管理者にお問い合わせください"]
            )
    
    def check_processing_capability(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        処理可能性の詳細判定
        
        Args:
            file_path: ファイルパス
            
        Returns:
            Dict[str, Any]: 処理可能性の詳細情報
        """
        validation_result = self.validate_file(file_path)
        
        if not validation_result.is_valid:
            return {
                "can_process": False,
                "reason": validation_result.errors[0] if validation_result.errors else "不明なエラー",
                "capability_level": ProcessingCapability.NO_SUPPORT.value,
                "supported_features": [],
                "estimated_time": 0.0
            }
        
        return {
            "can_process": True,
            "capability_level": validation_result.processing_capability.value,
            "supported_features": validation_result.supported_features,
            "estimated_time": validation_result.estimated_processing_time,
            "warnings": validation_result.warnings,
            "recommendations": validation_result.recommendations
        }
    
    def generate_error_message(self, error_type: str, details: Dict[str, Any]) -> str:
        """
        詳細なエラーメッセージの生成
        
        Args:
            error_type: エラータイプ
            details: エラー詳細
            
        Returns:
            str: ユーザーフレンドリーなエラーメッセージ
        """
        error_messages = {
            "file_not_found": f"❌ ファイルが見つかりません\nファイル名: {details.get('filename', '不明')}\n\n💡 解決方法:\n• ファイルパスを確認してください\n• ファイルが移動・削除されていないか確認してください",
            
            "file_too_large": f"❌ ファイルサイズが大きすぎます\nファイルサイズ: {details.get('file_size', '不明')}\n最大サイズ: 50MB\n\n💡 解決方法:\n• ファイルを50MB以下に圧縮してください\n• ファイルを分割してアップロードしてください",
            
            "unsupported_format": f"❌ サポートされていないファイル形式です\nファイル形式: {details.get('format', '不明')}\n\n💡 解決方法:\n• サポートされている形式に変換してください\n• サポート形式: {', '.join(self.supported_formats.keys())}",
            
            "processing_error": f"❌ ファイル処理中にエラーが発生しました\nエラー: {details.get('error', '不明')}\n\n💡 解決方法:\n• ファイルが破損していないか確認してください\n• 管理者にお問い合わせください",
            
            "empty_file": "❌ ファイルが空です\n\n💡 解決方法:\n• ファイルが正しく保存されているか確認してください\n• ファイルに内容があるか確認してください"
        }
        
        return error_messages.get(error_type, f"❌ 不明なエラーが発生しました: {error_type}")
    
    def _create_error_result(self, status: FileValidationStatus, error_msg: str, recommendations: List[str]) -> UploadValidationResult:
        """エラー結果の作成"""
        return UploadValidationResult(
            is_valid=False,
            file_size=0,
            file_format="",
            processing_capability=ProcessingCapability.NO_SUPPORT,
            warnings=[],
            errors=[error_msg],
            recommendations=recommendations,
            estimated_processing_time=0.0,
            supported_features=[]
        )
    
    def _format_file_size(self, size_bytes: int) -> str:
        """ファイルサイズのフォーマット"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def _generate_warnings(self, file_path: Path, file_size: int, capability: ProcessingCapability) -> List[str]:
        """警告メッセージの生成"""
        warnings = []
        
        # ファイルサイズ警告
        if file_size > 10 * 1024 * 1024:  # 10MB以上
            warnings.append(f"ファイルサイズが大きいです ({self._format_file_size(file_size)})。処理に時間がかかる可能性があります。")
        
        # 処理能力警告
        if capability == ProcessingCapability.PARTIAL_SUPPORT:
            warnings.append("このファイル形式は部分対応です。一部の機能が制限される可能性があります。")
        elif capability == ProcessingCapability.BASIC_SUPPORT:
            warnings.append("このファイル形式は基本対応です。高度な機能は利用できません。")
        
        # 特殊な警告
        if file_path.suffix.lower() in ['.docx', '.pdf']:
            warnings.append("画像が含まれている場合、OCR処理により処理時間が長くなる可能性があります。")
        
        return warnings
    
    def _generate_recommendations(self, file_path: Path, capability: ProcessingCapability) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        # 処理能力に応じた推奨事項
        if capability == ProcessingCapability.PARTIAL_SUPPORT:
            recommendations.append("より良い結果を得るために、ファイルをテキスト形式に変換することを検討してください。")
        elif capability == ProcessingCapability.BASIC_SUPPORT:
            recommendations.append("高度な機能を利用するために、サポートされている形式に変換してください。")
        
        # ファイル形式別の推奨事項
        if file_path.suffix.lower() in ['.docx', '.pdf']:
            recommendations.append("画像が含まれている場合は、OCR処理の精度を向上させるため、高解像度で保存してください。")
        
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            recommendations.append("複数シートがある場合は、各シートが適切に処理されます。")
        
        return recommendations


def validate_upload_files(file_paths: List[Union[str, Path]]) -> Dict[str, UploadValidationResult]:
    """
    複数ファイルの一括検証
    
    Args:
        file_paths: ファイルパスのリスト
        
    Returns:
        Dict[str, UploadValidationResult]: ファイルパスをキーとした検証結果の辞書
    """
    validator = UploadValidator()
    results = {}
    
    for file_path in file_paths:
        file_path = Path(file_path)
        results[str(file_path)] = validator.validate_file(file_path)
    
    return results


# 使用例とテスト用の関数
def test_upload_validation():
    """アップロード検証のテスト関数"""
    validator = UploadValidator()
    
    # テスト用のファイルパス（実際のファイルが存在しない場合のテスト）
    test_files = [
        "test.txt",
        "test.docx", 
        "test.pdf",
        "test.xlsx",
        "nonexistent.txt",
        "test.unsupported"
    ]
    
    print("=== アップロード検証テスト ===")
    for file_path in test_files:
        result = validator.validate_file(file_path)
        print(f"\nファイル: {file_path}")
        print(f"有効: {result.is_valid}")
        if result.errors:
            print(f"エラー: {result.errors}")
        if result.warnings:
            print(f"警告: {result.warnings}")
        if result.recommendations:
            print(f"推奨: {result.recommendations}")


if __name__ == "__main__":
    test_upload_validation()
