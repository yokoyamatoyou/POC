"""
処理方法選択器
Phase 2の画像混じり文書処理の最適な処理方法を自動選択

機能:
- 最適な処理方法の選択
- 文書のルーティング処理
- OCR処理方法の自動選択
- 処理パフォーマンスの最適化
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from .multimodal_detector import MultimodalDetector, MultimodalDetectionResult, ProcessingComplexity, ImageType

# ログ設定
logger = logging.getLogger(__name__)


class ProcessingMethod(Enum):
    """処理方法"""
    TEXT_ONLY = "text_only"  # テキストのみ処理
    MULTIMODAL_BASIC = "multimodal_basic"  # 基本マルチモーダル処理
    MULTIMODAL_ADVANCED = "multimodal_advanced"  # 高度マルチモーダル処理
    MULTIMODAL_HEAVY = "multimodal_heavy"  # 重いマルチモーダル処理
    IMAGE_HEAVY = "image_heavy"  # 画像中心処理
    OCR_FOCUSED = "ocr_focused"  # OCR中心処理


class OCRMethod(Enum):
    """OCR処理方法"""
    TESSERACT = "tesseract"  # Tesseract OCR
    EASYOCR = "easyocr"  # EasyOCR
    HYBRID = "hybrid"  # ハイブリッド（両方使用）
    NONE = "none"  # OCR不要


@dataclass
class ProcessingRoute:
    """処理ルート情報"""
    file_path: str
    detected_type: str
    selected_method: ProcessingMethod
    ocr_method: OCRMethod
    estimated_time: float
    confidence: float
    fallback_methods: List[ProcessingMethod]
    processing_parameters: Dict[str, Any] = field(default_factory=dict)


class ProcessingRouter:
    """処理方法選択器"""
    
    def __init__(self):
        """初期化"""
        self.detector = MultimodalDetector()
        self.ocr_router = OCRRouter()
        
        # 処理方法の優先度設定
        self.method_priorities = {
            ProcessingMethod.TEXT_ONLY: 1,
            ProcessingMethod.MULTIMODAL_BASIC: 2,
            ProcessingMethod.MULTIMODAL_ADVANCED: 3,
            ProcessingMethod.MULTIMODAL_HEAVY: 4,
            ProcessingMethod.IMAGE_HEAVY: 5,
            ProcessingMethod.OCR_FOCUSED: 6
        }
        
        # 文書タイプ別の推奨処理方法
        self.document_type_methods = {
            '.txt': ProcessingMethod.TEXT_ONLY,
            '.md': ProcessingMethod.TEXT_ONLY,
            '.xlsx': ProcessingMethod.MULTIMODAL_BASIC,
            '.xls': ProcessingMethod.MULTIMODAL_BASIC,
            '.docx': ProcessingMethod.MULTIMODAL_ADVANCED,
            '.pdf': ProcessingMethod.MULTIMODAL_ADVANCED,
            '.pptx': ProcessingMethod.MULTIMODAL_BASIC,
            '.jpg': ProcessingMethod.IMAGE_HEAVY,
            '.jpeg': ProcessingMethod.IMAGE_HEAVY,
            '.png': ProcessingMethod.IMAGE_HEAVY,
            '.gif': ProcessingMethod.IMAGE_HEAVY,
            '.bmp': ProcessingMethod.IMAGE_HEAVY,
            '.tiff': ProcessingMethod.IMAGE_HEAVY
        }
    
    def select_processing_method(self, document_info: Union[MultimodalDetectionResult, Dict[str, Any]]) -> ProcessingMethod:
        """
        最適な処理方法の選択
        
        Args:
            document_info: 文書情報（MultimodalDetectionResultまたは辞書）
            
        Returns:
            ProcessingMethod: 選択された処理方法
        """
        try:
            # 入力検証
            if document_info is None:
                logger.warning("文書情報がNoneです。デフォルトの処理方法を返します。")
                return ProcessingMethod.TEXT_ONLY
            
            # 文書情報の正規化
            if isinstance(document_info, MultimodalDetectionResult):
                info = document_info
            else:
                # 辞書からMultimodalDetectionResultを作成
                info = self._create_detection_result_from_dict(document_info)
            
            # ファイルパスの検証
            if not info.file_path:
                logger.warning("ファイルパスが空です。デフォルトの処理方法を返します。")
                return ProcessingMethod.TEXT_ONLY
            
            # ファイル拡張子による基本判定
            file_path = Path(info.file_path)
            file_ext = file_path.suffix.lower()
            
            # 画像がない場合はテキストのみ処理（ただし画像拡張子は例外）
            if not info.has_images:
                # 画像拡張子の場合は強制的に画像処理
                if file_ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}:
                    logger.warning(f"画像拡張子({file_ext})だがhas_images=False。強制的にIMAGE_HEAVYに設定")
                    return ProcessingMethod.IMAGE_HEAVY
                return ProcessingMethod.TEXT_ONLY
            
            # 画像タイプによる処理方法の選択
            method = self._select_method_by_image_types(info.image_types, file_ext)
            
            # 処理複雑度による調整
            method = self._adjust_method_by_complexity(method, info.processing_complexity)
            
            # 画像混じり度による最終調整
            method = self._adjust_method_by_image_ratio(method, info.image_ratio)
            
            logger.info(f"処理方法選択: {file_path.name} -> {method.value}")
            return method
            
        except AttributeError as e:
            logger.error(f"文書情報の属性エラー: {e}")
            return ProcessingMethod.TEXT_ONLY
        except KeyError as e:
            logger.error(f"文書情報のキーエラー: {e}")
            return ProcessingMethod.TEXT_ONLY
        except Exception as e:
            logger.error(f"処理方法選択エラー: {e}")
            return ProcessingMethod.TEXT_ONLY
    
    def route_document(self, file_path: Union[str, Path]) -> ProcessingRoute:
        """
        文書のルーティング処理
        
        Args:
            file_path: ファイルパス
            
        Returns:
            ProcessingRoute: 処理ルート情報
        """
        try:
            file_path = Path(file_path)

            # ファイル存在チェックと空ファイルチェック
            if not file_path.exists() or not file_path.is_file():
                logger.warning(f"route_document: ファイル不在または不正: {file_path}")
                return self._create_fallback_route(file_path)

            try:
                size_bytes = file_path.stat().st_size
            except Exception:
                size_bytes = 0

            if size_bytes == 0:
                logger.warning(f"route_document: 空ファイルを検出: {file_path}")
                return self._create_fallback_route(file_path)

            # マルチモーダル検出
            detection_result = self.detector.detect_multimodal_content(file_path)
            
            # 処理方法の選択
            selected_method = self.select_processing_method(detection_result)
            
            # OCR処理方法の選択
            ocr_method = self.ocr_router.select_ocr_method(
                detection_result.images if detection_result.images else [],
                selected_method
            )
            
            # フォールバック方法の決定
            fallback_methods = self._get_fallback_methods(selected_method)
            
            # 処理パラメータの設定
            processing_parameters = self._get_processing_parameters(
                selected_method, ocr_method, detection_result
            )
            
            # 大容量ファイルの場合は処理パラメータにフラグを立てる
            if size_bytes >= int(os.environ.get("LARGE_FILE_THRESHOLD_BYTES", 50 * 1024 * 1024)):
                processing_parameters["large_file"] = True

            # チャンク生成元を明示（互換ヘルパー向け）
            processing_parameters.setdefault("chunk_source", "processing_router")
            processing_parameters.setdefault("file_size", size_bytes)

            # 信頼度の計算
            confidence = self._calculate_confidence(detection_result, selected_method)
            
            return ProcessingRoute(
                file_path=str(file_path),
                detected_type=file_path.suffix.lower(),
                selected_method=selected_method,
                ocr_method=ocr_method,
                estimated_time=detection_result.estimated_processing_time,
                confidence=confidence,
                fallback_methods=fallback_methods,
                processing_parameters=processing_parameters
            )
            
        except FileNotFoundError as e:
            logger.error(f"ファイルが見つかりません: {file_path}, エラー: {e}")
            return self._create_fallback_route(file_path)
        except PermissionError as e:
            logger.error(f"ファイルアクセス権限エラー: {file_path}, エラー: {e}")
            return self._create_fallback_route(file_path)
        except Exception as e:
            logger.error(f"文書ルーティングエラー: {file_path}, エラー: {e}")
            return self._create_fallback_route(file_path)
    
    def _select_method_by_image_types(self, image_types: List[ImageType], file_ext: str) -> ProcessingMethod:
        """画像タイプによる処理方法の選択"""
        if not image_types:
            return ProcessingMethod.TEXT_ONLY
        
        # 画像タイプの分析
        type_counts: Dict[ImageType, int] = {}
        for img_type in image_types:
            type_counts[img_type] = type_counts.get(img_type, 0) + 1

        # 主要な画像タイプを特定
        if type_counts:
            dominant_type = max(type_counts, key=lambda k: type_counts[k])
        else:
            dominant_type = ImageType.UNKNOWN
        
        # 画像タイプ別の処理方法選択
        if dominant_type == ImageType.SCREENSHOT:
            return ProcessingMethod.OCR_FOCUSED
        elif dominant_type in [ImageType.CHART, ImageType.DIAGRAM]:
            return ProcessingMethod.MULTIMODAL_ADVANCED
        elif dominant_type == ImageType.PHOTO:
            return ProcessingMethod.IMAGE_HEAVY
        elif dominant_type == ImageType.DRAWING:
            return ProcessingMethod.MULTIMODAL_HEAVY
        else:
            return ProcessingMethod.MULTIMODAL_BASIC
    
    def _adjust_method_by_complexity(self, method: ProcessingMethod, complexity: ProcessingComplexity) -> ProcessingMethod:
        """処理複雑度による処理方法の調整"""
        if complexity == ProcessingComplexity.SIMPLE:
            # 画像系のメソッドは降格させない（SIMPLEでも元のmethodを維持）
            if method in [ProcessingMethod.IMAGE_HEAVY, ProcessingMethod.MULTIMODAL_HEAVY, 
                          ProcessingMethod.MULTIMODAL_ADVANCED, ProcessingMethod.MULTIMODAL_BASIC,
                          ProcessingMethod.OCR_FOCUSED]:
                return method
            # TEXT_ONLYのみ降格対象
            return ProcessingMethod.TEXT_ONLY
        elif complexity == ProcessingComplexity.MODERATE:
            if method in [ProcessingMethod.MULTIMODAL_HEAVY, ProcessingMethod.IMAGE_HEAVY]:
                return ProcessingMethod.MULTIMODAL_BASIC
        elif complexity == ProcessingComplexity.COMPLEX:
            if method == ProcessingMethod.TEXT_ONLY:
                return ProcessingMethod.MULTIMODAL_BASIC
        elif complexity == ProcessingComplexity.VERY_COMPLEX:
            if method in [ProcessingMethod.TEXT_ONLY, ProcessingMethod.MULTIMODAL_BASIC]:
                return ProcessingMethod.MULTIMODAL_ADVANCED
        
        return method
    
    def _adjust_method_by_image_ratio(self, method: ProcessingMethod, image_ratio: float) -> ProcessingMethod:
        """画像混じり度による処理方法の調整"""
        if image_ratio < 0.1:  # 画像が少ない
            if method in [ProcessingMethod.MULTIMODAL_HEAVY, ProcessingMethod.IMAGE_HEAVY]:
                return ProcessingMethod.MULTIMODAL_BASIC
        elif image_ratio > 0.5:  # 画像が多い
            if method == ProcessingMethod.MULTIMODAL_BASIC:
                return ProcessingMethod.MULTIMODAL_ADVANCED
            elif method == ProcessingMethod.MULTIMODAL_ADVANCED:
                return ProcessingMethod.MULTIMODAL_HEAVY
        elif image_ratio > 0.8:  # 画像が非常に多い
            return ProcessingMethod.IMAGE_HEAVY
        
        return method
    
    def _get_fallback_methods(self, primary_method: ProcessingMethod) -> List[ProcessingMethod]:
        """フォールバック方法の取得"""
        fallback_map = {
            ProcessingMethod.TEXT_ONLY: [ProcessingMethod.MULTIMODAL_BASIC],
            ProcessingMethod.MULTIMODAL_BASIC: [ProcessingMethod.TEXT_ONLY, ProcessingMethod.MULTIMODAL_ADVANCED],
            ProcessingMethod.MULTIMODAL_ADVANCED: [ProcessingMethod.MULTIMODAL_BASIC, ProcessingMethod.MULTIMODAL_HEAVY],
            ProcessingMethod.MULTIMODAL_HEAVY: [ProcessingMethod.MULTIMODAL_ADVANCED, ProcessingMethod.IMAGE_HEAVY],
            ProcessingMethod.IMAGE_HEAVY: [ProcessingMethod.MULTIMODAL_HEAVY, ProcessingMethod.OCR_FOCUSED],
            ProcessingMethod.OCR_FOCUSED: [ProcessingMethod.IMAGE_HEAVY, ProcessingMethod.MULTIMODAL_ADVANCED]
        }
        
        return fallback_map.get(primary_method, [ProcessingMethod.TEXT_ONLY])
    
    def _get_processing_parameters(self, method: ProcessingMethod, ocr_method: OCRMethod, detection_result: MultimodalDetectionResult) -> Dict[str, Any]:
        """処理パラメータの取得"""
        parameters = {
            "method": method.value,
            "ocr_method": ocr_method.value,
            "image_count": detection_result.image_count,
            "image_ratio": detection_result.image_ratio,
            "complexity": detection_result.processing_complexity.value
        }
        
        # 処理方法別のパラメータ
        if method == ProcessingMethod.TEXT_ONLY:
            parameters.update({
                "enable_ocr": False,
                "enable_image_analysis": False,
                "chunk_size": 500,
                "overlap": 50
            })
        elif method == ProcessingMethod.MULTIMODAL_BASIC:
            parameters.update({
                "enable_ocr": True,
                "enable_image_analysis": True,
                "chunk_size": 400,
                "overlap": 40,
                "ocr_confidence_threshold": 0.7
            })
        elif method == ProcessingMethod.MULTIMODAL_ADVANCED:
            parameters.update({
                "enable_ocr": True,
                "enable_image_analysis": True,
                "chunk_size": 300,
                "overlap": 30,
                "ocr_confidence_threshold": 0.8,
                "enable_image_classification": True
            })
        elif method == ProcessingMethod.MULTIMODAL_HEAVY:
            parameters.update({
                "enable_ocr": True,
                "enable_image_analysis": True,
                "chunk_size": 200,
                "overlap": 20,
                "ocr_confidence_threshold": 0.9,
                "enable_image_classification": True,
                "enable_advanced_ocr": True
            })
        elif method == ProcessingMethod.IMAGE_HEAVY:
            parameters.update({
                "enable_ocr": True,
                "enable_image_analysis": True,
                "chunk_size": 100,
                "overlap": 10,
                "ocr_confidence_threshold": 0.95,
                "enable_image_classification": True,
                "enable_advanced_ocr": True,
                "prioritize_images": True
            })
        elif method == ProcessingMethod.OCR_FOCUSED:
            parameters.update({
                "enable_ocr": True,
                "enable_image_analysis": False,
                "chunk_size": 150,
                "overlap": 15,
                "ocr_confidence_threshold": 0.9,
                "enable_advanced_ocr": True,
                "ocr_only": True
            })
        
        return parameters
    
    def _calculate_confidence(self, detection_result: MultimodalDetectionResult, method: ProcessingMethod) -> float:
        """信頼度の計算"""
        base_confidence = 0.8
        
        # 画像数による調整
        if detection_result.image_count == 0:
            if method == ProcessingMethod.TEXT_ONLY:
                base_confidence += 0.2
            else:
                base_confidence -= 0.3
        elif detection_result.image_count <= 3:
            if method in [ProcessingMethod.MULTIMODAL_BASIC, ProcessingMethod.MULTIMODAL_ADVANCED]:
                base_confidence += 0.1
        elif detection_result.image_count > 10:
            if method in [ProcessingMethod.MULTIMODAL_HEAVY, ProcessingMethod.IMAGE_HEAVY]:
                base_confidence += 0.1
            else:
                base_confidence -= 0.2
        
        # 画像混じり度による調整
        if detection_result.image_ratio < 0.1:
            if method == ProcessingMethod.TEXT_ONLY:
                base_confidence += 0.1
        elif detection_result.image_ratio > 0.5:
            if method in [ProcessingMethod.MULTIMODAL_ADVANCED, ProcessingMethod.MULTIMODAL_HEAVY, ProcessingMethod.IMAGE_HEAVY]:
                base_confidence += 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _create_detection_result_from_dict(self, info_dict: Dict[str, Any]) -> MultimodalDetectionResult:
        """辞書からMultimodalDetectionResultを作成"""
        return MultimodalDetectionResult(
            file_path=info_dict.get("file_path", ""),
            has_images=info_dict.get("has_images", False),
            image_count=info_dict.get("image_count", 0),
            image_ratio=info_dict.get("image_ratio", 0.0),
            image_types=info_dict.get("image_types", []),
            processing_complexity=ProcessingComplexity(info_dict.get("processing_complexity", "simple")),
            recommended_method=info_dict.get("recommended_method", "text_only"),
            estimated_processing_time=info_dict.get("estimated_processing_time", 1.0)
        )
    
    def _create_fallback_route(self, file_path: Union[str, Path]) -> ProcessingRoute:
        """フォールバックルートの作成"""
        try:
            file_path = Path(file_path)
            file_ext = file_path.suffix.lower() if file_path.suffix else "unknown"
        except Exception:
            file_ext = "unknown"
        
        return ProcessingRoute(
            file_path=str(file_path),
            detected_type=file_ext,
            selected_method=ProcessingMethod.TEXT_ONLY,
            ocr_method=OCRMethod.NONE,
            estimated_time=1.0,
            confidence=0.3,  # フォールバックなので信頼度を下げる
            fallback_methods=[ProcessingMethod.MULTIMODAL_BASIC, ProcessingMethod.OCR_FOCUSED],
            processing_parameters={
                "method": "text_only",
                "ocr_method": "none",
                "enable_ocr": False,
                "enable_image_analysis": False,
                "fallback_mode": True,
                "error_recovery": True,
                "chunk_source": "processing_router_fallback",
                "chunk_count": 0,
            }
        )


class OCRRouter:
    """OCR処理方法選択器"""
    
    def __init__(self):
        """初期化"""
        self.tesseract_available = self._check_tesseract()
        self.easyocr_available = self._check_easyocr()
        
        # OCR方法の優先度
        self.ocr_priorities = {
            OCRMethod.TESSERACT: 1,
            OCRMethod.EASYOCR: 2,
            OCRMethod.HYBRID: 3,
            OCRMethod.NONE: 4
        }
    
    def select_ocr_method(self, images: List[Any], processing_method: ProcessingMethod) -> OCRMethod:
        """
        最適なOCR処理方法の選択
        
        Args:
            images: 画像データのリスト
            processing_method: 処理方法
            
        Returns:
            OCRMethod: 選択されたOCR方法
        """
        try:
            # 画像がない場合はOCR不要
            if not images:
                return OCRMethod.NONE
            
            # テキストのみ処理の場合はOCR不要
            if processing_method == ProcessingMethod.TEXT_ONLY:
                return OCRMethod.NONE
            
            # 利用可能なOCR方法の確認
            available_methods = []
            if self.tesseract_available:
                available_methods.append(OCRMethod.TESSERACT)
            if self.easyocr_available:
                available_methods.append(OCRMethod.EASYOCR)
            
            if not available_methods:
                return OCRMethod.NONE
            
            # 画像数による選択
            if len(images) <= 2:
                # 画像が少ない場合は単一OCR方法
                return available_methods[0]
            elif len(images) <= 5:
                # 画像が中程度の場合は利用可能な方法を選択
                if OCRMethod.TESSERACT in available_methods:
                    return OCRMethod.TESSERACT
                else:
                    return available_methods[0]
            else:
                # 画像が多い場合はハイブリッド
                if len(available_methods) >= 2:
                    return OCRMethod.HYBRID
                else:
                    return available_methods[0]
                    
        except Exception as e:
            logger.error(f"OCR方法選択エラー: {e}")
            return OCRMethod.NONE
    
    def get_ocr_confidence(self, image_data: bytes) -> float:
        """
        OCR処理の信頼度を事前評価
        
        Args:
            image_data: 画像データ
            
        Returns:
            float: 信頼度（0.0-1.0）
        """
        try:
            # 簡易的な信頼度評価
            # 実際の実装では、画像解析を行ってより詳細な評価を行う
            
            file_size = len(image_data)
            
            # ファイルサイズによる簡易評価
            if file_size < 10000:  # 10KB未満
                return 0.3
            elif file_size < 100000:  # 100KB未満
                return 0.6
            elif file_size < 1000000:  # 1MB未満
                return 0.8
            else:
                return 0.9
                
        except Exception as e:
            logger.warning(f"OCR信頼度評価エラー: {e}")
            return 0.5
    
    def _check_tesseract(self) -> bool:
        """Tesseract OCRの利用可能性確認"""
        try:
            import pytesseract
            # 簡単なテスト実行
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def _check_easyocr(self) -> bool:
        """EasyOCRの利用可能性確認"""
        try:
            import easyocr
            # 簡単なテスト実行
            easyocr.Reader(['en'])
            return True
        except Exception:
            return False


# 使用例とテスト用の関数
def test_processing_router():
    """処理ルーターのテスト関数"""
    router = ProcessingRouter()
    
    # テスト用のファイルパス
    test_files = [
        "test.txt",
        "test.docx",
        "test.pdf",
        "test.xlsx"
    ]
    
    print("=== 処理ルーターテスト ===")
    for file_path in test_files:
        route = router.route_document(file_path)
        print(f"\nファイル: {file_path}")
        print(f"処理方法: {route.selected_method.value}")
        print(f"OCR方法: {route.ocr_method.value}")
        print(f"推定時間: {route.estimated_time:.1f}秒")
        print(f"信頼度: {route.confidence:.2f}")
        print(f"フォールバック: {[m.value for m in route.fallback_methods]}")


if __name__ == "__main__":
    test_processing_router()
