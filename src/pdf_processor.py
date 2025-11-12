"""
PDF文書の統合処理クラス

Phase 3: PDF文字画像処理のメインクラス
テキスト抽出、画像抽出、メタデータ抽出を統合してPDF文書を処理する機能を提供
"""

import errno
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import os

# 内部モジュールのインポート
from .text_extractor import TextExtractor, TextBlock, TableData
from .image_extractor import ImageExtractor, ImageData, ProcessedImage
from .metadata_extractor import MetadataExtractor, PDFMetadata, PDFStructure, PDFStatistics

# ログ設定
logger = logging.getLogger(__name__)

@dataclass
class PDFProcessingResult:
    """PDF処理結果のデータクラス"""
    success: bool
    file_path: str
    processing_time: float
    text_blocks: List[TextBlock]
    tables: List[TableData]
    images: List[ImageData]
    processed_images: List[ProcessedImage]
    metadata: PDFMetadata
    structure: PDFStructure
    statistics: PDFStatistics
    quality_score: float
    error_message: Optional[str] = None

class PDFProcessor:
    """PDF文書の統合処理クラス"""
    
    def __init__(self, image_output_dir: str = "temp_images"):
        """
        PDF処理器の初期化
        
        Args:
            image_output_dir (str): 画像出力ディレクトリ
        """
        self.logger = logging.getLogger(__name__)
        
        # 各抽出器を初期化
        self.text_extractor = TextExtractor()
        self.image_extractor = ImageExtractor(image_output_dir)
        self.metadata_extractor = MetadataExtractor()
        
        # 処理統計
        self.processing_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "total_processing_time": 0.0
        }

    def _normalize_input_path(self, file_path: Any) -> str:
        """ファイルパスを文字列として正規化し、空文字列を排除する"""
        if file_path is None:
            raise FileNotFoundError("ファイルが見つかりません: None")

        try:
            normalized = os.fspath(file_path)
        except TypeError:
            normalized = str(file_path)

        normalized_str = str(normalized)
        if not normalized_str.strip() or normalized_str.strip().lower() == "none":
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

        return normalized_str

    def _safe_display_path(self, file_path: Any) -> str:
        """ログ出力用に安全な文字列表現を返す"""
        try:
            return self._normalize_input_path(file_path)
        except Exception:
            return str(file_path)

    def _ensure_fitz_can_open(self, file_path: str) -> None:
        """fitz.open が成功するかを確認し、失敗時は詳細に分類する"""
        try:
            import fitz
            doc_test = fitz.open(file_path)
            try:
                doc_test.close()
            except Exception:
                pass
        except Exception as error:
            self._handle_fitz_open_error(file_path, error)

    def _handle_fitz_open_error(self, file_path: str, error: Exception) -> None:
        message = str(error).strip() or repr(error)
        lowered = message.lower()

        if self._is_missing_file_error(error, lowered):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}") from error

        detailed_message = f"fitz.open に失敗しました ({message})"
        self.logger.error(detailed_message, exc_info=True)
        raise RuntimeError(detailed_message) from error

    @staticmethod
    def _is_missing_file_error(error: Exception, message_lower: str) -> bool:
        if isinstance(error, FileNotFoundError):
            return True

        errno_value = getattr(error, "errno", None)
        if isinstance(error, OSError) and errno_value == errno.ENOENT:
            return True

        missing_tokens = (
            "no such file",
            "file not found",
            "cannot find",
            "does not exist",
            "no file or directory",
        )

        return any(token in message_lower for token in missing_tokens)
    
    def process_pdf(self, file_path: str) -> PDFProcessingResult:
        """
        PDF文書の統合処理
        
        Args:
            file_path (str): PDFファイルのパス
            
        Returns:
            PDFProcessingResult: 処理結果
        """
        # 高分解能タイマーを使用して処理時間を計測
        import time as _time
        perf_start = _time.perf_counter()
        start_time = datetime.now()
        
        try:
            normalized_path = self._normalize_input_path(file_path)
            self.logger.info(f"PDF処理開始: {normalized_path}")

            if not os.path.isfile(normalized_path):
                raise FileNotFoundError(f"ファイルが見つかりません: {normalized_path}")

            self._ensure_fitz_can_open(normalized_path)

            # 各抽出処理を実行
            text_blocks = self.text_extractor.extract_text(normalized_path)
            tables = self.text_extractor.extract_tables(normalized_path)
            images = self.image_extractor.extract_images(normalized_path)
            processed_images = self.image_extractor.process_images(images)
            metadata = self.metadata_extractor.extract_metadata(normalized_path)
            structure = self.metadata_extractor.extract_structure(normalized_path)
            statistics = self.metadata_extractor.extract_statistics(normalized_path)
            
            # 品質スコアを計算
            quality_score = self._calculate_quality_score(
                text_blocks, tables, images, processed_images, metadata, structure, statistics
            )
            
            # 処理時間を計算（高分解能）
            processing_time = _time.perf_counter() - perf_start
            if processing_time <= 0:
                processing_time = 1e-6
            
            # 処理結果を構築
            result = PDFProcessingResult(
                success=True,
                file_path=normalized_path,
                processing_time=processing_time,
                text_blocks=text_blocks,
                tables=tables,
                images=images,
                processed_images=processed_images,
                metadata=metadata,
                structure=structure,
                statistics=statistics,
                quality_score=quality_score
            )
            
            # 統計を更新
            self._update_processing_stats(True, processing_time)
            
            self.logger.info(f"PDF処理完了: {file_path} (品質スコア: {quality_score:.2f})")
            return result
            
        except Exception as e:
            processing_time = _time.perf_counter() - perf_start
            # FileNotFoundError の場合は日本語メッセージを優先して返す
            if isinstance(e, FileNotFoundError):
                error_message = f"PDF処理エラー: ファイルが見つかりません: {self._safe_display_path(file_path)}"
            else:
                error_message = f"PDF処理エラー: {str(e)}"

            self.logger.error(error_message, exc_info=not isinstance(e, FileNotFoundError))
            
            # 統計を更新
            self._update_processing_stats(False, processing_time)
            
            # エラー結果を返す
            return PDFProcessingResult(
                success=False,
                file_path=self._safe_display_path(file_path),
                processing_time=processing_time,
                text_blocks=[],
                tables=[],
                images=[],
                processed_images=[],
                metadata=self.metadata_extractor._create_empty_metadata(),
                structure=self.metadata_extractor._create_empty_structure(),
                statistics=self.metadata_extractor._create_empty_statistics(),
                quality_score=0.0,
                error_message=error_message
            )
    
    def _calculate_quality_score(
        self,
        text_blocks: List[TextBlock],
        tables: List[TableData],
        images: List[ImageData],
        processed_images: List[ProcessedImage],
        metadata: PDFMetadata,
        structure: PDFStructure,
        statistics: PDFStatistics
    ) -> float:
        """
        品質スコアを計算
        
        Args:
            text_blocks: テキストブロック
            tables: 表データ
            images: 画像データ
            processed_images: 処理済み画像
            metadata: メタデータ
            structure: 構造情報
            statistics: 統計情報
            
        Returns:
            float: 品質スコア (0.0-1.0)
        """
        try:
            scores = []
            
            # テキスト抽出品質 (40%)
            if text_blocks:
                text_quality = min(len(text_blocks) / 100.0, 1.0)  # 100ブロックで満点
                scores.append(("text", text_quality, 0.4))
            else:
                scores.append(("text", 0.0, 0.4))
            
            # 表抽出品質 (20%)
            if tables:
                table_quality = min(len(tables) / 10.0, 1.0)  # 10表で満点
                scores.append(("table", table_quality, 0.2))
            else:
                scores.append(("table", 0.0, 0.2))
            
            # 画像抽出品質 (20%)
            if images:
                image_quality = min(len(images) / 20.0, 1.0)  # 20画像で満点
                scores.append(("image", image_quality, 0.2))
            else:
                scores.append(("image", 0.0, 0.2))
            
            # OCR品質 (10%)
            if processed_images:
                ocr_qualities = [img.ocr_confidence for img in processed_images if img.ocr_confidence > 0]
                ocr_quality = sum(ocr_qualities) / len(ocr_qualities) if ocr_qualities else 0.0
                scores.append(("ocr", ocr_quality, 0.1))
            else:
                scores.append(("ocr", 0.0, 0.1))
            
            # メタデータ品質 (10%)
            metadata_quality = 1.0 if metadata.title or metadata.author else 0.5
            scores.append(("metadata", metadata_quality, 0.1))
            
            # 重み付き平均を計算
            total_score = sum(score * weight for _, score, weight in scores)
            
            return min(total_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"品質スコア計算エラー: {e}")
            return 0.0
    
    def _update_processing_stats(self, success: bool, processing_time: float):
        """処理統計を更新"""
        self.processing_stats["total_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        if success:
            self.processing_stats["successful_processed"] += 1
        else:
            self.processing_stats["failed_processed"] += 1
    
    def process_multiple_pdfs(self, file_paths: List[str]) -> List[PDFProcessingResult]:
        """
        複数のPDFファイルを処理
        
        Args:
            file_paths (List[str]): PDFファイルのパスリスト
            
        Returns:
            List[PDFProcessingResult]: 処理結果のリスト
        """
        try:
            self.logger.info(f"複数PDF処理開始: {len(file_paths)}ファイル")
            
            results = []
            
            for file_path in file_paths:
                try:
                    result = self.process_pdf(file_path)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"PDF処理エラー ({file_path}): {e}")
                    # エラー結果を作成
                    error_result = PDFProcessingResult(
                        success=False,
                        file_path=file_path,
                        processing_time=0.0,
                        text_blocks=[],
                        tables=[],
                        images=[],
                        processed_images=[],
                        metadata=self.metadata_extractor._create_empty_metadata(),
                        structure=self.metadata_extractor._create_empty_structure(),
                        statistics=self.metadata_extractor._create_empty_statistics(),
                        quality_score=0.0,
                        error_message=str(e)
                    )
                    results.append(error_result)
            
            self.logger.info(f"複数PDF処理完了: {len(results)}ファイル")
            return results
            
        except Exception as e:
            self.logger.error(f"複数PDF処理エラー: {e}")
            return []
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """処理統計を取得"""
        stats = self.processing_stats.copy()
        
        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["successful_processed"] / stats["total_processed"]
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["average_processing_time"] = 0.0
        
        return stats
    
    def export_processing_result(self, result: PDFProcessingResult, output_path: str) -> bool:
        """
        処理結果をJSONファイルにエクスポート
        
        Args:
            result (PDFProcessingResult): 処理結果
            output_path (str): 出力ファイルパス
            
        Returns:
            bool: エクスポート成功フラグ
        """
        try:
            self.logger.info(f"処理結果エクスポート開始: {output_path}")
            
            # 結果を辞書に変換
            export_data = {
                "file_info": {
                    "file_path": result.file_path,
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "quality_score": result.quality_score,
                    "error_message": result.error_message,
                    "export_timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "title": result.metadata.title,
                    "author": result.metadata.author,
                    "subject": result.metadata.subject,
                    "creator": result.metadata.creator,
                    "producer": result.metadata.producer,
                    "creation_date": result.metadata.creation_date,
                    "modification_date": result.metadata.modification_date,
                    "keywords": result.metadata.keywords,
                    "page_count": result.metadata.page_count,
                    "file_size": result.metadata.file_size,
                    "pdf_version": result.metadata.pdf_version,
                    "encryption": result.metadata.encryption,
                    "permissions": result.metadata.permissions
                },
                "structure": {
                    "page_count": result.structure.page_count,
                    "page_dimensions": result.structure.page_dimensions,
                    "text_blocks_count": result.structure.text_blocks_count,
                    "image_blocks_count": result.structure.image_blocks_count,
                    "table_blocks_count": result.structure.table_blocks_count,
                    "font_info": result.structure.font_info,
                    "outline": result.structure.outline,
                    "annotations": result.structure.annotations
                },
                "statistics": {
                    "total_characters": result.statistics.total_characters,
                    "total_words": result.statistics.total_words,
                    "total_lines": result.statistics.total_lines,
                    "average_words_per_page": result.statistics.average_words_per_page,
                    "average_characters_per_page": result.statistics.average_characters_per_page,
                    "text_density": result.statistics.text_density,
                    "image_density": result.statistics.image_density,
                    "table_density": result.statistics.table_density
                },
                "content": {
                    "text_blocks": [
                        {
                            "text": block.text,
                            "page_number": block.page_number,
                            "bbox": block.bbox,
                            "font_size": block.font_size,
                            "font_name": block.font_name,
                            "is_bold": block.is_bold,
                            "is_italic": block.is_italic,
                            "block_type": block.block_type,
                            "confidence": block.confidence
                        }
                        for block in result.text_blocks
                    ],
                    "tables": [
                        {
                            "table_id": table.table_id,
                            "page_number": table.page_number,
                            "bbox": table.bbox,
                            "rows": table.rows,
                            "headers": table.headers,
                            "confidence": table.confidence
                        }
                        for table in result.tables
                    ],
                    "images": [
                        {
                            "image_id": img.image_id,
                            "page_number": img.page_number,
                            "bbox": img.bbox,
                            "width": img.width,
                            "height": img.height,
                            "format": img.image_format,
                            "confidence": img.confidence
                        }
                        for img in result.images
                    ],
                    "processed_images": [
                        {
                            "image_id": proc_img.image_id,
                            "ocr_text": proc_img.ocr_text,
                            "ocr_confidence": proc_img.ocr_confidence,
                            "image_type": proc_img.image_type,
                            "processing_metadata": proc_img.processing_metadata
                        }
                        for proc_img in result.processed_images
                    ]
                }
            }
            
            # JSONファイルに保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info("処理結果エクスポート完了")
            return True
            
        except Exception as e:
            self.logger.error(f"処理結果エクスポートエラー: {e}")
            return False
    
    def validate_processing_result(self, result: PDFProcessingResult) -> Dict[str, Any]:
        """
        処理結果の検証
        
        Args:
            result (PDFProcessingResult): 処理結果
            
        Returns:
            Dict[str, Any]: 検証結果
        """
        try:
            validation_result: Dict[str, Any] = {
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "recommendations": []
            }
            
            # 基本検証
            if not result.success:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"処理失敗: {result.error_message}")
                return validation_result
            
            # テキスト抽出検証
            if not result.text_blocks:
                validation_result["warnings"].append("テキストが抽出されませんでした")
            elif len(result.text_blocks) < 5:
                validation_result["warnings"].append("テキストブロック数が少ないです")
            
            # 画像抽出検証
            if result.images and not result.processed_images:
                validation_result["warnings"].append("画像は抽出されましたが、処理に失敗しました")
            
            # 品質スコア検証
            if result.quality_score < 0.5:
                validation_result["warnings"].append(f"品質スコアが低いです: {result.quality_score:.2f}")
                validation_result["recommendations"].append("PDFの品質を確認してください")
            
            # メタデータ検証
            if not result.metadata.title and not result.metadata.author:
                validation_result["warnings"].append("メタデータが不足しています")
            
            # 処理時間検証
            if result.processing_time > 30.0:
                validation_result["warnings"].append(f"処理時間が長いです: {result.processing_time:.2f}秒")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"処理結果検証エラー: {e}")
            return {
                "is_valid": False,
                "issues": [f"検証エラー: {str(e)}"],
                "warnings": [],
                "recommendations": []
            }
    
    def cleanup_temp_files(self, result: PDFProcessingResult) -> bool:
        """
        一時ファイルのクリーンアップ
        
        Args:
            result (PDFProcessingResult): 処理結果
            
        Returns:
            bool: クリーンアップ成功フラグ
        """
        try:
            self.logger.info("一時ファイルクリーンアップ開始")
            
            # 画像ファイルのクリーンアップ
            for image in result.images:
                # 画像ファイルが存在する場合は削除
                # 実際の実装では、画像ファイルのパスを管理する必要がある
                pass
            
            self.logger.info("一時ファイルクリーンアップ完了")
            return True
            
        except Exception as e:
            self.logger.error(f"一時ファイルクリーンアップエラー: {e}")
            return False
