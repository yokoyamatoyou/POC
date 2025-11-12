"""
ファイルタイプ別高度表示システム
画像サムネイル、Excel表プレビュー、PDFページプレビュー等の高度な表示機能
"""

import logging
import base64
import io
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)

class FileType(Enum):
    """ファイルタイプ"""
    IMAGE = "image"
    PDF = "pdf"
    EXCEL = "excel"
    WORD = "word"
    TEXT = "text"
    TABULAR = "tabular"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    CODE = "code"
    HTML = "html"
    UNKNOWN = "unknown"

@dataclass
class DisplayConfig:
    """表示設定"""
    show_thumbnail: bool = True
    show_preview: bool = True
    max_preview_size: int = 500
    thumbnail_size: Tuple[int, int] = (200, 150)
    enable_interactive: bool = True

class FileTypeDisplay:
    """ファイルタイプ別表示クラス"""
    
    def __init__(self, config: Optional[DisplayConfig] = None):
        self.config = config or DisplayConfig()

    def _resolve_existing_path(self, *candidates: Optional[str]) -> Optional[Path]:
        for candidate in candidates:
            if not candidate or not isinstance(candidate, str):
                continue
            try:
                path_obj = Path(candidate)
                if path_obj.exists():
                    return path_obj
                alt_path = Path.cwd() / candidate
                if alt_path.exists():
                    return alt_path
            except Exception:
                continue
        return None

    def _render_source_reference(
        self,
        chunk_id: str,
        metadata: Dict[str, Any],
        file_path: Optional[str],
    ) -> None:
        stage1 = metadata.get("stage1_basic") or {}
        stage2 = metadata.get("stage2_processing") or {}
        source_path = self._resolve_existing_path(
            metadata.get("stored_path"),
            metadata.get("file_path"),
            metadata.get("source_file"),
            stage1.get("stored_path"),
            stage1.get("file_path"),
            stage2.get("stored_path"),
            file_path,
        )

        if not source_path:
            return

        st.caption(f"元ファイル: {source_path}")
        try:
            with source_path.open("rb") as file_obj:
                st.download_button(
                    label="元ファイルをダウンロード",
                    data=file_obj,
                    file_name=source_path.name,
                    key=f"download_{chunk_id}",
                )
        except Exception as exc:
            st.caption(f"ダウンロード準備に失敗しました: {exc}")
    
    def detect_file_type(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> FileType:
        """ファイルタイプを検出"""
        if metadata and metadata.get('file_type'):
            type_str = metadata['file_type'].lower()
            if 'image' in type_str or 'png' in type_str or 'jpg' in type_str or 'jpeg' in type_str:
                return FileType.IMAGE
            elif 'pdf' in type_str:
                return FileType.PDF
            elif 'excel' in type_str or 'xlsx' in type_str or 'xls' in type_str:
                return FileType.EXCEL
            elif 'word' in type_str or 'docx' in type_str or 'doc' in type_str:
                return FileType.WORD
            elif 'html' in type_str:
                return FileType.HTML
            elif 'tabular' in type_str:
                return FileType.TABULAR
            elif type_str == 'json':
                return FileType.JSON
            elif type_str in {'yaml', 'yml'}:
                return FileType.YAML
            elif type_str == 'xml':
                return FileType.XML
            elif type_str == 'code':
                return FileType.CODE
            elif 'text' in type_str or 'txt' in type_str:
                return FileType.TEXT
        
        # ファイル拡張子から判定
        ext = Path(file_path).suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            return FileType.IMAGE
        elif ext == '.pdf':
            return FileType.PDF
        elif ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
            return FileType.EXCEL
        elif ext in ['.docx', '.doc']:
            return FileType.WORD
        elif ext in ['.txt', '.md', '.csv']:
            return FileType.TEXT
        elif ext in ['.csv', '.tsv']:
            return FileType.TABULAR
        elif ext == '.json':
            return FileType.JSON
        elif ext in ['.yaml', '.yml']:
            return FileType.YAML
        elif ext == '.xml':
            return FileType.XML
        elif ext in ['.py', '.java', '.bas', '.cls', '.vba']:
            return FileType.CODE
        elif ext in ['.html', '.htm']:
            return FileType.HTML
        else:
            return FileType.UNKNOWN
    
    def display_search_result(self, 
                            chunk_id: str,
                            text: str,
                            score: float,
                            metadata: Dict[str, Any],
                            file_path: Optional[str] = None,
                            summary: Optional[str] = None) -> None:
        """検索結果をファイルタイプに応じて表示"""
        file_type = self.detect_file_type(file_path or "", metadata)
        
        with st.container():
            # 共通ヘッダー
            st.markdown(f"**チャンクID**: `{chunk_id}`")
            st.markdown(f"**関連度スコア**: {score:.3f}")

            if isinstance(metadata, dict):
                metadata = dict(metadata)
            else:
                metadata = {}

            normalized_summary = summary.strip() if isinstance(summary, str) else ""
            if normalized_summary:
                metadata.setdefault('query_summary', normalized_summary)

            summary_text = normalized_summary or metadata.get('summary')
            if isinstance(summary_text, str) and summary_text.strip():
                st.markdown("**要約**")
                st.write(summary_text.strip())

            # 元ファイル
            self._render_source_reference(chunk_id, metadata, file_path)

            # ファイルタイプ別表示
            if file_type == FileType.IMAGE:
                self._display_image_result(chunk_id, text, metadata, file_path)
            elif file_type == FileType.PDF:
                self._display_pdf_result(chunk_id, text, metadata, file_path)
            elif file_type == FileType.EXCEL:
                self._display_excel_result(chunk_id, text, metadata, file_path)
            elif file_type == FileType.WORD:
                self._display_word_result(chunk_id, text, metadata, file_path)
            elif file_type == FileType.TABULAR:
                self._display_tabular_result(chunk_id, text, metadata, file_path)
            elif file_type in (FileType.JSON, FileType.YAML, FileType.XML):
                self._display_structured_result(chunk_id, text, metadata, file_path)
            elif file_type == FileType.CODE:
                self._display_code_result(chunk_id, text, metadata, file_path)
            elif file_type == FileType.HTML:
                self._display_html_result(chunk_id, text, metadata, file_path)
            elif metadata.get('file_type') == 'archive' or file_type == FileType.UNKNOWN and metadata.get('stage1_basic', {}).get('entry_count'):
                self._display_archive_result(chunk_id, text, metadata, file_path)
            else:
                self._display_text_result(chunk_id, text, metadata, file_path)
            
            st.divider()
    
    def _display_image_result(self, 
                            chunk_id: str,
                            text: str,
                            metadata: Dict[str, Any],
                            file_path: Optional[str] = None) -> None:
        """画像結果の表示"""
        st.markdown("**画像コンテンツ**")

        chunk_type = (metadata.get('file_type') or '').lower()
        image_meta = metadata.get('image_metadata', {}) if isinstance(metadata.get('image_metadata'), dict) else {}

        # 画像サムネイル表示（可能な限りファイルパスから表示）
        thumbnail_path = self._resolve_existing_path(
            metadata.get('image_path'),
            metadata.get('stored_path'),
            (metadata.get('stage1_basic') or {}).get('stored_path'),
            file_path,
        )
        if self.config.show_thumbnail and thumbnail_path:
            try:
                if thumbnail_path.exists():
                    try:
                        st.image(str(thumbnail_path), width=self.config.thumbnail_size[0])
                    except Exception:
                        with thumbnail_path.open('rb') as f:
                            st.image(f.read(), width=self.config.thumbnail_size[0])
                    st.caption(f"画像ファイル: {thumbnail_path.name}")
                else:
                    st.info("画像ファイルが見つかりません。ナレッジ登録時の保存先を確認してください。")
            except Exception as e:
                logger.warning(f"画像表示エラー: {e}")
                st.warning("サムネイルの表示に失敗しました")

        stage2_meta = metadata.get('stage2_processing') or {}
        stage3_meta = metadata.get('stage3_business') or {}
        stage4_meta = metadata.get('stage4_search') or {}

        # 要約と検索ヒント
        summary_text = metadata.get('query_summary') or metadata.get('summary') or stage2_meta.get('image_summary')
        if not summary_text and chunk_type == 'image_caption' and text.strip():
            summary_text = text.strip()

        if summary_text:
            st.markdown("**要約**")
            st.write(summary_text.strip())

        # ビジネスコンテキスト
        business_context = (
            metadata.get('business_context')
            or stage3_meta.get('image_use_case')
        )
        if business_context:
            st.caption(f"想定用途: {business_context}")

        scene_type = metadata.get('scene_type') or stage2_meta.get('scene_type') or stage3_meta.get('scene_type')
        if scene_type:
            st.markdown(f"**シーンタイプ**: {scene_type}")

        environment = metadata.get('environment') or stage2_meta.get('environment')
        if environment:
            st.markdown(f"**環境/状況**: {environment}")

        # 検索ヒント
        search_hints = metadata.get('search_metadata') or stage4_meta.get('search_metadata', [])
        if search_hints:
            with st.expander("検索ヒント", expanded=True):
                st.markdown("\n".join(f"- {hint}" for hint in search_hints))

        # タグ表示
        tags = metadata.get('tags') or image_meta.get('tags') or []
        if tags:
            st.markdown(
                """
                <div style="display:flex;flex-wrap:wrap;gap:0.4rem;">
                """
                + "".join(
                    f"<span style='background:#eef2ff;color:#1f2a6b;padding:0.2rem 0.5rem;border-radius:12px;font-size:0.8rem;'>{tag}</span>"
                    for tag in tags
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        keyword_chips = metadata.get('keywords') or stage4_meta.get('keywords')
        if keyword_chips and not tags:
            st.markdown(
                """
                <div style="display:flex;flex-wrap:wrap;gap:0.4rem;">
                """
                + "".join(
                    f"<span style='background:#e0f2ff;color:#084c8d;padding:0.2rem 0.5rem;border-radius:12px;font-size:0.8rem;'>{kw}</span>"
                    for kw in keyword_chips
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        def _render_chip_group(label: str, items: Optional[List[str]], *, badge_color: str = "#f1f5f9", text_color: str = "#0f172a") -> None:
            if not items:
                return
            normalized = [str(item).strip() for item in items if str(item).strip()]
            if not normalized:
                return
            st.markdown(f"**{label}**")
            st.markdown(
                """
                <div style="display:flex;flex-wrap:wrap;gap:0.4rem;">
                """
                + "".join(
                    f"<span style='background:{badge_color};color:{text_color};padding:0.2rem 0.55rem;border-radius:12px;font-size:0.78rem;'>{item}</span>"
                    for item in normalized
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        _render_chip_group("行動", metadata.get('actions') or stage2_meta.get('actions'), badge_color="#fef3c7", text_color="#92400e")
        _render_chip_group("主要オブジェクト", metadata.get('objects') or stage2_meta.get('objects'), badge_color="#ecfeff", text_color="#0e7490")

        visible_text = metadata.get('visible_text') or stage2_meta.get('visible_text')
        if visible_text:
            with st.expander("画像内テキスト", expanded=False):
                st.markdown("\n".join(f"- {line}" for line in visible_text if str(line).strip()))

        numbers = metadata.get('numbers') or stage2_meta.get('numbers')
        if numbers:
            _render_chip_group("数値・識別子", numbers, badge_color="#fce7f3", text_color="#831843")

        recommended_queries = (
            metadata.get('recommended_queries')
            or metadata.get('search_terms')
            or stage4_meta.get('recommended_queries')
        )
        if recommended_queries:
            with st.expander("推奨クエリ", expanded=False):
                st.markdown("\n".join(f"- {query}" for query in recommended_queries if str(query).strip()))

        # 画像メタデータ表示
        if image_meta:
            col1, col2 = st.columns(2)
            with col1:
                if image_meta.get('width') and image_meta.get('height'):
                    st.markdown(f"**サイズ**: {image_meta.get('width')} × {image_meta.get('height')}")
                aspect_ratio = image_meta.get('aspect_ratio')
                if isinstance(aspect_ratio, (int, float)):
                    st.markdown(f"**アスペクト比**: {aspect_ratio:.2f}")
                elif aspect_ratio:
                    st.markdown(f"**アスペクト比**: {aspect_ratio}")
            with col2:
                if image_meta.get('image_type'):
                    st.markdown(f"**画像タイプ**: {image_meta.get('image_type')}")
                if image_meta.get('has_text') is not None:
                    st.markdown(f"**テキスト含有**: {'あり' if image_meta.get('has_text') else 'なし'}")

        # OCRテキスト表示
        ocr_text = image_meta.get('ocr_text')
        if chunk_type == 'image_ocr' and text.strip():
            ocr_text = text.strip()
        if ocr_text:
            with st.expander("OCRテキスト", expanded=False):
                st.text(ocr_text)
    
    def _display_pdf_result(self, 
                          chunk_id: str,
                          text: str,
                          metadata: Dict[str, Any],
                          file_path: Optional[str] = None) -> None:
        """PDF結果の表示"""
        st.markdown("**PDF文書**")
        
        # PDF情報表示
        col1, col2 = st.columns(2)
        with col1:
            page_label = metadata.get('page_label') or metadata.get('page_number')
            if page_label is not None:
                st.markdown(f"**ページ**: {page_label}")
            if 'total_pages' in metadata:
                st.markdown(f"**総ページ数**: {metadata['total_pages']}")
        
        with col2:
            if 'file_size' in metadata:
                size_mb = metadata['file_size'] / (1024 * 1024)
                st.markdown(f"**ファイルサイズ**: {size_mb:.2f} MB")
            if 'creation_date' in metadata:
                st.markdown(f"**作成日**: {metadata['creation_date']}")
        
        # PDFページプレビュー
        if self.config.show_preview and 'pdf_page_image' in metadata:
            try:
                page_image_path = metadata['pdf_page_image']
                if Path(page_image_path).exists():
                    with open(page_image_path, 'rb') as f:
                        page_image = f.read()
                        st.image(page_image, width=self.config.max_preview_size)
            except Exception as e:
                logger.warning(f"PDFページプレビューエラー: {e}")
        
        # テキスト内容表示
        if text.strip():
            with st.expander("テキスト内容", expanded=False):
                st.text(text)

        reference_links = metadata.get('reference_links') or (metadata.get('stage4_search') or {}).get('reference_links')
        if isinstance(reference_links, list) and reference_links:
            with st.expander("参照リンク", expanded=True):
                for idx, entry in enumerate(reference_links, 1):
                    if not isinstance(entry, dict):
                        continue

                    label = entry.get('label')
                    if not label:
                        if entry.get('type') == 'page' and entry.get('page_number'):
                            label = f"p.{entry['page_number']}"
                        else:
                            label = f"参照{idx}"

                    status_label = "[解決]" if entry.get('resolved') else "[未解決]"
                    st.markdown(f"{status_label} **{label}**")

                    target_summaries = entry.get('target_chunk_summaries')
                    target_chunks = entry.get('target_chunks')

                    if isinstance(target_summaries, list) and target_summaries:
                        for target in target_summaries:
                            if not isinstance(target, dict):
                                continue
                            page_label = target.get('page')
                            summary_text = target.get('summary')
                            chunk_id = target.get('chunk_id')
                            prefix = f"p.{page_label}" if page_label is not None else (chunk_id or "対象チャンク")
                            st.markdown(f"• {prefix}: {summary_text}")
                    elif isinstance(target_chunks, list) and target_chunks:
                        display_ids = ", ".join(map(str, target_chunks))
                        st.caption(f"参照チャンク: {display_ids}")
                    else:
                        st.caption("参照先チャンクが未解決です。")
    
    def _display_excel_result(self, 
                            chunk_id: str,
                            text: str,
                            metadata: Dict[str, Any],
                            file_path: Optional[str] = None) -> None:
        """Excel結果の表示"""

        is_shape_chunk = metadata.get('file_type') == 'excel_shape'
        header_label = "**Excel図形**" if is_shape_chunk else "**Excel文書**"
        st.markdown(header_label)
        
        # Excel情報表示
        col1, col2 = st.columns(2)
        with col1:
            if 'sheet_name' in metadata:
                st.markdown(f"**シート名**: {metadata['sheet_name']}")
            if not is_shape_chunk and 'sheet_number' in metadata:
                st.markdown(f"**シート番号**: {metadata['sheet_number']}")
            if is_shape_chunk:
                if metadata.get('shape_name'):
                    st.markdown(f"**図形名**: {metadata['shape_name']}")
                position = metadata.get('position') or {}
                row = position.get('row')
                col = position.get('col')
                if row is not None or col is not None:
                    row_label = f"R{row + 1}" if row is not None else "R-"
                    col_label = f"C{col + 1}" if col is not None else "C-"
                    st.markdown(f"**配置セル**: {row_label} / {col_label}")
        
        with col2:
            if not is_shape_chunk:
                if 'total_sheets' in metadata:
                    st.markdown(f"**総シート数**: {metadata['total_sheets']}")
                if 'has_images' in metadata:
                    st.markdown(f"**画像含有**: {'あり' if metadata['has_images'] else 'なし'}")
            else:
                if metadata.get('shape_type'):
                    st.markdown(f"**図形タイプ**: {metadata['shape_type']}")
                sources = metadata.get('sources')
                source_value = metadata.get('source')
                display_source = None
                if isinstance(sources, list) and sources:
                    unique_sources = list(dict.fromkeys(sources))
                    display_source = " + ".join(src.upper() for src in unique_sources)
                elif isinstance(source_value, str):
                    if source_value.lower() == "mixed":
                        display_source = "XML + OCR"
                    else:
                        display_source = source_value.upper()
                if display_source:
                    st.markdown(f"**抽出ソース**: {display_source}")
        
        # Excel表プレビュー（セルチャンクのみ）
        if not is_shape_chunk and self.config.show_preview and 'excel_preview' in metadata:
            try:
                preview_data = metadata['excel_preview']
                if isinstance(preview_data, list) and preview_data:
                    st.markdown("**表プレビュー**")
                    # 簡易的な表表示
                    for i, row in enumerate(preview_data[:5]):  # 最大5行
                        if isinstance(row, list):
                            st.markdown(f"行{i+1}: {' | '.join(str(cell) for cell in row[:5])}")
            except Exception as e:
                logger.warning(f"Excelプレビューエラー: {e}")
        
        # テキスト内容表示
        if text.strip():
            expander_label = "図形テキスト" if is_shape_chunk else "セル内容"
            with st.expander(expander_label, expanded=False):
                st.text(text)

        if is_shape_chunk:
            context_excerpt = metadata.get('context_excerpt')
            context_cells = metadata.get('context_cells')
            context_headers = metadata.get('context_headers')
            selected_excerpt = metadata.get('context_selected_excerpt') or (
                metadata.get('stage4_search', {}).get('context_selected_excerpt')
                if isinstance(metadata.get('stage4_search'), dict)
                else None
            )
            selected_cells = metadata.get('context_selected_cells') or (
                metadata.get('stage4_search', {}).get('context_selected_cells')
                if isinstance(metadata.get('stage4_search'), dict)
                else None
            )
            selection_method = metadata.get('context_selection_method') or (
                metadata.get('stage4_search', {}).get('context_selection_method')
                if isinstance(metadata.get('stage4_search'), dict)
                else None
            )
            if selected_excerpt or context_excerpt or context_cells or context_headers:
                with st.expander("周辺セルコンテキスト", expanded=False):
                    if isinstance(context_headers, list) and context_headers:
                        header_preview = " / ".join(str(item) for item in context_headers[:5])
                        st.markdown(f"**推定ヘッダー**: {header_preview}")
                    if isinstance(selected_excerpt, str) and selected_excerpt.strip():
                        st.markdown("**厳選セル**")
                        st.markdown(selected_excerpt.replace("\n", "  \n"))
                        if isinstance(selected_cells, list) and selected_cells:
                            method_label = selection_method or "LLM推奨"
                            st.caption(f"選択手法: {method_label}")
                    elif isinstance(context_excerpt, str) and context_excerpt.strip():
                        st.markdown(context_excerpt.replace("\n", "  \n"))
                    if isinstance(context_cells, list) and context_cells:
                        limit = 10
                        for cell in context_cells[:limit]:
                            label = cell.get('label') or f"R{cell.get('row')}C{cell.get('col')}"
                            text_value = cell.get('text')
                            st.markdown(f"- {label}: {text_value}")
                        if len(context_cells) > limit:
                            st.caption(f"…ほか {len(context_cells) - limit} 件")
    
    def _display_word_result(self, 
                           chunk_id: str,
                           text: str,
                           metadata: Dict[str, Any],
                           file_path: Optional[str] = None) -> None:
        """Word結果の表示"""
        st.markdown("**Word文書**")
        
        # Word情報表示
        col1, col2 = st.columns(2)
        with col1:
            if 'paragraph_count' in metadata:
                st.markdown(f"**段落数**: {metadata['paragraph_count']}")
            if 'has_images' in metadata:
                st.markdown(f"**画像含有**: {'あり' if metadata['has_images'] else 'なし'}")
        
        with col2:
            if 'word_count' in metadata:
                st.markdown(f"**単語数**: {metadata['word_count']}")
            if 'has_tables' in metadata:
                st.markdown(f"**表含有**: {'あり' if metadata['has_tables'] else 'なし'}")
        
        # テキスト内容表示
        if text.strip():
            with st.expander("文書内容", expanded=False):
                st.text(text)
    
    def _display_text_result(self, 
                           chunk_id: str,
                           text: str,
                           metadata: Dict[str, Any],
                           file_path: Optional[str] = None) -> None:
        """テキスト結果の表示"""
        st.markdown("**テキスト文書**")
        
        # テキスト情報表示
        col1, col2 = st.columns(2)
        with col1:
            if 'line_count' in metadata:
                st.markdown(f"**行数**: {metadata['line_count']}")
            if 'word_count' in metadata:
                st.markdown(f"**単語数**: {metadata['word_count']}")
        
        with col2:
            if 'char_count' in metadata:
                st.markdown(f"**文字数**: {metadata['char_count']}")
            if 'encoding' in metadata:
                st.markdown(f"**エンコーディング**: {metadata['encoding']}")
        
        # テキスト内容表示
        if text.strip():
            with st.expander("文書内容", expanded=False):
                st.text(text)

    def _display_tabular_result(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        file_path: Optional[str] = None,
    ) -> None:
        st.markdown("**表形式データ**")

        summary = metadata.get("csv_summary") or {}
        col1, col2 = st.columns(2)
        with col1:
            if summary.get("row_count") is not None:
                st.markdown(f"**データ行数**: {summary.get('row_count')}")
            if summary.get("column_count") is not None:
                st.markdown(f"**列数**: {summary.get('column_count')}")
        with col2:
            delimiter = metadata.get("delimiter", "comma")
            st.markdown(f"**区切り文字**: {delimiter}")
            column_names = summary.get("column_names")
            if column_names:
                st.markdown(f"**ヘッダー**: {', '.join(column_names[:6])}")

        preview = metadata.get("preview_rows")
        if preview:
            with st.expander("プレビュー", expanded=True):
                for idx, row in enumerate(preview):
                    st.markdown(f"行{idx + 1}: {' | '.join(str(cell) for cell in row[:8])}")

        summary_text = metadata.get("stage2_processing", {}).get("summary")
        if summary_text:
            st.write(summary_text)

        search_hints = metadata.get("stage4_search", {}).get("search_metadata") or []
        if search_hints:
            with st.expander("検索ヒント", expanded=True):
                st.markdown("\n".join(f"- {hint}" for hint in search_hints))

        tags = metadata.get("tags", [])
        if tags:
            st.markdown(
                "<div style=\"display:flex;flex-wrap:wrap;gap:0.4rem;\">"
                + "".join(
                    f"<span style='background:#eef2ff;color:#1f2a6b;padding:0.2rem 0.5rem;border-radius:12px;font-size:0.8rem;'>{tag}</span>"
                    for tag in tags
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        if text.strip():
            with st.expander("チャンク内容", expanded=False):
                st.text(text)

    def _display_structured_result(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        file_path: Optional[str] = None,
    ) -> None:
        st.markdown("**構造化データ**")

        entry_count = metadata.get("entry_count")
        top_keys = (
            metadata.get("json_top_keys")
            or metadata.get("top_keys")
            or metadata.get("stage2_processing", {}).get("top_keys")
        )
        col1, col2 = st.columns(2)
        with col1:
            if entry_count is not None:
                st.markdown(f"**エントリ数**: {entry_count}")
            if metadata.get("document_count"):
                st.markdown(f"**ドキュメント数**: {metadata.get('document_count')}")
        with col2:
            if top_keys:
                st.markdown(f"**主要キー**: {', '.join(str(k) for k in top_keys[:6])}")
            if metadata.get("root_tag"):
                st.markdown(f"**ルートタグ**: {metadata.get('root_tag')}")

        summary_text = metadata.get("stage2_processing", {}).get("summary")
        if summary_text:
            st.write(summary_text)

        search_hints = metadata.get("stage4_search", {}).get("search_metadata") or []
        if search_hints:
            with st.expander("検索ヒント", expanded=True):
                st.markdown("\n".join(f"- {hint}" for hint in search_hints))

        tags = metadata.get("tags", [])
        if tags:
            st.markdown(
                "<div style=\"display:flex;flex-wrap:wrap;gap:0.4rem;\">"
                + "".join(
                    f"<span style='background:#e9f5ff;color:#0b3d91;padding:0.2rem 0.5rem;border-radius:12px;font-size:0.8rem;'>{tag}</span>"
                    for tag in tags
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        if text.strip():
            with st.expander("チャンク内容", expanded=False):
                st.text(text)

    def _display_code_result(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        file_path: Optional[str] = None,
    ) -> None:
        st.markdown("**コードスニペット**")

        language = metadata.get("language", "unknown").upper()
        imports = metadata.get("imports", [])
        top_symbols = metadata.get("top_symbols", [])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**言語**: {language}")
            if imports:
                st.markdown(f"**主要インポート**: {', '.join(imports[:5])}")
        with col2:
            if top_symbols:
                symbol_names = [symbol.get("name", "") for symbol in top_symbols[:5]]
                st.markdown(f"**主要シンボル**: {', '.join(symbol_names)}")

        summary_text = metadata.get("stage2_processing", {}).get("summary")
        if summary_text:
            st.write(summary_text)

        search_hints = metadata.get("stage4_search", {}).get("search_metadata") or []
        if search_hints:
            with st.expander("検索ヒント", expanded=True):
                st.markdown("\n".join(f"- {hint}" for hint in search_hints))

        tags = metadata.get("tags", [])
        if tags:
            st.markdown(
                "<div style=\"display:flex;flex-wrap:wrap;gap:0.4rem;\">"
                + "".join(
                    f"<span style='background:#eefbf3;color:#0f5132;padding:0.2rem 0.5rem;border-radius:12px;font-size:0.8rem;'>{tag}</span>"
                    for tag in tags
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        symbol_meta = metadata.get("symbol_name") or metadata.get("metadata", {}).get("symbol_name")
        if symbol_meta:
            st.caption(f"シンボル: {symbol_meta}")

        if text.strip():
            with st.expander("コードチャンク", expanded=True):
                st.code(text, language=language.lower())

    def _display_html_result(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        file_path: Optional[str] = None,
    ) -> None:
        st.markdown("🌐 **HTMLセクション**")

        stage1 = metadata.get("stage1_basic") or {}
        section_meta = metadata.get("metadata") or metadata

        page_title = stage1.get("page_title") or metadata.get("page_title")
        canonical_url = stage1.get("canonical_url") or metadata.get("canonical_url")
        if page_title:
            st.markdown(f"**ページタイトル**: {page_title}")
        if canonical_url:
            st.caption(f"URL: {canonical_url}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if stage1.get("domain"):
                st.markdown(f"**ドメイン**: {stage1['domain']}")
            if stage1.get("site_name"):
                st.markdown(f"**サイト名**: {stage1['site_name']}")
        with col2:
            if stage1.get("language"):
                st.markdown(f"**言語**: {stage1['language']}")
            if stage1.get("page_type_hint"):
                st.markdown(f"**ページ種別**: {stage1['page_type_hint']}")
        with col3:
            if stage1.get("organization"):
                st.markdown(f"**組織名**: {stage1['organization']}")
            if stage1.get("author"):
                st.markdown(f"**作成者**: {stage1['author']}")

        section_path = section_meta.get("section_path")
        heading_text = section_meta.get("heading_text") or section_meta.get("heading")
        anchor_id = section_meta.get("anchor_id") or section_meta.get("anchor")
        with st.expander("セクション情報", expanded=True):
            if section_path:
                st.markdown(f"**パンくず**: {section_path}")
            if heading_text:
                st.markdown(f"**見出し**: {heading_text}")
            if anchor_id:
                st.markdown(f"**アンカー**: `#{anchor_id}`")
            counts = [
                ("段落", section_meta.get("paragraph_count")),
                ("リスト", section_meta.get("list_count")),
                ("表", section_meta.get("table_count")),
                ("画像", section_meta.get("image_count")),
                ("リンク", section_meta.get("link_count")),
            ]
            count_line = ", ".join(f"{label}:{count}" for label, count in counts if isinstance(count, int) and count > 0)
            if count_line:
                st.markdown(f"**要素数**: {count_line}")

        summary_text = metadata.get("stage2_processing", {}).get("summary")
        if summary_text and summary_text != text.strip():
            st.markdown("**ページ要約**")
            st.write(summary_text)

        search_hints = metadata.get("stage4_search", {}).get("search_metadata") or []
        recommended_queries = metadata.get("stage4_search", {}).get("recommended_queries") or []
        if search_hints or recommended_queries:
            with st.expander("検索ヒント", expanded=True):
                if search_hints:
                    st.markdown("**キーワード候補**")
                    st.markdown("\n".join(f"- {hint}" for hint in search_hints))
                if recommended_queries:
                    st.markdown("**おすすめ検索クエリ**")
                    st.markdown("\n".join(f"- {query}" for query in recommended_queries))

        link_summary = metadata.get("link_summary") or section_meta.get("links") or []
        if link_summary:
            with st.expander("主要リンク", expanded=False):
                for link in link_summary[:5]:
                    href = link.get("href")
                    text_value = link.get("text") or href
                    domain = link.get("domain")
                    badge = f"<span style='background:#e8f1ff;color:#0b5394;padding:0.15rem 0.4rem;border-radius:10px;font-size:0.75rem;margin-right:0.4rem;'>{domain}</span>" if domain else ""
                    if href and href.startswith("http"):
                        st.markdown(f"- {badge}[{text_value}]({href})", unsafe_allow_html=True)
                    else:
                        st.markdown(f"- {badge}{text_value}", unsafe_allow_html=True)

        image_summary = metadata.get("image_summary") or section_meta.get("images") or []
        if image_summary:
            with st.expander("画像一覧", expanded=False):
                for img_meta in image_summary[:5]:
                    description = img_meta.get("description") or img_meta.get("alt") or "画像"
                    src = img_meta.get("src")
                    st.markdown(f"- {description}")
                    if src:
                        st.caption(src)

        tags = metadata.get("tags") or []
        if tags:
            st.markdown(
                "<div style=\"display:flex;flex-wrap:wrap;gap:0.4rem;\">"
                + "".join(
                    f"<span style='background:#f1f5f9;color:#1f2937;padding:0.2rem 0.6rem;border-radius:12px;font-size:0.8rem;'>{tag}</span>"
                    for tag in tags
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        if text.strip():
            with st.expander("セクション内容", expanded=True):
                st.text(text)

    def _display_archive_result(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        file_path: Optional[str] = None,
    ) -> None:
        st.markdown("**アーカイブ**")

        stage1 = metadata.get("stage1_basic", {})
        st.markdown(f"**アーカイブ名**: {stage1.get('file_name', Path(file_path or '').name)}")
        st.caption(f"パス: {stage1.get('file_path', file_path or '')}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**展開ファイル数**: {stage1.get('entry_count', 0)}")
        with col2:
            total_size = stage1.get('total_unpacked_size', 0)
            if total_size:
                st.markdown(f"**展開サイズ**: {total_size / (1024 * 1024):.2f} MB")

        warnings = stage1.get('warnings', []) or metadata.get('processing_warnings', [])
        if warnings:
            with st.expander("警告", expanded=False):
                for warn in warnings:
                    st.markdown(f"- {warn}")

        entries = metadata.get("archive_entries", [])
        if entries:
            with st.expander("展開ファイル一覧", expanded=True):
                for entry in entries:
                    name = entry.get("file_name")
                    size = entry.get("file_size", 0)
                    ext = entry.get("file_extension", "")
                    st.markdown(f"- `{name}` ({ext or 'unknown'} / {size} bytes)")

        if text.strip():
            with st.expander("チャンク内容", expanded=False):
                st.text(text)

class SearchResultsDisplay:
    """検索結果表示管理クラス"""
    
    def __init__(self, config: Optional[DisplayConfig] = None):
        self.display = FileTypeDisplay(config)
    
    def display_search_results(self, 
                             results: List[Dict[str, Any]],
                             query: str,
                             max_results: int = 10) -> None:
        """検索結果を一括表示"""
        st.markdown(f"### 検索結果: '{query}'")
        st.markdown(f"**{len(results)}件**の結果が見つかりました")
        
        if not results:
            st.info("検索結果が見つかりませんでした")
            return
        
        # 結果を制限
        display_results = results[:max_results]
        
        # 各結果を表示
        for i, result in enumerate(display_results, 1):
            with st.container():
                st.markdown(f"#### 結果 {i}")
                
                # 必要な情報を抽出
                chunk_id = result.get('chunk_id', f'unknown-{i}')
                text = result.get('text', result.get('content', ''))
                score = result.get('score', 0.0)
                metadata = result.get('metadata', {})
                
                # ファイルパスを取得
                file_path = metadata.get('file_path', metadata.get('source_file', ''))
                
                # ファイルタイプ別表示
                self.display.display_search_result(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    metadata=metadata,
                    file_path=file_path,
                    summary=result.get('summary')
                )
        
        # ページネーション
        if len(results) > max_results:
            st.info(f"他に {len(results) - max_results}件の結果があります")
    
    def display_file_type_summary(self, results: List[Dict[str, Any]]) -> None:
        """ファイルタイプ別サマリー表示"""
        if not results:
            return
        
        # ファイルタイプ集計
        type_counts = {}
        for result in results:
            metadata = result.get('metadata', {})
            file_path = metadata.get('file_path', metadata.get('source_file', ''))
            file_type = self.display.detect_file_type(file_path, metadata)
            type_counts[file_type.value] = type_counts.get(file_type.value, 0) + 1
        
        # サマリー表示
        st.markdown("### ファイルタイプ別サマリー")
        cols = st.columns(len(type_counts))
        
        for i, (file_type, count) in enumerate(type_counts.items()):
            with cols[i]:
                st.metric(
                    label=f"{file_type.upper()}",
                    value=count,
                    help=f"{file_type}ファイルの検索結果数"
                )

def create_display_config(
    show_thumbnail: bool = True,
    show_preview: bool = True,
    max_preview_size: int = 500,
    thumbnail_size: Tuple[int, int] = (200, 150),
    enable_interactive: bool = True
) -> DisplayConfig:
    """表示設定を作成する便利関数"""
    return DisplayConfig(
        show_thumbnail=show_thumbnail,
        show_preview=show_preview,
        max_preview_size=max_preview_size,
        thumbnail_size=thumbnail_size,
        enable_interactive=enable_interactive
    )








