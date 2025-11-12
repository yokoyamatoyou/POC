"""
CodeProcessor: delegate to language-specific parsers for various code file types.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Import language-specific parsers
try:
    from .code_parsers.python_parser import process_python_file
    PYTHON_PARSER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Python parser import failed: {e}")
    PYTHON_PARSER_AVAILABLE = False
try:
    from .code_parsers.js_ts_parser import process_js_ts_file
    JS_TS_PARSER_AVAILABLE = True
except ImportError:
    JS_TS_PARSER_AVAILABLE = False
try:
    from .code_parsers.java_parser import process_java_file
    JAVA_PARSER_AVAILABLE = True
except ImportError:
    JAVA_PARSER_AVAILABLE = False
try:
    from .code_parsers.c_cpp_parser import process_c_cpp_file
    C_CPP_PARSER_AVAILABLE = True
except ImportError:
    C_CPP_PARSER_AVAILABLE = False


def process_code_file(file_path: str, chunk_size: int = 100) -> Tuple[List[Dict[str, Any]], Optional[None]]:
    """
    Process a code file by delegating to the appropriate parser based on extension.
    """
    ext = Path(file_path).suffix.lower()
    chunks: List[Dict[str, Any]] = []
    try:
        if ext == '.py' and PYTHON_PARSER_AVAILABLE:
            chunks, _ = process_python_file(file_path, chunk_size)
        elif ext in ('.js', '.ts') and JS_TS_PARSER_AVAILABLE:
            chunks, _ = process_js_ts_file(file_path, chunk_size)
        elif ext == '.java' and JAVA_PARSER_AVAILABLE:
            chunks, _ = process_java_file(file_path, chunk_size)
        elif ext in ('.c', '.cpp', '.h', '.hpp') and C_CPP_PARSER_AVAILABLE:
            chunks, _ = process_c_cpp_file(file_path, chunk_size)
        else:
            logger.warning(f"No parser available for extension {ext}, returning full file chunk")
            source = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            chunks = [{
                'text': source,
                'metadata': {
                    'source_file': Path(file_path).name,
                    'type': 'code',
                    'language': ext.lstrip('.') or 'unknown',
                    'chunk_type': 'full_file'
                }
            }]
        return chunks, None
    except Exception as e:
        logger.error(f"Code processing error: {e}")
        return [], None


