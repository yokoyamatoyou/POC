"""
C/C++ parser stub: return full file as single chunk.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def process_c_cpp_file(file_path: str, chunk_size: int = 100) -> Tuple[List[Dict[str, Any]], Optional[None]]:
    try:
        source = Path(file_path).read_text(encoding='utf-8', errors='ignore')
        metadata = {
            'source_file': Path(file_path).name,
            'type': 'code',
            'language': 'c/cpp',
            'chunk_type': 'full_file'
        }
        return [{'text': source, 'metadata': metadata}], None
    except Exception as e:
        logger.error(f"C/C++ parsing error: {e}")
        return [], None


