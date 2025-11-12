"""
Python parser for Phase 6: extract functions and classes as chunks.
"""

import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def process_python_file(file_path: str, chunk_size: int = 100) -> Tuple[List[Dict[str, Any]], Optional[None]]:
    """
    Parse a Python file, extract function and class definitions into chunks.
    """
    try:
        source = Path(file_path).read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(source)
        lines = source.splitlines()
        chunks: List[Dict[str, Any]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', start + 1)
                segment = "\n".join(lines[start:end])
                chunk_type = 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class'
                metadata: Dict[str, Any] = {
                    'source_file': Path(file_path).name,
                    'type': 'code',
                    'language': 'python',
                    'chunk_type': chunk_type,
                    'name': node.name,
                    'start_line': start + 1,
                    'end_line': end
                }
                chunks.append({'text': segment, 'metadata': metadata})
        if not chunks:
            # fallback to full file
            metadata = {
                'source_file': Path(file_path).name,
                'type': 'code',
                'language': 'python',
                'chunk_type': 'full_file'
            }
            chunks.append({'text': source, 'metadata': metadata})
        return chunks, None
    except Exception as e:
        logger.error(f"Python parsing error: {e}")
        return [], None


