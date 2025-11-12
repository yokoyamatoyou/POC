"""
JSON/YAML processor for Phase 6.
"""

import json
import yaml
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def process_json_yaml_file(file_path: str, chunk_size: int = 100) -> Tuple[List[Dict[str, Any]], Optional[None]]:
    """
    Read JSON or YAML file, parse, pretty-print, chunk lines, and generate metadata.
    """
    try:
        ext = Path(file_path).suffix.lower()
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                data = json.load(f)
            elif ext in ('.yaml', '.yml'):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported extension: {ext}")
        # Pretty-print content as string
        if ext == '.json':
            content = json.dumps(data, indent=2, ensure_ascii=False)
            chunk_type = 'json'
        else:
            content = yaml.dump(data, allow_unicode=True)
            chunk_type = 'yaml'
        # Split into lines and chunk
        lines = content.splitlines()
        total_lines = len(lines)
        chunks: List[Dict[str, Any]] = []
        for start in range(0, total_lines, chunk_size):
            block = lines[start:start+chunk_size]
            text = "\n".join(block)
            metadata: Dict[str, Any] = {
                'source_file': Path(file_path).name,
                'type': ext.lstrip('.'),
                'chunk_type': chunk_type,
                'line_offset': start,
                'total_lines': total_lines
            }
            chunks.append({'text': text, 'metadata': metadata})
        return chunks, None
    except Exception as e:
        logger.error(f"JSON/YAML processing error: {e}")
        return [], None


