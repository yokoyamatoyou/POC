"""
CSV/TSV processor for Phase 6.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def process_csv_file(file_path: str, chunk_size: int = 100) -> Tuple[List[Dict[str, Any]], Optional[None]]:
    """
    Read CSV or TSV file, detect header, infer data types, chunk rows, and generate metadata.
    """
    try:
        # Determine delimiter by extension
        delimiter = ',' if file_path.lower().endswith('.csv') else '\t'
        # Read all data as string for safe processing
        df = pd.read_csv(file_path, delimiter=delimiter, dtype=str)
        num_rows = df.shape[0]
        # Infer data types for each column
        dtypes = df.infer_objects().dtypes.apply(lambda dt: dt.name).to_dict()
        chunks: List[Dict[str, Any]] = []
        # Chunk rows into blocks
        for start in range(0, num_rows, chunk_size):
            sub_df = df.iloc[start:start+chunk_size]
            # Convert chunk back to text
            text = sub_df.to_csv(index=False, sep=delimiter)
            metadata: Dict[str, Any] = {
                "source_file": Path(file_path).name,
                "type": "csv",
                "delimiter": delimiter,
                "columns": list(df.columns),
                "dtypes": dtypes,
                "row_offset": start,
                "num_rows": num_rows,
                "chunk_type": "table"
            }
            chunks.append({"text": text, "metadata": metadata})
        return chunks, None
    except Exception as e:
        logger.error(f"CSV processing error: {e}")
        return [], None


