import zipfile
import tarfile
from typing import List, Dict

# Optional libraries for RAR and 7z
try:
    import rarfile
    RARFILE_AVAILABLE = True
except Exception:
    rarfile = None
    RARFILE_AVAILABLE = False

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except Exception:
    py7zr = None
    PY7ZR_AVAILABLE = False

class ArchiveProcessor:
    """
    ArchiveProcessor extracts ZIP, TAR, RAR, 7z archives and lists member files as chunks.
    """
    def process(self, file_path: str) -> List[Dict]:
        chunks: List[Dict] = []
        # ZIPファイル処理
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path) as zf:
                for name in zf.namelist():
                    chunks.append({'text': name, 'metadata': {'archive': file_path, 'member': name}})
        # TAR系ファイル処理
        elif tarfile.is_tarfile(file_path):
            with tarfile.open(file_path) as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        chunks.append({'text': member.name, 'metadata': {'archive': file_path, 'member': member.name}})
        else:
            # Check for RAR
            lower = file_path.lower()
            if lower.endswith('.rar'):
                if not RARFILE_AVAILABLE:
                    raise RuntimeError('RAR support not available: install `rarfile` and system unrar/unar')
                rf = rarfile.RarFile(file_path)
                for member in rf.infolist():
                    if not member.isdir():
                        chunks.append({'text': member.filename, 'metadata': {'archive': file_path, 'member': member.filename}})
            elif lower.endswith('.7z'):
                if not PY7ZR_AVAILABLE:
                    raise RuntimeError('7z support not available: install `py7zr`')
                with py7zr.SevenZipFile(file_path, mode='r') as zf:
                    for name in zf.getnames():
                        chunks.append({'text': name, 'metadata': {'archive': file_path, 'member': name}})
            else:
                raise ValueError(f"Unsupported archive format: {file_path}")
        return chunks

