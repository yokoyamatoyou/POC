import xml.etree.ElementTree as ET

from typing import List, Dict

class XMLProcessor:
    """
    XMLProcessor parses XML files (including XSD and DTD) and generates text chunks with metadata.
    """
    def process(self, file_path: str) -> List[Dict]:
        tree = ET.parse(file_path)
        root = tree.getroot()
        chunks: List[Dict] = []
        for elem in root.iter():
            text = (elem.text or '').strip()
            if text:
                namespace = None
                if '}' in elem.tag:
                    namespace = elem.tag.split('}')[0].strip('{')
                metadata = {
                    'tag': elem.tag,
                    'attributes': elem.attrib,
                    'namespace': namespace,
                }
                chunks.append({'text': text, 'metadata': metadata})
        return chunks


