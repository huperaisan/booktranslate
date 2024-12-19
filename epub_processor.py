
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile
from io import BytesIO
import re

class EPubProcessor:
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path

    def split_content(self, content: str, max_chunk_size: int = 10000) -> List[str]:
        """Split HTML content into manageable chunks"""
        # ... existing split_html_by_paragraph code ...

    def extract_chunks(self) -> Tuple[List[Tuple[str, str]], Dict[str, Tuple[str, int]]]:
        """Extract chunks from EPUB file"""
        chunks = []
        chapter_map = {}
        chunk_counter = 0

        with zipfile.ZipFile(self.input_path, 'r') as zip_ref:
            html_files = [f for f in zip_ref.namelist() 
                         if f.endswith(('.html', '.xhtml'))]

            for html_file in html_files:
                with zip_ref.open(html_file) as file:
                    content = file.read().decode('utf-8')

                text_chunks = self.split_content(content)
                
                for pos, chunk in enumerate(text_chunks):
                    chunk_id = f'chunk-{chunk_counter}'
                    chunks.append((chunk_id, chunk))
                    chapter_map[chunk_id] = (html_file, pos)
                    chunk_counter += 1

        return chunks, chapter_map

    def reassemble(self, translations: Dict[str, str], chapter_map: Dict[str, Tuple[str, int]]) -> None:
        """Reassemble translated content into new EPUB"""
        # ... existing reassemble_translation code ...