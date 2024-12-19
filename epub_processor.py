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
        paragraphs = content.split('</p>')
        # Remove empty paragraphs and add closing tags back
        paragraphs = [p.strip() + '</p>' for p in paragraphs if p.strip()]
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    # This paragraph is too big - split by sentences
                    sentences = self._split_sentences(paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) > max_chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = sentence
                        else:
                            temp_chunk += ' ' if temp_chunk else ''
                            temp_chunk += sentence
                    
                    if temp_chunk:
                        chunks.append(temp_chunk)
                    current_chunk = ""
            else:
                current_chunk += ' ' if current_chunk else ''
                current_chunk += paragraph
        
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Helper method to split text into sentences while preserving special cases"""
        abbreviations = r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|vs|e\.g|i\.e|viz|cf|Ch|p|pp|vol|ex|no)\.'
        numbers = r'\d+'
        no_split_pattern = f'(?:{abbreviations}|{numbers})'
        sentences = re.split(f'(?<!{no_split_pattern})\\. +(?=[A-Z])', text)
        return [s + '.' for s in sentences[:-1]] + [sentences[-1]]

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
        with zipfile.ZipFile(self.input_path, 'r') as zip_ref:
            with zipfile.ZipFile(self.output_path, 'w') as out_zip:
                for item in zip_ref.infolist():
                    if item.filename.endswith(('.html', '.xhtml')):
                        with zip_ref.open(item.filename) as file:
                            content = file.read().decode('utf-8')
                        chunks = self.split_content(content)
                        translated_chunks = []
                        for chunk in chunks:
                            chunk_id = [k for k, v in chapter_map.items() if v[0] == item.filename and v[1] == chunks.index(chunk)]
                            if chunk_id:
                                translated_chunks.append(translations.get(chunk_id[0], chunk))
                            else:
                                translated_chunks.append(chunk)
                        new_content = ''.join(translated_chunks)
                        out_zip.writestr(item, new_content.encode('utf-8'))
                    else:
                        out_zip.writestr(item, zip_ref.read(item.filename))