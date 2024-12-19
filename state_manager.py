from pathlib import Path
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class JobMetadata:
    input_file: str
    from_lang: str
    to_lang: str
    model: str
    timestamp: str
    chapter_map: Dict[str, Tuple[str, int]]

class StateManager:
    def __init__(self, job_id: str, temp_dir: Path):
        self.job_id = job_id
        self.job_dir = temp_dir / job_id
        self.job_dir.mkdir(exist_ok=True)
        self.paths = {
            'state_file': self.job_dir / 'job_state.json',
            'chunks_file': self.job_dir / 'chunks.json',
            'translations_file': self.job_dir / 'translations.json',
            'progress_log': self.job_dir / 'progress.log'
        }
        self._state_cache = None
        self._translations_cache = None

    def _atomic_write(self, path: Path, data: dict) -> None:
        """Write data atomically using a temporary file"""
        temp_path = path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def save_state(self, chunks_total: int, chunks_completed: int) -> None:
        """Save current job state with atomic write"""
        state = {
            'chunks_total': chunks_total,
            'chunks_completed': chunks_completed,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        self._atomic_write(self.paths['state_file'], state)
        self._state_cache = state

    def save_translations(self, translations: Dict[str, str]) -> None:
        """Save translations with atomic write"""
        self._atomic_write(self.paths['translations_file'], translations)
        self._translations_cache = translations

    def save_chunks(self, chunks: List[Tuple[str, str]], chapter_map: Dict[str, Tuple[str, int]]) -> None:
        """Save chunks and chapter mapping with atomic write"""
        chunks_data = {
            'chunks': chunks,
            'chapter_map': {
                id: {'item': str(item), 'pos': pos}
                for id, (item, pos) in chapter_map.items()
            }
        }
        self._atomic_write(self.paths['chunks_file'], chunks_data)

    def load_state(self) -> Optional[dict]:
        """Load state with caching"""
        if self._state_cache is not None:
            return self._state_cache
        
        try:
            with open(self.paths['state_file'], 'r', encoding='utf-8') as f:
                self._state_cache = json.load(f)
                return self._state_cache
        except FileNotFoundError:
            return None

    def load_translations(self) -> Dict[str, str]:
        """Load translations with caching"""
        if self._translations_cache is not None:
            return self._translations_cache
        
        try:
            with open(self.paths['translations_file'], 'r', encoding='utf-8') as f:
                self._translations_cache = json.load(f)
                return self._translations_cache
        except FileNotFoundError:
            return {}

    def load_chunks(self) -> Tuple[Optional[List[Tuple[str, str]]], Optional[Dict[str, Tuple[str, int]]]]:
        """Load chunks and chapter mapping"""
        try:
            with open(self.paths['chunks_file'], 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks = [(id, text) for id, text in data['chunks']]
                chapter_map = {
                    chunk_id: (data['item'], data['pos'])
                    for chunk_id, data in data['chapter_map'].items()
                }
                return chunks, chapter_map
        except FileNotFoundError:
            return None, None

    def log_progress(self, message: str) -> None:
        """Log progress with timestamp"""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        try:
            with open(self.paths['progress_log'], 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"Warning: Could not write to progress log: {e}")
            print(f"Progress: {message}")

    def clear_cache(self) -> None:
        """Clear internal caches"""
        self._state_cache = None
        self._translations_cache = None
