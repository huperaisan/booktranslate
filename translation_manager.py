import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from state_manager import StateManager
from epub_processor import EPubProcessor

class TranslationManager:
    def __init__(self, client: OpenAI, state_manager: StateManager, epub_processor: EPubProcessor):
        self.client = client
        self.state_manager = state_manager
        self.epub_processor = epub_processor

    async def process_chunk(self, chunk_id: str, text: str, from_lang: str, to_lang: str, model: str) -> Optional[str]:
        """Process a single chunk with retries and error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            'role': 'system',
                            'content': self._get_system_prompt(from_lang, to_lang)
                        },
                        {
                            'role': 'user',
                            'content': text
                        }
                    ],
                    temperature=0.2
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    self.state_manager.log_progress(f"Failed to translate chunk {chunk_id} after {max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def _get_system_prompt(self, from_lang: str, to_lang: str) -> str:
        """Get system prompt for translation"""
        return f"""You are a {from_lang}-to-{to_lang} translator.

CRITICAL: You must preserve ALL HTML/XML structure exactly as provided:
- Never remove or modify HTML/XML tags
- Keep ALL class names and IDs unchanged
- Preserve ALL style attributes completely 
- Maintain ALL paragraph (<p>) and div tags with their full attributes
- Copy opening and closing tags exactly as they appear
- Do not merge or split HTML/XML elements
- Do not add new HTML/XML formatting

Translate ONLY the text that is in {from_lang} between tags.
Leave in place WITHOUT translating or modifying in any way:
- Greek/Latin words or text, including block quotations
- HTML/XML attributes and values"""

    async def process_batch(self, chunks: List[Tuple[str, str]], from_lang: str, to_lang: str, model: str) -> Dict[str, str]:
        """Process chunks in batches with parallel execution"""
        batch_size = 10  # Configurable batch size
        translations = {}
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            tasks = [
                self.process_chunk(chunk_id, text, from_lang, to_lang, model)
                for chunk_id, text in batch
            ]
            results = await asyncio.gather(*tasks)
            
            for (chunk_id, _), result in zip(batch, results):
                if result:
                    translations[chunk_id] = result
                    self.state_manager.save_translations(translations)
            
            self.state_manager.log_progress(f"Processed {len(translations)}/{len(chunks)} chunks")
        
        return translations
