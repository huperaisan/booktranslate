#v0.9: Flags: language settings (--from-lang DE --to-lang EN), processing --mode (--batch, --savebatch, --resume, --test), model selection (--model gpt-4o-mini), save temp files (--debug). Can resume expired or cancelled batch jobs. All jobs can be resumed in either fast or batch mode.

import argparse
import re
import yaml
import os
from openai import OpenAI
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import time  # Add this import at the top with other imports
import asyncio

# Add UTC constant
UTC = timezone.utc

# Define global variable for keeping temp files
KEEP_TEMP = False  # Defaults to False

class BatchProcessingError(Exception):
    """Custom exception for batch processing errors"""
    pass

def ensure_dir(name):
    """Create directory relative to script location"""
    dir_path = Path(__file__).parent / name
    dir_path.mkdir(exist_ok=True)
    return dir_path

def ensure_temp_dir():
    """Create temp directory if it doesn't exist"""
    return ensure_dir("temp")

def ensure_batch_dir():
    """Create batch directory if it doesn't exist"""
    return ensure_dir("batch")

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    return ensure_dir("output")

def chmod_recursive(path):
    """Recursively change permissions"""
    try:
        os.chmod(path, 0o777)  # Full permissions for owner, group, others
        
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o777)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o777)
    except Exception as e:
        print(f"Warning: Failed to change permissions for {path}: {e} [chmod_recursive]")

def cleanup_files(client, file_ids, temp_dir=None, keep_temp=False, keep_batch_files=False):
    """Clean up temporary files and directories with error handling"""
    # First print the keep_temp status
    print(f"Cleanup called with keep_temp={keep_temp} for {temp_dir} [cleanup_files]")
    
    # Exit early if we should keep temp files
    if keep_temp:
        print(f"Keeping temp directory: {temp_dir} [cleanup_files]")
        return

    if not keep_temp and temp_dir and temp_dir.exists():
        try:
            print(f"\nStarting cleanup of: {temp_dir} [cleanup_files]")
            
            # Force permissions on the directory and its contents
            chmod_recursive(temp_dir)
            
            # First, forcefully close and remove all files
            for item in temp_dir.glob('**/*'):
                if item.is_file():
                    try:
                        # Try to force close any open handles to the file
                        with open(item, 'r') as f:
                            try:
                                os.fsync(f.fileno())
                            except:
                                pass
                        
                        if keep_batch_files and (item.name.startswith("batch_input") or 
                                               item.name.startswith("batch_output")):
                            continue
                            
                        item.unlink(missing_ok=True)
                        print(f"Removed file: {item} [cleanup_files]")
                    except Exception as e:
                        print(f"Warning: Could not remove file {item}: {e} [cleanup_files]")
                        try:
                            # Last resort: use os.remove
                            os.remove(item)
                            print(f"Removed file using os.remove: {item} [cleanup_files]")
                        except Exception as e2:
                            print(f"Failed to remove file even with os.remove: {e2} [cleanup_files]")
            
            # Then remove all directories bottom-up
            for item in sorted(temp_dir.glob('**/*'), reverse=True):
                if item.is_dir():
                    try:
                        item.rmdir()
                        print(f"Removed directory: {item} [cleanup_files]")
                    except Exception as e:
                        print(f"Warning: Could not remove directory {item}: {e} [cleanup_files]")
                        try:
                            # Last resort: use shutil.rmtree
                            shutil.rmtree(item, ignore_errors=True)
                            print(f"Removed directory using rmtree: {item} [cleanup_files]")
                        except Exception as e2:
                            print(f"Failed to remove directory even with rmtree: {e2} [cleanup_files]")
            
            # Finally try to remove the temp directory itself
            try:
                temp_dir.rmdir()
                print(f"Removed temp directory: {temp_dir} [cleanup_files]")
            except Exception as e:
                print(f"Could not remove temp directory {temp_dir}: {e} [cleanup_files]")
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"Removed temp directory using rmtree: {temp_dir} [cleanup_files]")
                except Exception as e2:
                    print(f"Failed to remove temp directory even with rmtree: {e2} [cleanup_files]")
                
        except Exception as e:
            print(f"Warning: Error during cleanup: {e} [cleanup_files]")
            
    # Clean up OpenAI files
    for file_id in file_ids:
        try:
            client.files.delete(file_id)
            print(f"Deleted OpenAI file: {file_id} [cleanup_files]")
        except Exception as e:
            print(f"Warning: Failed to delete OpenAI file {file_id}: {e} [cleanup_files]")

def read_config():
    config_file = 'config.yaml'
    try:
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except FileNotFoundError:
        print(f"Error: {config_file} not found. Please create it with your OpenAI API key. [read_config]")
        print("Example config.yaml content:\nopenai:\n  api_key: 'your-api-key-here' [read_config]")
        exit(1)

def split_sentences(text):
    """Split text into sentences while preserving common abbreviations and numerical citations"""
    # Common abbreviations to preserve
    abbreviations = r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|vs|e\.g|i\.e|viz|cf|Ch|p|pp|vol|ex|no)\.'
    # Add numerical citations pattern
    numbers = r'\d+'
    # Combine patterns - match either abbreviations or number.number pattern
    no_split_pattern = f'(?:{abbreviations}|{numbers})'
    # Split on period followed by space and capital letter, 
    # but not if preceded by abbreviations or numbers
    sentences = re.split(f'(?<!{no_split_pattern})\\. +(?=[A-Z])', text)
    return [s + '.' for s in sentences[:-1]] + [sentences[-1]]  # Add periods back except for last sentence

def split_html_by_paragraph(html_str, max_chunk_size=10000):
    """Split HTML content by paragraphs, falling back to sentences only when necessary"""
    paragraphs = html_str.split('</p>')
    # Remove empty paragraphs and add closing tags back
    paragraphs = [p.strip() + '</p>' for p in paragraphs if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph would exceed max size
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            if current_chunk:
                # Finalize current chunk, start new with this paragraph
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # This paragraph is first in chunk AND too big - split by sentences
                sentences = split_sentences(paragraph)
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
            # Paragraph fits, add it
            current_chunk += ' ' if current_chunk else ''  # Changed from '\n\n' to ' '
            current_chunk += paragraph
    
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def create_job_id(input_epub_path, from_lang, to_lang, model, timestamp=None):
    """Create unique job ID based on input parameters and timestamp"""
    if timestamp is None:
        timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    base_name = Path(input_epub_path).stem
    return f"{base_name}_{from_lang}_{to_lang}_{model}_{timestamp}"

def ensure_temp_structure(job_id):
    """Create and return paths for all temporary files"""
    temp_dir = ensure_temp_dir()
    job_dir = temp_dir / job_id
    job_dir.mkdir(exist_ok=True)
    
    paths = {
        'job_dir': job_dir,
        'state_file': job_dir / 'job_state.json',
        'chunks_file': job_dir / 'chunks.json',  # Store all original chunks
        'translations_file': job_dir / 'translations.json',  # Store completed translations
        'progress_log': job_dir / 'progress.log'  # Log translation progress
    }
    return paths

def save_job_state(paths, chunks_total, chunks_completed, translations):
    """Save current job state and translations"""
    state = {
        'chunks_total': chunks_total,
        'chunks_completed': chunks_completed,
        'last_updated': datetime.now(UTC).isoformat()
    }
    
    try:
        # Save state with fsync for durability
        with open(paths['state_file'], 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # Save translations separately with fsync
        with open(paths['translations_file'], 'w', encoding='utf-8') as f:
            json.dump(translations, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            
    except Exception as e:
        print(f"Warning: Could not save state: {e} [save_job_state]")
        # Continue processing but warn user
        print("Warning: Progress may not be resumable if the script is interrupted [save_job_state]")

def load_job_state(paths):
    """Load existing state and translations, making job_state.json optional"""
    state = {}
    try:
        # Load chunks.json (required)
        print(f"Loading chunks from: {paths['chunks_file']} [load_job_state]")
        if not paths['chunks_file'].exists():
            print(f"No chunks file found at: {paths['chunks_file']} [load_job_state]")
            return None
            
        with open(paths['chunks_file'], 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            # Explicitly reconstruct chunks as list of tuples (id, text)
            state['chunks'] = [(id, text) for id, text in chunks_data['chunks']]
            # Properly reconstruct chapter_map from the stored format
            state['chapter_map'] = {
                chunk_id: (data['item'], data['pos']) 
                for chunk_id, data in chunks_data['chapter_map'].items()
            }

        # Add total chunks count
        state['chunks_total'] = len(state['chunks'])
            
        # Load translations.json (optional)
        print(f"Loading translations from: {paths['translations_file']} [load_job_state]")
        if paths['translations_file'].exists():
            with open(paths['translations_file'], 'r', encoding='utf-8') as f:
                state['translations'] = json.load(f)
                state['chunks_completed'] = len(state['translations'])
        else:
            state['translations'] = {}
            state['chunks_completed'] = 0

        # Extract timestamp from job_id directory name if exists
        if paths['job_dir'].exists():
            timestamp = paths['job_dir'].name.split('_')[-1]
            state['last_updated'] = f"{timestamp[:8]}T{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}.000000+00:00"
                
    except Exception as e:
        print(f"Warning: Could not load state files: {e} [load_job_state]")
        print(f"Attempted to load from paths: [load_job_state]")
        for key, path in paths.items():
            print(f"  {key}: {path} (exists: {path.exists()}) [load_job_state]")
        return None
        
    return state

def save_chunks(paths, all_chunks, chapter_map):
    """Save initial chunks and chapter mapping"""
    chunks_data = {
        'chunks': [(id, text) for id, text in all_chunks],
        'chapter_map': {id: {'item': str(item), 'pos': pos} for id, (item, pos) in chapter_map.items()}
    }
    with open(paths['chunks_file'], 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2)

def log_progress(paths, message):
    """Log progress with timestamp"""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        with open(paths['progress_log'], 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Warning: Could not write to progress log: {e} [log_progress]")
        print(f"Progress: {message} [log_progress]")

def system_prompt(from_lang, to_lang):
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
- HTML/XML attributes and values

Bad example:
Input: <p class="block_36" style="font-size: 1.33333em">Der Text...</p>
Output: The text...

Good example:
Input: <p class="block_36" style="font-size: 1.33333em">Der Text...</p>
Output: <p class="block_36" style="font-size: 1.33333em">The text...</p>

As a SECOND step: find numbers which are part of the CONTENT and not part of HTML code. If the number DIRECTLY follows text (letters not numerals) or the punctuation marks .,!?"' without a space, then enclose the number in superscript (<sup></sup>) tags, WITHOUT adding a space before the opening <sup> tag.

Good examples:
Input: Elysian traits seen.10 
Output: Elysian traits seen.<sup>10</sup>

Input: disinterest in it.7</p>
Output: disinterest in it.<sup>7</sup></p>

Input: joy to man.5 
Output: joy to man.<sup>5</sup>

Input: incessantly (82-84).6
Output: incessantly (82-84).<sup>6</sup>

Do not touch [number].[number] or Od.[number] or Il.[number] e.g.: 5.59, 12.52, Od.5, Il.12
"""

def load_test_translations(test_file):
    """Load test translations from a JSON file with flat chunk IDs"""
    test_path = Path("test") / Path(test_file).stem / "translations.json"
    
    if not test_path.exists():
        print(f"Error: Test translations file not found: {test_path} [load_test_translations]")
        return None
    
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            normalized_test_translations = json.load(f)
        
        if not isinstance(normalized_test_translations, dict):
            print("Warning: Test translations should be a dictionary of chunk_id to translation. [load_test_translations]")
            return {}
        
        print(f"Loaded {len(normalized_test_translations)} test translations [load_test_translations]")
        return normalized_test_translations
    except Exception as e:
        print(f"Warning: Could not load test translations from {test_path}: {e} [load_test_translations]")
        return {}

def save_batch_state(temp_dir, batch_id, input_file_id, timestamp, job_metadata, paths):
    """Save batch processing state for later checking"""
    state_file = temp_dir / f"batch_status_{timestamp}.json"
    state = {
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "timestamp": timestamp,
        "job_metadata": job_metadata,
        "paths": {k: str(v) for k, v in paths.items()}
    }
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    return state_file

def load_batch_state(temp_dir):
    """Load most recent batch state file"""
    state_files = list(temp_dir.glob("batch_status_*.json"))
    if not state_files:
        return None
    latest_file = max(state_files, key=lambda f: f.stat().st_mtime)
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f), latest_file

def batch_translate_chunks(client, chunks, from_lang, to_lang, mode=None, model='gpt-4o-mini', 
                         test_translations=None, keep_temp=False, paths=None, chapter_map=None):
    """Handle translation of all chunks in a single batch"""
    # Remove batchcheck handling from here since it's now in process_translations
    if mode == 'batchcheck':
        return {}, None, None  # Let process_translations handle batchcheck mode

    # Extract input_epub_path from the job directory name
    if paths:
        input_epub_path = paths['job_dir'].parent.parent / paths['job_dir'].name.split('_')[0]
    else:
        input_epub_path = "unknown_input"  # Fallback value if paths not provided

    temp_dir = ensure_temp_dir()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    import select
    import sys
    
    # Create batch input file
    batch_file_path = temp_dir / f"batch_input_{timestamp}.jsonl"
    with open(batch_file_path, "w", encoding="utf-8") as f:
        for chunk_id, chunk_text in chunks:
            request = {
                "custom_id": chunk_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt(from_lang, to_lang)
                        },
                        {
                            "role": "user",
                            "content": chunk_text
                        }
                    ],
                    "temperature": 0.2
                }
            }
            f.write(json.dumps(request) + "\n")
    
    if mode == 'savebatch':
        # Copy input file to batch directory
        batch_dir = ensure_batch_dir()
        batch_input_path = batch_dir / f"batch_input_{timestamp}.jsonl"
        Path(batch_file_path).replace(batch_input_path)
        print(f"Saved batch input file: {batch_input_path} [batch_translate_chunks]")
        
        # Save batch configuration
        config_path = batch_dir / f"batch_config_{timestamp}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'from_lang': from_lang,
                'to_lang': to_lang,
                'chunk_count': len(chunks)
            }, f, indent=2)
        print(f"Saved batch configuration: {config_path} [batch_translate_chunks]")
        
    # Process batch
    print("Uploading batch file... [batch_translate_chunks]")
    batch_file = client.files.create(
        file=open(batch_file_path if mode != 'savebatch' else batch_input_path, "rb"),
        purpose="batch"
    )
    input_file_id = batch_file.id
    
    print(f"Creating batch job with file ID: {input_file_id} [batch_translate_chunks]")
    batch_job = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    batch_id = batch_job.id
    print(f"Batch job created with ID: {batch_id} [batch_translate_chunks]")
    
    if mode == 'savebatch':
        # Save batch identifiers
        ids_path = batch_dir / f"batch_ids_{timestamp}.json"
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump({
                'batch_id': batch_id,
                'input_file_id': input_file_id
            }, f, indent=2)
        print(f"Saved batch identifiers: {ids_path} [batch_translate_chunks]")
    
    print("\nBatch processing started. Checking initial status in 5 seconds... [batch_translate_chunks]")
    time.sleep(5)
    
    status = client.batches.retrieve(batch_id)
    print(f"Status: {status.status} - Completed: {status.request_counts.completed}/{status.request_counts.total} [batch_translate_chunks]")
    
    if status.status == "failed":
        print("Batch processing failed. [batch_translate_chunks]")
        return {}, input_file_id, status
        
    # Save state and exit if not failed
    job_metadata = {
        "input_file": str(input_epub_path),
        "from_lang": from_lang,
        "to_lang": to_lang,
        "model": model,
        "chapter_map": {chunk_id: {"item": str(item), "pos": pos} 
                       for chunk_id, (item, pos) in chapter_map.items()} if chapter_map else {}
    }
    
    state_file = save_batch_state(temp_dir, batch_id, input_file_id, timestamp, job_metadata, paths)
    print(f"\nBatch processing in progress. Saved state to: {state_file} [batch_translate_chunks]")
    print("Run script again with --mode batchcheck to check status and retrieve results [batch_translate_chunks]")
    return {}, input_file_id, status

def parse_batch_response(response):
    """Parse batch response into translations dictionary"""
    translations = {}
    for line in response.splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        chunk_id = result['custom_id']
        if 'response' in result and 'body' in result['response']:
            if 'choices' in result['response']['body'] and result['response']['body']['choices']:
                translated_text = result['response']['body']['choices'][0]['message']['content']
                # Remove markdown code block markers if present
                translated_text = re.sub(r'^```[a-zA-Z]*\n', '', translated_text)
                translated_text = re.sub(r'```$', '', translated_text)
                translations[chunk_id] = translated_text
    return translations

def translate_chunk(client, text, chunk_id=None, from_lang='EN', to_lang='PL', model='gpt-4o-mini', test_translations=None):
    """Translate a single chunk, with optional test mode"""
    if test_translations is not None:
        # Directly look up translation by chunk_id
        translation = test_translations.get(chunk_id)
        if translation:
            return translation
        else:
            print(f"Warning: No test translation found for chunk {chunk_id} [translate_chunk]")
            return f"[Translation content for chunk-{chunk_id} would go here]"
    
    # Normal API mode
    system_message = system_prompt(from_lang, to_lang)
    user_message = text
    
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            { 'role': 'system', 'content': system_message },
            { 'role': 'user', 'content': user_message },
        ]
    )
    
    translated_text = response.choices[0].message.content
    # Remove markdown code block markers if present
    translated_text = re.sub(r'^```[a-zA-Z]*\n', '', translated_text)
    translated_text = re.sub(r'```$', '', translated_text)
    return translated_text

def handle_interrupt(signum, frame):
    """Handle keyboard interrupt"""
    print("\nInterrupt received. Saving state and exiting... [handle_interrupt]")
    raise KeyboardInterrupt

def build_chunks(input_epub_path, paths):  # Remove inline_css parameter
    """Extract HTML files from the EPUB and split into chunks without modifying content."""
    import zipfile

    # Open the EPUB archive
    with zipfile.ZipFile(input_epub_path, 'r') as zip_ref:
        # List all files in the EPUB
        file_list = zip_ref.namelist()

        # Filter for HTML/XHTML files
        html_files = [f for f in file_list if f.endswith(('.html', '.xhtml'))]

        all_chunks = []
        chapter_map = {}
        chunk_counter = 0

        for html_file in html_files:
            # Read the raw content of the HTML file
            with zip_ref.open(html_file) as file:
                content = file.read().decode('utf-8')

            print(f"Processing {html_file} [build_chunks]")

            # Split content into chunks (without modifying the content)
            chunks = split_html_by_paragraph(content)
            print(f"Split into {len(chunks)} chunks [build_chunks]")

            for pos, chunk in enumerate(chunks):
                chunk_id = f'chunk-{chunk_counter}'
                # Ensure chunk_id is a string
                chunk_id = str(chunk_id)
                all_chunks.append((chunk_id, chunk))
                chapter_map[chunk_id] = (html_file, pos)
                chunk_counter += 1

    print(f"Total chunks created: {len(all_chunks)} [build_chunks]")

    # Save chunks and chapter map
    save_chunks(paths, all_chunks, chapter_map)

    return all_chunks, chapter_map

def save_translations(paths, translations):
    """Save translations to translations.json with fsync for durability."""
    # Ensure all keys in translations are strings
    translations = {str(k): v for k, v in translations.items()}
    with open(paths['translations_file'], 'w', encoding='utf-8') as f:
        json.dump(translations, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

def process_translations(client, all_chunks, translations, mode, from_lang, to_lang, 
                         paths, model='gpt-4o-mini', test_translations=None, debug=False,
                         chapter_map=None):
    """Handle translation based on the specified mode and update translations."""
    if mode == 'batchcheck':
        # Load saved batch state and check status
        temp_dir = ensure_temp_dir()
        state, state_file = load_batch_state(temp_dir)
        if not state:
            print("No batch state found. Run with --mode batch first. [process_translations]")
            return {}, None, None
        
        batch_id = state['batch_id']
        print(f"Checking status for batch {batch_id} [process_translations]")
        status = client.batches.retrieve(batch_id)
        
        if status.status != "completed":
            print(f"Batch status: {status.status} [process_translations]")
            print(f"Progress: {status.request_counts.completed}/{status.request_counts.total} [process_translations]")
            # Return status instead of exiting, so the main function can handle the messaging
            return {}, state['input_file_id'], status
            
        print("Batch completed! Retrieving results... [process_translations]")
        
        # Load existing translations first
        paths = {k: Path(v) for k, v in state['paths'].items()}
        translations = {}
        if Path(paths['translations_file']).exists():
            print("Loading existing translations... [process_translations]")
            with open(paths['translations_file'], 'r', encoding='utf-8') as f:
                translations = json.load(f)
            print(f"Loaded {len(translations)} existing translations [process_translations]")
        
        # Get and merge new translations
        response = client.files.content(status.output_file_id)
        response_text = response.read().decode('utf-8')
        new_translations = parse_batch_response(response_text)
        print(f"Got {len(new_translations)} new translations from batch [process_translations]")
        
        # Merge and save
        translations.update(new_translations)
        save_translations(paths, translations)
        print(f"Total translations after merge: {len(translations)} [process_translations]")
        
        # Remove state file cleanup from here since translate() will handle it
        return translations, state['input_file_id'], status

    # Ensure translations keys are strings
    translations = {str(k): v for k, v in translations.items()}
    
    # Define untranslated_chunks once for all modes that require it
    untranslated_chunks = [(chunk_id, chunk_text) for chunk_id, chunk_text in all_chunks if str(chunk_id) not in translations]
    
    if mode == 'resumebatch':
        print(f"\nResuming batch translation for {len(untranslated_chunks)} remaining chunks [process_translations]")
        if not untranslated_chunks:
            print("No untranslated chunks found - all translations complete! [process_translations]")
            return translations, None, None
            
        print(f"Creating new batch job for remaining chunks... [process_translations]")
        new_translations, input_file_id, status = batch_translate_chunks(
            client, untranslated_chunks, from_lang, to_lang, mode='batch', model=model,
            test_translations=test_translations, keep_temp=debug, paths=paths,
            chapter_map=chapter_map
        )
        translations.update(new_translations)
        save_translations(paths, translations)
        print(f"Batch job created for remaining chunks. Use --mode batchcheck to monitor progress. [process_translations]")
        return translations, input_file_id, status
        
    if mode in ['fast', 'resume']:
        untranslated = len(untranslated_chunks)
        skipped = len(all_chunks) - untranslated
        
        if skipped > 0:
            print(f"Skipping already translated {skipped} chunks [process_translations]")
        
        print(f"Translating {untranslated} chunks in 'fast' mode [process_translations]")
        for idx, (chunk_id, chunk_text) in enumerate(untranslated_chunks, skipped + 1):
            print(f"Translating chunk #{idx}... [process_translations]")
            try:
                translated_text = translate_chunk(
                    client, chunk_text, chunk_id=chunk_id, from_lang=from_lang, to_lang=to_lang,
                    model=model, test_translations=test_translations
                )
                # Ensure chunk_id is a string when saving translation
                translations[str(chunk_id)] = translated_text
                save_translations(paths, translations)
            except Exception as e:
                print(f"Error translating chunk {chunk_id}: {e} [process_translations]")
                raise
        print(f"Total translations after 'fast' mode: {len(translations)} [process_translations]")  # Added logging
        return translations, None, None  # No cleanup needed in fast mode
    elif mode == 'batch':
        print(f"Translating {len(untranslated_chunks)} chunks in 'batch' mode [process_translations]")
        new_translations, input_file_id, status = batch_translate_chunks(
            client, untranslated_chunks, from_lang, to_lang, mode=mode, model=model,
            test_translations=test_translations, keep_temp=debug, paths=paths,
            chapter_map=chapter_map  # Pass chapter_map to batch function
        )
        translations.update(new_translations)
        save_translations(paths, translations)
        print(f"Total translations after 'batch' mode: {len(translations)} [process_translations]")  # Added logging
        return translations, input_file_id, status
    elif mode == 'test':
        if test_translations:
            for chunk_id, _ in untranslated_chunks:
                if chunk_id in test_translations:
                    translations[str(chunk_id)] = test_translations[chunk_id]
                else:
                    print(f"Warning: No test translation found for chunk {chunk_id} [process_translations]")
                    translations[str(chunk_id)] = f"[TEST MODE] No translation for chunk: {chunk_id}"
        else:
            print("Error: test_translations not provided [process_translations]")
        print(f"Total translations after 'test' mode: {len(translations)} [process_translations]")  # Added logging
        return translations, None, None
    else:
        print(f"Unknown mode: {mode} [process_translations]")
        return translations, None, None  # Or handle appropriately

def reassemble_translation(input_epub_path, output_epub_path, chapter_map, translations):
    """Reassemble the translated HTML files and create a new EPUB."""
    import zipfile
    from io import BytesIO

    # Convert chapter_map to use filenames instead of ebooklib items
    filename_map = {}
    for chunk_id, (item, pos) in chapter_map.items():
        filename = item if isinstance(item, str) else item.file_name
        filename_map[chunk_id] = (filename, pos)

    # Create a new EPUB file
    with zipfile.ZipFile(input_epub_path, 'r') as zip_in:
        with zipfile.ZipFile(output_epub_path, 'w') as zip_out:
            # Group chunks by filename
            file_chunks = {}
            for chunk_id, (filename, pos) in filename_map.items():
                if filename not in file_chunks:
                    file_chunks[filename] = {}
                if chunk_id in translations:
                    file_chunks[filename][pos] = translations[chunk_id]

            # Process all files from the original EPUB
            for item in zip_in.infolist():
                if item.filename in file_chunks:
                    # This is a file that needs translation
                    print(f"Processing translations for {item.filename} [reassemble_translation]")
                    content = zip_in.read(item.filename).decode('utf-8')
                    chunks = split_html_by_paragraph(content)
                    
                    # Replace chunks with translations
                    for pos, chunk in file_chunks[item.filename].items():
                        if pos < len(chunks):  # Ensure position is valid
                            chunks[pos] = chunk
                    
                    # Reassemble the content
                    translated_content = ''.join(chunks)
                    
                    # Create a new ZipInfo to avoid duplicate warnings
                    new_info = zipfile.ZipInfo(filename=item.filename, 
                                             date_time=item.date_time)
                    new_info.compress_type = item.compress_type
                    
                    # Write the translated content
                    zip_out.writestr(new_info, translated_content.encode('utf-8'))
                else:
                    # Copy unchanged files
                    buffer = BytesIO(zip_in.read(item.filename))
                    new_info = zipfile.ZipInfo(filename=item.filename,
                                             date_time=item.date_time)
                    new_info.compress_type = item.compress_type
                    zip_out.writestr(new_info, buffer.getvalue())

    print(f"Processed {len(file_chunks)} files with translations [reassemble_translation]")

def check_batch_status(client):
    """Check status of most recent batch job and return state if complete"""
    temp_dir = ensure_temp_dir()
    state, state_file = load_batch_state(temp_dir)
    
    if not state:
        print("No batch state found. Run with --mode batch first. [check_batch_status]")
        return None, None, None
    
    batch_id = state['batch_id']
    print(f"Checking status for batch {batch_id} [check_batch_status]")
    status = client.batches.retrieve(batch_id)
    
    print(f"Status: {status.status} [check_batch_status]")
    print(f"Progress: {status.request_counts.completed}/{status.request_counts.total} [check_batch_status]")
    
    # Handle expired or cancelled batches with partial results
    if status.status in ['expired', 'cancelled'] and status.request_counts.completed > 0:
        print(f"\nBatch {status.status} but has {status.request_counts.completed} completed requests [check_batch_status]")
        print("Attempting to save partial results... [check_batch_status]")
        
        translations = save_partial_batch_results(client, temp_dir, batch_id, state_file, status)
        if translations:
            print("\nPartial results saved successfully [check_batch_status]")
            print("You can now use --mode resume to complete the remaining translations [check_batch_status]")
            return None, None, None
    
    if status.status != "completed":
        return None, None, None
    
    print("Batch completed! Will retrieve results... [check_batch_status]")
    response = client.files.content(status.output_file_id)
    response_text = response.read().decode('utf-8')
    return state, state_file, status

def save_partial_batch_results(client, temp_dir, batch_id, state_file_path, status):
    """Save partial results from an expired/cancelled batch job"""
    try:
        # First download the output file from the API
        if not status.output_file_id:
            print("No output file available [save_partial_batch_results]")
            return None
            
        print(f"Downloading output file {status.output_file_id}... [save_partial_batch_results]")
        response = client.files.content(status.output_file_id)
        output_text = response.read().decode('utf-8')
        
        # Save the raw output file
        output_filename = f"batch_{batch_id}_output.jsonl"
        output_path = temp_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Saved raw output to: {output_path} [save_partial_batch_results]")
            
        translations = {}
        # Process the downloaded content line by line
        for line in output_text.splitlines():
            # Rest of processing remains the same
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                chunk_id = result.get('custom_id')
                if not chunk_id:
                    continue
                    
                response = result.get('response', {})
                if not response or response.get('error'):
                    continue
                    
                body = response.get('body', {})
                choices = body.get('choices', [])
                if not choices:
                    continue
                    
                content = choices[0].get('message', {}).get('content')
                if content:
                    translations[chunk_id] = content
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Warning: Error processing result for chunk {chunk_id}: {e} [save_partial_batch_results]")
                continue
        
        if not translations:
            print("No valid translations found in partial results [save_partial_batch_results]")
            return None
            
        print(f"Found {len(translations)} valid translations in partial results [save_partial_batch_results]")
        
        # Load the original job state to get the temp directory structure
        with open(state_file_path, 'r', encoding='utf-8') as f:
            original_state = json.load(f)
            
        # Reconstruct the paths dictionary
        paths = {k: Path(v) for k, v in original_state['paths'].items()}
        
        # Save the translations to the original job's translations file
        save_translations(paths, translations)
        print(f"Saved {len(translations)} translations to {paths['translations_file']} [save_partial_batch_results]")
        
        return translations
        
    except Exception as e:
        print(f"Error processing partial results: {e} [save_partial_batch_results]")
        return None

def select_resumable_job(input_epub_path, from_lang, to_lang, model, mode):
    """Helper function to select a resumable job, used by both resume and resumebatch modes"""
    print("\nLooking for resumable translation jobs... [select_resumable_job]")
    resumable_jobs = find_resumable_jobs(input_epub_path, from_lang, to_lang, model)
    if not resumable_jobs:
        print("\nNo resumable jobs found. [select_resumable_job]")
        if mode == 'resumebatch':
            print("Please run with --mode batch first to create a new batch job. [select_resumable_job]")
            sys.exit(1)
        choice = input("Start new translation job? (y/N): [select_resumable_job]")
        if choice.lower() != 'y':
            print("Aborting. [select_resumable_job]")
            return None  # Return None instead of sys.exit()
        return None
        
    print("\nFound resumable translation jobs: [select_resumable_job]")
    for i, (job_id, timestamp, state) in enumerate(resumable_jobs, 1):
        print(f"{i}. Job from {timestamp} [select_resumable_job]")
        print(f"   Progress: {state['chunks_completed']}/{state['chunks_total']} chunks [select_resumable_job]")
        print(f"   Last updated: {state['last_updated']} [select_resumable_job]")
    
    while True:
        choice = input("\nEnter job number to resume (or 'q' to quit): [select_resumable_job]")
        if choice.lower() == 'q':
            print("Aborting. [select_resumable_job]")
            return None  # Return None instead of sys.exit()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(resumable_jobs):
                return resumable_jobs[idx][0]
        except ValueError:
            pass
        print("Invalid choice, please try again [select_resumable_job]")

from state_manager import StateManager
from translation_manager import TranslationManager
from epub_processor import EPubProcessor

async def translate(client, input_epub_path, output_epub_path, from_lang='DE', to_lang='EN', 
                   mode=None, model='gpt-4o-mini', fast=False, resume_job_id=None, 
                   debug=False):
    """Main translation function using the manager classes"""
    try:
        # Create managers
        job_id = resume_job_id or create_job_id(input_epub_path, from_lang, to_lang, model)
        state_manager = StateManager(job_id, ensure_temp_dir())
        epub_processor = EPubProcessor(Path(input_epub_path), Path(output_epub_path))
        translation_manager = TranslationManager(client, state_manager, epub_processor)

        # Try to load existing state
        state = state_manager.load_state()
        if state:
            chunks = state['chunks']
            chapter_map = state['chapter_map']
            translations = state.get('translations', {})
            print(f"Resumed with {len(translations)}/{len(chunks)} translations")
        else:
            # Extract new chunks
            chunks, chapter_map = epub_processor.extract_chunks()
            translations = {}
            state_manager.save_chunks(chunks, chapter_map)
            
        if mode == 'batch':
            translations = await translation_manager.process_batch(chunks, from_lang, to_lang, model)
        elif mode in ['fast', 'resume']:
            untranslated = [(cid, text) for cid, text in chunks if cid not in translations]
            for chunk_id, text in untranslated:
                result = await translation_manager.process_chunk(chunk_id, text, from_lang, to_lang, model)
                if result:
                    translations[chunk_id] = result
                    state_manager.save_translations(translations)
        
        if translations:
            epub_processor.reassemble(translations, chapter_map)
            print(f"Translated EPUB saved to {output_epub_path}")
        
        return True
            
    except Exception as e:
        print(f"Error during translation: {e}")
        return False

def find_resumable_jobs(input_epub_path, from_lang, to_lang, model):
    """Find all resumable jobs for the given parameters without requiring job_state.json"""
    temp_dir = ensure_temp_dir()
    resumable_jobs = []
    
    # Look for job directories
    prefix = f"{Path(input_epub_path).stem}_{from_lang}_{to_lang}_{model}_"
    for job_dir in temp_dir.iterdir():
        if not job_dir.is_dir():
            continue
            
        try:
            # Check if this is a job directory for our input file
            if not job_dir.name.startswith(prefix):
                print(f"Skipping directory {job_dir.name}: doesn't match pattern {prefix}* [find_resumable_jobs]")
                continue

            paths = ensure_temp_structure(job_dir.name)
            
            # Only require chunks.json and optionally translations.json
            if not paths['chunks_file'].exists():
                print(f"Skipping directory {job_dir.name}: missing chunks.json [find_resumable_jobs]")
                continue
                
            # Count total chunks
            with open(paths['chunks_file'], 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                chunks_total = len(chunks_data['chunks'])
            
            # Count completed translations
            chunks_completed = 0
            if paths['translations_file'].exists():
                with open(paths['translations_file'], 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                    chunks_completed = len(translations)
            else:
                print(f"Note: Directory {job_dir.name} has no translations.json yet [find_resumable_jobs]")
            
            # Skip if no translations yet
            if chunks_completed == 0:
                print(f"Skipping directory {job_dir.name}: no translations yet (0/{chunks_total} chunks) [find_resumable_jobs]")
                continue
            
            # Note if job is complete
            if chunks_completed >= chunks_total:
                print(f"Found completed job in {job_dir.name}: {chunks_completed}/{chunks_total} chunks [find_resumable_jobs]")
            else:
                print(f"Found resumable job in {job_dir.name}: {chunks_completed}/{chunks_total} chunks completed [find_resumable_jobs]")
            
            # Extract and properly format timestamp from directory name
            date_part = job_dir.name.split('_')[-2]  # Format: YYYYMMDD
            time_part = job_dir.name.split('_')[-1]  # Format: HHMMSS
            
            # Parse each component
            year = date_part[:4]
            month = date_part[4:6]
            day = date_part[6:8]
            hour = time_part[:2]
            minute = time_part[2:4]
            second = time_part[4:6]
            
            raw_timestamp = f"{date_part}_{time_part}"  # For sorting
            padded_timestamp = f"{year}-{month}-{day}T{hour}:{minute}:{second}.000000+00:00"
            
            state = {
                'chunks_total': chunks_total,
                'chunks_completed': chunks_completed,
                'last_updated': padded_timestamp
            }
                
            resumable_jobs.append((job_dir.name, raw_timestamp, state))
                
        except Exception as e:
            print(f"Warning: Could not process directory {job_dir.name}: {e} [find_resumable_jobs]")
            continue
    
    return sorted(resumable_jobs, key=lambda x: x[2]['last_updated'], reverse=True)

if __name__ == "__main__":
    import sys
    try:
        parser = argparse.ArgumentParser(description='App to translate EPUB books.')

        parser.add_argument('--input', required=True, help='Input file path.')
        parser.add_argument('--output', help='Output file path. Defaults to [input]_[to-lang]_[model].epub')
        parser.add_argument('--from-lang', help='Source language.', default='DE')
        parser.add_argument('--to-lang', help='Target language.', default='EN')
        parser.add_argument('--debug', action='store_true', help='Keep batch files and temp directory for debugging.')
        parser.add_argument('--mode', choices=['batch', 'savebatch', 'resume', 'test', 'batchcheck', 'resumebatch'], 
                           help='Processing mode: "batch" for batch API, "savebatch" to save batch files, '
                                '"resume" to continue previous job, "test" to use test translations, '
                                '"batchcheck" to check batch status, "resumebatch" to resume with batch processing')
        parser.add_argument('--model', help='The model to use for translation (default: gpt-4o-mini)', default='gpt-4o-mini')
        # Remove --inline-css argument

        args = parser.parse_args()

        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"Input file not found: {args.input} [main]")
        if input_path.suffix.lower() != '.epub':
            parser.error(f"Input file must be an EPUB file: {args.input} [main]")
        
        # Check for resumable jobs if in resume mode
        job_id = None
        if args.mode in ['resume', 'resumebatch']:  # Changed condition to catch both modes
            print("\nLooking for resumable translation jobs... [main]")
            resumable_jobs = find_resumable_jobs(args.input, args.from_lang, args.to_lang, args.model)
            if resumable_jobs:
                print("\nFound resumable translation jobs: [main]")
                for i, (job_id, timestamp, state) in enumerate(resumable_jobs, 1):
                    print(f"{i}. Job from {timestamp} [main]")
                    print(f"   Progress: {state['chunks_completed']}/{state['chunks_total']} chunks [main]")
                    print(f"   Last updated: {state['last_updated']} [main]")
                
                while True:
                    choice = input("\nEnter job number to resume (or 'n' for new job, 'q' to quit) [main]:")
                    if choice.lower() == 'n':
                        job_id = None
                        break
                    elif choice.lower() == 'q':
                        print("Aborting. [main]")
                        sys.exit(0)
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(resumable_jobs):
                            job_id = resumable_jobs[idx][0]
                            break
                    except ValueError:
                        pass
                    print("Invalid choice, please try again [main]")
            else:
                print("\nNo resumable jobs found. [main]")
                if args.mode == 'resumebatch':  # Add special handling for resumebatch
                    print("Please run with --mode batch first to create a new batch job. [main]")
                    sys.exit(1)
                choice = input("Start new translation job? (y/N): [main]")
                if choice.lower() != 'y':
                    print("Aborting. [main]")
                    sys.exit(0)
                job_id = None

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_dir = ensure_output_dir()
            output_path = str(output_dir / f'{input_path.stem}_{args.to_lang.lower()}_{args.model}.epub')

        config = read_config()
        client = OpenAI(api_key=config['openai']['api_key'])
        
        # Call translate - remove test_translations_file parameter
        asyncio.run(translate(client, args.input, output_path, args.from_lang, args.to_lang, 
                 mode=args.mode, model=args.model, 
                 fast=(args.mode not in ['batch', 'savebatch']),
                 resume_job_id=job_id, 
                 debug=args.debug))  # Remove inline_css argument
    except KeyboardInterrupt:
        print("\nTranslation interrupted. Use --mode resume to continue later. [main]")
        sys.exit(1)


