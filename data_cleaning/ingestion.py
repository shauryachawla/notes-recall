import logging
from pathlib import Path
import json
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Note(BaseModel):
    text: str
    creation_date: int
    isArchived: bool

def fetch_notes(folder_path: str) -> list[Note]:
    notes = []
    base_dir = Path(folder_path)
    
    if not base_dir.is_dir():
        logging.error(f"Error: '{folder_path}' is not a valid directory.")
        return
    
    for file_path in base_dir.glob('*'):
        if file_path.is_file():
            note = parse_file(file_path)
            if note:
                notes.append(note)
    return notes
    
def parse_file(file_path: Path) -> Note:
    match file_path.suffix.lower():
        case '.json':
            return process_json_file(file_path)
        case _:
            logging.warning(f"Unsupported file type: {file_path.suffix} for file {file_path}")
            
def process_json_file(file_path: Path) -> Note:
    logging.info(f"Processing JSON file: {file_path}")
    try:
        content = file_path.read_text(encoding='utf-8')
        data = json.loads(content)
        return Note.model_validate({
            "text": data["textContent"],
            "creation_date": data["createdTimestampUsec"],
            "isArchived": data["isArchived"],
        })
    except Exception as e:
        logging.error(f"Error processing JSON file {file_path}: {e}")
    