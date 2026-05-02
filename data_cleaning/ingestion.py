import logging
from pathlib import Path
import json
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Note(BaseModel):
    """Represents a note with text, creation date, and archived status."""
    title: str
    text: str
    creation_date: int|str
    isArchived: bool
    embedding: list[float] = None
    labels: list[str] = Field(default_factory=list)

def fetch_notes(folder_path: str) -> list[Note]:
    """Fetch notes from a directory and return validated Note objects."""
    notes = []
    base_dir = Path(folder_path)
    if not base_dir.is_dir():
        logging.error("Error: '%s' is not a valid directory.", folder_path)
        return []

    for file_path in base_dir.glob('*.json'):
        if file_path.is_file():
            note = parse_file(file_path)
            if note:
                notes.append(note)
    return notes

def parse_file(file_path: Path) -> Note:
    """Parse a file and return a Note object."""
    match file_path.suffix.lower():
        case '.json':
            return process_json_file(file_path)
        case _:
            logging.warning("Unsupported file type: %s for file %s", file_path.suffix, file_path)
            return None

def process_json_file(file_path: Path) -> Note:
    """Process a JSON file and return a validated Note object."""
    logging.info("Processing JSON file: %s", file_path)
    try:
        content = file_path.read_text(encoding='utf-8')
        data = json.loads(content)
        label_names = [n for x in (data.get("labels") or []) if isinstance(x, dict) and (n := x.get("name"))]
        if "credz" in label_names:
            return None
        return Note.model_validate({
            "title": data.get("title", ""),
            "text": data["textContent"],
            "labels": label_names,
            "creation_date": data["createdTimestampUsec"],
            "isArchived": data["isArchived"],
        })
    except (json.JSONDecodeError, KeyError, OSError, UnicodeDecodeError) as e:
        logging.error("Error processing JSON file %s: %s", file_path, e)
        return None
