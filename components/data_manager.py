import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class DataManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.generated_dir = os.path.join(base_dir, "generated")
        self.resources_dir = os.path.join(base_dir, "resources")
        
    def _ensure_dir(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_json(self, data: Any, path: str):
        """Saves data to a JSON file."""
        self._ensure_dir(os.path.dirname(path))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_json(self, path: str) -> Any:
        """Loads data from a JSON file."""
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_conversation(self, conversation_id: str, data: Dict[str, Any]):
        """Saves conversation data."""
        # Save to temp first or directly to experiences? 
        # Plan says data/generated/experiences
        path = os.path.join(self.generated_dir, "experiences", f"{conversation_id}.json")
        self.save_json(data, path)
        return path

    def save_report(self, conversation_id: str, report: Dict[str, Any]):
        """Saves evaluation report."""
        path = os.path.join(self.generated_dir, "reports", f"{conversation_id}_report.json")
        self.save_json(report, path)
        return path

    def load_resource(self, resource_type: str, filename: str) -> Optional[Dict[str, Any]]:
        """Loads a resource (jd, resume, etc)."""
        path = os.path.join(self.resources_dir, resource_type, filename)
        return self.load_json(path)

    def save_txt(self, data: str, path: str):
        """Saves data to a text file."""
        self._ensure_dir(os.path.dirname(path))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
