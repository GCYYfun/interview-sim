
import os
from typing import List, Dict, Optional

class InterviewSelector:
    def __init__(self, resource_path: str = "data/resources/conversations"):
        self.resource_path = resource_path
        self.transcripts: List[Dict] = []

    def scan(self) -> List[Dict]:
        """Scans the directory for transcript files and parses metadata."""
        self.transcripts = []
        if not os.path.exists(self.resource_path):
            return []

        files = [f for f in os.listdir(self.resource_path) if f.endswith('.pdf') or f.endswith('.txt')]
        files.sort()

        for f in files:
            name_without_ext = os.path.splitext(f)[0]
            # Expected format: name_desc_transcript_x OR name_transcript_x (fallback)
            parts = name_without_ext.split('_')
            
            # Simple heuristic parsing
            candidate_name = parts[0]
            
            # Try to identify JD/desc
            # Common pattern: name_jd_transcript_stage
            # But the user example was: name_desc_transcript_x
            # Let's extract what we can
            
            job_desc = "unknown"
            if len(parts) > 1:
                # If the second part isn't 'transcript', it might be the JD
                if parts[1] != 'transcript':
                    job_desc = parts[1]

            item = {
                "file": f,
                "name": name_without_ext,
                "candidate": candidate_name,
                "jd": job_desc,
                "path": os.path.join(self.resource_path, f)
            }
            self.transcripts.append(item)
        
        return self.transcripts

    def filter(self, criteria: Dict[str, str]) -> List[Dict]:
        """Filters scanned transcripts based on criteria."""
        if not self.transcripts:
            self.scan()
            
        filtered = self.transcripts
        
        if 'jd' in criteria:
            target_jd = criteria['jd'].lower()
            filtered = [t for t in filtered if target_jd in t['jd'].lower()]
            
        if 'candidate' in criteria:
            target_name = criteria['candidate'].lower()
            filtered = [t for t in filtered if target_name in t['candidate'].lower()]
            
        return filtered

    def interactive_select(self) -> List[Dict]:
        """Interactive CLI for selecting transcripts."""
        if not self.transcripts:
            self.scan()
            
        if not self.transcripts:
            print("No transcripts found.")
            return []

        while True:
            print("\nAvailable Transcripts:")
            print(f"{'Idx':<4} {'Filename':<40} {'JD':<15} {'Candidate':<15}")
            print("-" * 80)
            
            for i, t in enumerate(self.transcripts):
                print(f"{i+1:<4} {t['file']:<40} {t['jd']:<15} {t['candidate']:<15}")
            
            print("\nOptions:")
            print(" - Enter numbers (e.g., '1', '1,3', '1-5') to select")
            print(" - Enter 'all' to select all")
            print(" - Enter 'f:jd:<name>' to filter by JD (e.g., 'f:jd:java')")
            print(" - Enter 'r' to reset filters")
            print(" - Enter 'q' to quit")
            
            choice = input("\nSelect > ").strip()
            
            if choice.lower() == 'q':
                return [] # Or exit? Let caller decide, but empty means nothing selected.
                
            if choice.lower() == 'r':
                self.scan()
                continue
                
            if choice.lower().startswith('f:jd:'):
                jd_filter = choice.split(':', 2)[2]
                self.transcripts = self.filter({'jd': jd_filter})
                continue
                
            if choice.lower() == 'all':
                return self.transcripts
                
            # Parse selection
            selected_indices = set()
            try:
                parts = choice.split(',')
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        for i in range(start, end + 1):
                            selected_indices.add(i - 1)
                    else:
                        selected_indices.add(int(part) - 1)
                
                selected_items = []
                for idx in sorted(selected_indices):
                    if 0 <= idx < len(self.transcripts):
                        selected_items.append(self.transcripts[idx])
                
                if selected_items:
                    print(f"\nSelected {len(selected_items)} transcripts.")
                    return selected_items
                else:
                    print("No valid selection.")
            except ValueError:
                print("Invalid input format.")
