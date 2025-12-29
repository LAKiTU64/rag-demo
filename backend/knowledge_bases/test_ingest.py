#!/usr/bin/env python3
import json, sys
from pathlib import Path

# Ensure parent of 'backend' is on sys.path so that 'backend.knowledge_bases' is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from backend.knowledge_bases.kb_ingest import ingest_json_to_faiss, DEFAULT_INDEX_DIR
except Exception as e:
    print(json.dumps({"error": "import_failed", "detail": str(e), "root": str(ROOT)}))
    sys.exit(1)

def main(path: str):
    p = Path(path)
    if not p.exists():
        print(json.dumps({"error": "file_not_found", "path": path}))
        return
    content = p.read_text(encoding='utf-8')
    result = ingest_json_to_faiss(content)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '/workspace/Agent/AI_Agent_Complete/mini_kb.json'
    main(target)
