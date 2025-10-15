import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Find embedding with user_id
print("Finding embedding with user_id...")
batch = rag.collection.get(limit=1000, include=['metadatas'])
test_id = None
test_meta = None

for i, meta in enumerate(batch['metadatas']):
    if 'user_id' in meta:
        test_id = batch['ids'][i]
        test_meta = meta
        break

if not test_id:
    print("No embeddings with user_id found in first 1000")
    sys.exit(0)

print(f"Found: {test_id}")
print(f"Current metadata: {test_meta}")

# Try updating with all fields except user_id
new_meta = {k: v for k, v in test_meta.items() if k != 'user_id'}
print(f"New metadata (without user_id): {new_meta}")

print("\nUpdating...")
rag.collection.update(ids=[test_id], metadatas=[new_meta])

print("Fetching updated...")
updated = rag.collection.get(ids=[test_id], include=['metadatas'])
print(f"Result: {updated['metadatas'][0]}")

if 'user_id' in updated['metadatas'][0]:
    print("\n[FAIL] user_id still present!")
else:
    print("\n[OK] user_id removed!")
