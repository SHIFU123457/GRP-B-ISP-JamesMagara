import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Search for embeddings with user_id
batch = rag.collection.get(limit=5000, offset=10000, include=['metadatas'])
found = [(i, batch['ids'][i], batch['metadatas'][i]) for i, m in enumerate(batch['metadatas']) if 'user_id' in m]

print(f"Found {len(found)} embeddings with user_id in this batch")
if found:
    print("\nFirst 3 samples:")
    for idx, id, meta in found[:3]:
        print(f"  ID: {id}")
        print(f"  Metadata: {meta}")
        print()
