import sys
from pathlib import Path

print("=== Script Starting ===")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

from src.core.rag_pipeline import RAGPipeline

print("Initializing RAG pipeline...")
rag = RAGPipeline()
collection = rag.collection

print(f"Collection name: {collection.name}")
print(f"Total embeddings: {collection.count()}")

# Get sample
print("\nGetting sample embeddings...")
sample = collection.get(limit=5, include=['metadatas'])

print(f"Retrieved {len(sample['ids'])} embeddings")

if sample['metadatas']:
    print("\nSample metadata:")
    for i, metadata in enumerate(sample['metadatas'][:3], 1):
        print(f"  {i}. {metadata}")
        if 'user_id' in metadata:
            print(f"     ^ Has user_id: {metadata['user_id']}")

print("\n=== Script Complete ===")
