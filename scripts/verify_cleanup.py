import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_pipeline import RAGPipeline

print("Checking cleanup status...")
rag = RAGPipeline()

total = rag.collection.count()
print(f"Total embeddings: {total}")

# Count in batches
print("\nScanning all embeddings...")
batch_size = 5000
count_with_user_id = 0

for offset in range(0, total, batch_size):
    print(f"  Batch {offset//batch_size + 1}/{(total-1)//batch_size + 1}...", end='\r')
    batch = rag.collection.get(
        limit=min(batch_size, total - offset),
        offset=offset,
        include=['metadatas']
    )
    count_with_user_id += sum(1 for m in batch['metadatas'] if 'user_id' in m)

print(f"\n\nResults:")
print(f"  With user_id: {count_with_user_id}")
print(f"  Without user_id: {total - count_with_user_id}")

if count_with_user_id == 0:
    print("\n[OK] All embeddings are clean!")
else:
    print(f"\n[INFO] {count_with_user_id} embeddings still have user_id")
    print("The previous cleanup may not have worked correctly.")
