#!/usr/bin/env python3
"""
Debug embedding model consistency
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def debug_embeddings():
    try:
        from src.core.rag_pipeline import RAGPipeline
        from config.database import db_manager
        from src.data.models import Document

        print("=== Embedding Model Debug ===")

        # Initialize RAG pipeline
        rag = RAGPipeline()
        print(f"Embedding model: {rag.embedding_model_name}")

        # Test embedding generation
        test_text = "data structure algorithms"
        embedding = rag.embedding_model.encode(test_text).tolist()
        print(f"Test embedding shape: {len(embedding)}")
        print(f"Test embedding sample: {embedding[:5]}")
        print(f"Test embedding range: min={min(embedding):.3f}, max={max(embedding):.3f}")

        # Check collection stats
        collection = rag.collection
        print(f"Collection count: {collection.count()}")

        # Get a sample stored embedding to compare
        try:
            sample_results = collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=['documents', 'metadatas', 'distances', 'embeddings']
            )

            if sample_results['documents'] and sample_results['documents'][0]:
                stored_embedding = sample_results['embeddings'][0][0] if 'embeddings' in sample_results else None
                distance = sample_results['distances'][0][0]

                print(f"Sample stored document: {sample_results['documents'][0][0][:50]}...")
                print(f"Distance to test query: {distance}")
                print(f"Calculated similarity: {1 - distance:.3f}")

                if stored_embedding:
                    print(f"Stored embedding shape: {len(stored_embedding)}")
                    print(f"Stored embedding sample: {stored_embedding[:5]}")
                    print(f"Stored embedding range: min={min(stored_embedding):.3f}, max={max(stored_embedding):.3f}")

                    # Calculate manual cosine similarity
                    import numpy as np

                    emb1 = np.array(embedding)
                    emb2 = np.array(stored_embedding)

                    # Cosine similarity
                    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    print(f"Manual cosine similarity: {cos_sim:.3f}")

                    # Euclidean distance
                    euclidean = np.linalg.norm(emb1 - emb2)
                    print(f"Manual euclidean distance: {euclidean:.3f}")

        except Exception as e:
            print(f"Error getting sample embedding: {e}")

        # Test with a simple, direct query
        print(f"\n=== Testing Simple Query ===")
        simple_query = "algorithm"
        simple_embedding = rag.embedding_model.encode(simple_query).tolist()

        simple_results = collection.query(
            query_embeddings=[simple_embedding],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )

        print(f"Simple query: '{simple_query}'")
        if simple_results['documents'] and simple_results['documents'][0]:
            for i, (doc, distance) in enumerate(zip(simple_results['documents'][0], simple_results['distances'][0])):
                similarity = 1 - distance
                print(f"  Result {i+1}: similarity={similarity:.3f}, content={doc[:50]}...")

        return True

    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_embeddings()