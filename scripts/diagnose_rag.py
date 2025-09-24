#!/usr/bin/env python3
"""
Diagnostic script to identify RAG issues
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def diagnose_rag():
    try:
        from config.database import db_manager
        from src.core.rag_pipeline import RAGPipeline
        from src.data.models import Document

        print("=== RAG Diagnosis ===")

        # Check database
        print("\n1. Checking database...")
        with db_manager.get_session() as session:
            doc_count = session.query(Document).count()
            processed_docs = session.query(Document).filter(Document.is_processed == True).count()
            print(f"   Total documents: {doc_count}")
            print(f"   Processed documents: {processed_docs}")

            if processed_docs > 0:
                print("   Sample processed documents:")
                sample_docs = session.query(Document).filter(Document.is_processed == True).limit(3).all()
                for doc in sample_docs:
                    print(f"     - {doc.title} ({doc.file_type})")

        # Check RAG pipeline
        print("\n2. Initializing RAG pipeline...")
        rag = RAGPipeline()

        # Check vector store
        print("\n3. Checking vector store...")
        stats = rag.get_vector_store_stats()
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   LLM service available: {stats['llm_service_available']}")

        # Test simple retrieval
        print("\n4. Testing retrieval...")
        test_query = "document"
        print(f"   Query: '{test_query}'")

        chunks = rag.retrieve_relevant_chunks(test_query, top_k=5)
        print(f"   Retrieved chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks[:2]):
            print(f"     Chunk {i+1}: similarity={chunk['similarity_score']:.3f}")
            print(f"     Text preview: {chunk['text'][:100]}...")

        # Test context generation
        print("\n5. Testing context generation...")
        context_data = rag.generate_context(test_query)
        print(f"   Has relevant content: {context_data['has_relevant_content']}")

        if context_data['has_relevant_content']:
            print(f"   Sources: {len(context_data['sources'])}")
            print(f"   Context length: {len(context_data['context'])}")

        # Test full RAG response
        print("\n6. Testing full RAG response...")
        try:
            response = rag.generate_rag_response(test_query)
            print(f"   Context used: {response['context_used']}")
            print(f"   Confidence: {response['confidence']}")
            if 'error' in response:
                print(f"   Error: {response['error']}")
        except Exception as e:
            print(f"   RAG response failed: {e}")

        print("\n=== Diagnosis Complete ===")

    except Exception as e:
        print(f"Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_rag()