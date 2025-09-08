#!/usr/bin/env python3
"""
Test script for RAG pipeline functionality
Run this to verify your RAG implementation works correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database import db_manager
from src.core.rag_pipeline import RAGPipeline
from src.data.models import Document, Course

def test_rag_pipeline():
    """Test the RAG pipeline with existing documents"""
    print("🧪 Testing RAG Pipeline...")
    
    try:
        # Initialize RAG pipeline
        print("📡 Initializing RAG pipeline...")
        rag = RAGPipeline()
        
        # Check vector store stats
        stats = rag.get_vector_store_stats()
        print(f"📊 Vector Store Stats: {stats}")
        
        # Test document processing
        print("\n📄 Testing document processing...")
        with db_manager.get_session() as session:
            # Get first unprocessed document
            unprocessed_doc = session.query(Document).filter(
                Document.is_processed == False
            ).first()
            
            if unprocessed_doc:
                print(f"Processing document: {unprocessed_doc.title}")
                success = rag.process_and_store_document(unprocessed_doc.id)
                print(f"✅ Processing {'succeeded' if success else 'failed'}")
            else:
                print("ℹ️ No unprocessed documents found")
        
        # Test retrieval
        print("\n🔍 Testing document retrieval...")
        test_queries = [
            "What is a stack?",
            "How do you implement a queue?", 
            "Explain data structures",
            "What are algorithms?"
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            results = rag.retrieve_relevant_chunks(query, top_k=3)
            
            if results:
                print(f"Found {len(results)} relevant chunks:")
                for i, result in enumerate(results[:2], 1):  # Show top 2
                    print(f"  {i}. Similarity: {result['similarity_score']:.3f}")
                    print(f"     Source: {result['metadata'].get('title', 'Unknown')}")
                    print(f"     Text: {result['text'][:100]}...")
            else:
                print("  No relevant chunks found")
        
        # Test context generation
        print("\n🎯 Testing context generation...")
        context = rag.generate_context("What is a stack data structure?")
        
        if context['has_relevant_content']:
            print("✅ Context generated successfully")
            print(f"   Sources: {len(context['sources'])}")
            print(f"   Context length: {len(context['context'])} characters")
            print(f"   Preview: {context['context'][:200]}...")
        else:
            print("ℹ️ No relevant content found for context")
        
        print("\n🎉 RAG pipeline test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        print("\n🏁 RAG pipeline test completed!")
