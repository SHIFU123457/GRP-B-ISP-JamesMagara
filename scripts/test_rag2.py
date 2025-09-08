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
        print(f"❌ Error during RAG testing: {e}")
        import traceback
        traceback.print_exc()

def create_test_document():
    """Create a test document for RAG testing"""
    print("\n📝 Creating test document...")
    
    # Create test content
    test_content = """
    Data Structures and Algorithms - Week 1 Notes
    
    Stack Data Structure:
    A stack is a linear data structure that follows the Last In First Out (LIFO) principle.
    Elements are added and removed from the same end, called the top of the stack.
    
    Main operations:
    - Push: Add an element to the top of the stack
    - Pop: Remove and return the top element
    - Peek/Top: Return the top element without removing it
    - isEmpty: Check if the stack is empty
    
    Queue Data Structure:
    A queue is a linear data structure that follows the First In First Out (FIFO) principle.
    Elements are added at the rear and removed from the front.
    
    Main operations:
    - Enqueue: Add an element to the rear of the queue
    - Dequeue: Remove and return the front element
    - Front: Return the front element without removing it
    - isEmpty: Check if the queue is empty
    
    Applications:
    Stacks are used in function calls, expression evaluation, and undo operations.
    Queues are used in scheduling, breadth-first search, and handling requests.
    """
    
    # Create test file
    test_file_path = project_root / "data" / "test_document.txt"
    test_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_file_path, 'w') as f:
        f.write(test_content)
    
    print(f"✅ Test document created at: {test_file_path}")
    
    # Add to database
    with db_manager.get_session() as session:
        # Get or create a test course
        test_course = session.query(Course).filter(Course.course_code == "ICS201").first()
        if not test_course:
            test_course = Course(
                course_code="ICS201",
                course_name="Data Structures and Algorithms",
                description="Test course for RAG pipeline",
                semester="1.2",
                year=2025,
                lms_platform="test"
            )
            session.add(test_course)
            session.flush()
        
        # Create test document
        test_doc = Document(
            course_id=test_course.id,
            title="Week 1 - Data Structures Introduction",
            file_path=str(test_file_path),
            file_type="txt",
            is_processed=False,
            processing_status="pending"
        )
        session.add(test_doc)
        session.commit()
        
        print(f"✅ Test document added to database with ID: {test_doc.id}")
        return test_doc.id

if __name__ == "__main__":
    print("🚀 Starting RAG Pipeline Tests")
    print("=" * 50)
    
    # Create test document first
    test_doc_id = create_test_document()
    
    # Run RAG tests
    test_rag_pipeline()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")