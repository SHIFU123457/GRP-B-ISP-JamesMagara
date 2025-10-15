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
from src.data.models import Document, Course, User, DocumentAccess

def test_rag_pipeline(user_id: int):
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
            from src.data.models import DocumentAccess

            # Get document IDs user has access to
            accessible_doc_ids = [
                access.document_id
                for access in session.query(DocumentAccess).filter(
                    DocumentAccess.user_id == user_id,
                    DocumentAccess.is_active == True
                ).all()
            ]

            # Get first unprocessed document that user has access to
            unprocessed_doc = session.query(Document).filter(
                Document.id.in_(accessible_doc_ids),
                Document.is_processed == False
            ).first()

            if unprocessed_doc:
                print(f"Processing document: {unprocessed_doc.title}")
                success = rag.process_and_store_document(unprocessed_doc.id)
                print(f"✅ Processing {'succeeded' if success else 'failed'}")
            else:
                print("ℹ️ No unprocessed documents found for this user")
        
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
            results = rag.retrieve_relevant_chunks(query, user_id=user_id, top_k=3)
            
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
        context = rag.generate_context("What is a stack data structure?", user_id=user_id)
        
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
    """Create a test document for RAG testing, returning the document ID and user ID."""
    print("\n📝 Creating test user and document...")
    
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
        # Get or create a test user
        test_user = session.query(User).filter(User.telegram_id == "rag_test_user").first()
        if not test_user:
            test_user = User(telegram_id="rag_test_user", first_name="RAG Test")
            session.add(test_user)
            session.flush()

        # Get or create a test course
        test_course = session.query(Course).filter(Course.course_code == "RAG-TEST").first()
        if not test_course:
            test_course = Course(
                course_code="RAG-TEST",
                course_name="RAG Test Course",
                lms_platform="test"
            )
            session.add(test_course)
            session.flush()

        # HYBRID ARCHITECTURE: Check if document already exists at COURSE level
        test_doc = session.query(Document).filter(
            Document.title == "RAG Test Document - Data Structures",
            Document.course_id == test_course.id
        ).first()

        if not test_doc:
            # Create test document WITHOUT user_id (shared at course level)
            test_doc = Document(
                course_id=test_course.id,
                title="RAG Test Document - Data Structures",
                file_path=str(test_file_path),
                file_type="txt",
                is_processed=False,
                processing_status="pending"
            )
            session.add(test_doc)
            session.flush()
            print(f"✅ Created new test document ID {test_doc.id}")
        else:
            print(f"✅ Found existing test document ID {test_doc.id}")

        # HYBRID ARCHITECTURE: Grant access to this user via DocumentAccess
        existing_access = session.query(DocumentAccess).filter(
            DocumentAccess.user_id == test_user.id,
            DocumentAccess.document_id == test_doc.id
        ).first()

        if not existing_access:
            doc_access = DocumentAccess(
                user_id=test_user.id,
                document_id=test_doc.id,
                access_source="test",
                is_active=True
            )
            session.add(doc_access)
            print(f"✅ Granted access to user {test_user.id} for document {test_doc.id}")
        else:
            print(f"✅ User {test_user.id} already has access to document {test_doc.id}")

        session.commit()

        print(f"✅ Test setup complete: Document ID {test_doc.id}, User ID {test_user.id}")
        return test_doc.id, test_user.id

if __name__ == "__main__":
    print("🚀 Starting RAG Pipeline Tests")
    print("=" * 50)
    
    # Create test document first
    test_doc_id, test_user_id = create_test_document()
    
    # Run RAG tests
    test_rag_pipeline(test_user_id)
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")