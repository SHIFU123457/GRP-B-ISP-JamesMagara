import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import uuid

from config.settings import settings
from config.database import db_manager
from src.data.models import Document, DocumentEmbedding
from src.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for semantic document search"""
    
    def __init__(self):
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.vector_store_path = settings.VECTOR_STORE_PATH
        
        # Initialize components
        self.embedding_model = None
        self.vector_store = None
        self.collection = None
        self.document_processor = DocumentProcessor()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and vector store"""
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Initialize ChromaDB vector store
            logger.info(f"Initializing vector store at: {self.vector_store_path}")
            
            # Create directory if it doesn't exist
            Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.vector_store = chromadb.PersistentClient(
                path=self.vector_store_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.vector_store.get_or_create_collection(
                name="study_documents",
                metadata={"description": "Academic documents for Study Helper Agent"}
            )
            
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def process_and_store_document(self, document_id: int) -> bool:
        """Process a document and store its embeddings"""
        try:
            with db_manager.get_session() as session:
                # Get document from database
                document = session.query(Document).filter(Document.id == document_id).first()
                
                if not document:
                    logger.error(f"Document with ID {document_id} not found")
                    return False
                
                if not document.file_path or not Path(document.file_path).exists():
                    logger.error(f"Document file not found: {document.file_path}")
                    return False
                
                # Process document into chunks
                metadata = {
                    'document_id': document.id,
                    'title': document.title,
                    'course_id': document.course_id,
                    'file_type': document.file_type,
                    'course_code': document.course.course_code if document.course else 'Unknown'
                }
                
                chunks = self.document_processor.process_document(
                    document.file_path, 
                    document.file_type, 
                    metadata
                )
                
                if not chunks:
                    logger.error(f"No chunks created from document {document_id}")
                    return False
                
                # Create embeddings and store in vector database
                stored_count = 0
                for chunk in chunks:
                    if self._store_chunk_embedding(chunk, document_id, session):
                        stored_count += 1
                
                # Update document status
                document.is_processed = True
                document.processing_status = "completed"
                document.content_text = chunks[0]['text'][:1000] + "..." if chunks else ""
                
                session.commit()
                logger.info(f"Successfully processed document {document_id}: {stored_count} chunks stored")
                return True
                
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            return False
    
    def _store_chunk_embedding(self, chunk: Dict[str, Any], document_id: int, session) -> bool:
        """Store a single chunk embedding"""
        try:
            # Generate embedding
            text = chunk['text']
            embedding = self.embedding_model.encode(text).tolist()
            
            # Generate unique ID for this chunk
            chunk_id = str(uuid.uuid4())
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[chunk['metadata']],
                ids=[chunk_id]
            )
            
            # Store embedding reference in database
            doc_embedding = DocumentEmbedding(
                document_id=document_id,
                chunk_text=text,
                chunk_index=chunk['metadata']['chunk_index'],
                embedding_vector_id=chunk_id
            )
            session.add(doc_embedding)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunk embedding: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, course_id: Optional[int] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a given query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare filter for course-specific search
            where_filter = {}
            if course_id:
                where_filter["course_id"] = course_id
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            relevant_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    relevant_chunks.append({
                        'text': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query: {query[:50]}...")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            return []
    
    def generate_context(self, query: str, course_id: Optional[int] = None, max_context_length: int = 2000) -> Dict[str, Any]:
        """Generate context for RAG-enhanced response"""
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query, course_id)
            
            if not relevant_chunks:
                return {
                    'context': '',
                    'sources': [],
                    'has_relevant_content': False
                }
            
            # Build context string
            context_parts = []
            sources = []
            current_length = 0
            
            for chunk in relevant_chunks:
                chunk_text = chunk['text']
                
                # Check if adding this chunk would exceed limit
                if current_length + len(chunk_text) > max_context_length:
                    break
                
                context_parts.append(f"[Source: {chunk['metadata'].get('title', 'Unknown')}]\n{chunk_text}")
                sources.append({
                    'title': chunk['metadata'].get('title', 'Unknown'),
                    'course_code': chunk['metadata'].get('course_code', 'Unknown'),
                    'similarity_score': chunk['similarity_score'],
                    'chunk_index': chunk['metadata'].get('chunk_index', 0)
                })
                
                current_length += len(chunk_text)
            
            context = "\n\n".join(context_parts)
            
            return {
                'context': context,
                'sources': sources,
                'has_relevant_content': True,
                'total_chunks': len(relevant_chunks),
                'used_chunks': len(context_parts)
            }
            
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return {
                'context': '',
                'sources': [],
                'has_relevant_content': False
            }
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            count = self.collection.count()
            
            with db_manager.get_session() as session:
                processed_docs = session.query(Document).filter(Document.is_processed == True).count()
                total_docs = session.query(Document).count()
            
            return {
                'total_embeddings': count,
                'processed_documents': processed_docs,
                'total_documents': total_docs,
                'processing_rate': f"{(processed_docs/total_docs*100):.1f}%" if total_docs > 0 else "0%"
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                'total_embeddings': 0,
                'processed_documents': 0,
                'total_documents': 0,
                'processing_rate': "0%"
            }