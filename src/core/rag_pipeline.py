import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import uuid
from typing import Union
from functools import wraps
import time

from config.settings import settings
from config.database import db_manager
from src.data.models import Document, DocumentEmbedding
from src.services.document_processor import DocumentProcessor
from src.services.llm_integration import LLMService

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator

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
        self.llm_service = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model, vector store and llm service"""
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
            
            # Get or create collection with cosine similarity
            self.collection = self.vector_store.get_or_create_collection(
                name="study_documents",
                metadata={"description": "Academic documents for Study Helper Agent", "hnsw:space": "cosine"}
            )
            
            # Initialize LLM service
            logger.info("Initializing LLM service...")
            self.llm_service = LLMService()
            
            logger.info("RAG pipeline initialized successfully with LLM integration")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=2.0)
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
    
    @retry_with_backoff(max_retries=2, base_delay=1.0)
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
    
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def retrieve_relevant_chunks(self, query: str, course_id: Optional[int] = None, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a given query"""
        try:
            # Use configured top_k if not specified
            if top_k is None:
                top_k = settings.TOP_K_RETRIEVAL

            # Generate query embedding
            logger.info(f"Using embedding model: {self.embedding_model_name}")
            query_embedding = self.embedding_model.encode(query).tolist()
            logger.info(f"Query embedding shape: {len(query_embedding)}")
            logger.info(f"Query embedding sample: {query_embedding[:5]}")  # First 5 values

            # Prepare filter for course-specific search
            where_filter = {}
            if course_id:
                where_filter["course_id"] = course_id

            # Search in ChromaDB - temporarily remove where filter to debug
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    # where=where_filter if where_filter else None,  # Temporarily disabled
                    include=['documents', 'metadatas', 'distances']
                )
                logger.info(f"ChromaDB query successful - where filter disabled for debugging")
            except Exception as query_error:
                logger.error(f"ChromaDB query failed: {query_error}")
                return []

            # Format results with debug logging
            relevant_chunks = []
            all_chunks_count = 0

            if results['documents'] and results['documents'][0]:
                all_chunks_count = len(results['documents'][0])
                logger.info(f"Found {all_chunks_count} total chunks from vector search")

                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Handle both cosine and euclidean distance metrics
                    if distance > 1.0:
                        # Euclidean distance - convert to similarity differently
                        # For euclidean, smaller distance = higher similarity
                        # Normalize to 0-1 range where 1 is most similar
                        max_distance = 2.0  # Approximate max for sentence embeddings
                        similarity_score = max(0, (max_distance - distance) / max_distance)
                        logger.info(f"    Euclidean distance {distance:.3f} -> similarity {similarity_score:.3f}")
                    else:
                        # Cosine distance - standard conversion
                        similarity_score = 1 - distance

                    # Add debug logging
                    logger.info(f"Chunk {i+1}: similarity={similarity_score:.3f}, threshold={settings.SIMILARITY_THRESHOLD}")

                    # Use a reasonable threshold now that we fixed distance calculation
                    if similarity_score >= 0.1:  # Accept chunks with 10%+ similarity
                        relevant_chunks.append({
                            'text': doc,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'rank': i + 1
                        })
            else:
                logger.warning("No documents returned from ChromaDB query")

            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks (from {all_chunks_count} total) for query: {query[:50]}...")

            # Log similarity scores for all chunks found
            if all_chunks_count > 0:
                logger.info("All chunk similarity scores and content preview:")
                for i, (distance, doc, metadata) in enumerate(zip(results['distances'][0], results['documents'][0], results['metadatas'][0])):
                    # Apply same distance conversion as above
                    if distance > 1.0:
                        max_distance = 2.0
                        similarity = max(0, (max_distance - distance) / max_distance)
                    else:
                        similarity = 1 - distance
                    logger.info(f"  Chunk {i+1}: similarity={similarity:.3f}")
                    logger.info(f"    Content: {doc[:100]}...")
                    logger.info(f"    Metadata: {metadata}")
                    logger.info(f"    Distance: {distance}")
                    logger.info(f"    Query: {query[:50]}...")

            # If no chunks found with relaxed threshold, there's a deeper issue
            if len(relevant_chunks) == 0:
                logger.error(f"No chunks found even with relaxed threshold. Vector store may be empty or query embedding failed.")

                # Check collection stats
                try:
                    collection_count = self.collection.count()
                    logger.error(f"Collection has {collection_count} total embeddings")
                except Exception as count_error:
                    logger.error(f"Failed to get collection count: {count_error}")
            else:
                logger.info(f"Successfully retrieved {len(relevant_chunks)} chunks with similarities: {[f'{c['similarity_score']:.3f}' for c in relevant_chunks]}")

            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            return []
    
    def generate_context(self, query: str, course_id: Optional[int] = None, max_context_length: Optional[int] = None) -> Dict[str, Any]:
        """Generate enhanced context for RAG-enhanced response with better source integration"""
        try:
            # Use configurable context length
            if max_context_length is None:
                max_context_length = settings.CONTEXT_MAX_LENGTH

            # Retrieve relevant chunks with higher similarity threshold for better quality
            relevant_chunks = self.retrieve_relevant_chunks(query, course_id, top_k=settings.TOP_K_RETRIEVAL)

            if not relevant_chunks:
                return {
                    'context': '',
                    'sources': [],
                    'has_relevant_content': False
                }

            # Filter chunks by minimum similarity score for higher quality
            # Use extremely low threshold for debugging
            high_quality_chunks = [chunk for chunk in relevant_chunks if chunk['similarity_score'] >= 0.01]

            if not high_quality_chunks:
                # If no high-quality chunks, use ALL chunks we retrieved
                high_quality_chunks = relevant_chunks
                logger.warning("No chunks passed quality filter, using all retrieved chunks")

            logger.info(f"Filtered to {len(high_quality_chunks)} high-quality chunks from {len(relevant_chunks)} total")

            # Build enhanced context string with clear source attribution
            context_parts = []
            sources = []
            current_length = 0

            for chunk in high_quality_chunks:
                chunk_text = chunk['text']
                chunk_title = chunk['metadata'].get('title', 'Unknown Document')
                course_code = chunk['metadata'].get('course_code', 'Unknown')
                similarity_score = chunk['similarity_score']

                # Check if adding this chunk would exceed limit
                if current_length + len(chunk_text) > max_context_length:
                    break

                # Format with simple, clean attribution
                formatted_chunk = f"[Source: {chunk_title} ({course_code})]\n{chunk_text}"

                context_parts.append(formatted_chunk)
                sources.append({
                    'title': chunk_title,
                    'course_code': course_code,
                    'similarity_score': similarity_score,
                    'chunk_index': chunk['metadata'].get('chunk_index', 0),
                    'confidence_level': 'high' if similarity_score >= 0.8 else 'medium' if similarity_score >= 0.6 else 'low'
                })

                current_length += len(chunk_text) + 100  # Account for formatting

            # Create simple context without complex formatting
            context = "\n\n".join(context_parts)

            return {
                'context': context,
                'sources': sources,
                'has_relevant_content': True,
                'total_chunks': len(relevant_chunks),
                'used_chunks': len(context_parts),
                'average_similarity': sum(s['similarity_score'] for s in sources) / len(sources) if sources else 0
            }

        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return {
                'context': '',
                'sources': [],
                'has_relevant_content': False
            }
   
    def _is_casual_conversation(self, query: str) -> bool:
        """Determine if query is casual conversation vs academic question"""
        query_lower = query.lower().strip()

        # Greetings and casual conversation patterns
        casual_patterns = [
            # Greetings
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'how\'s your day', 'how is your day', 'whats up', 'what\'s up',

            # Personal questions about the bot
            'tell me about yourself', 'who are you', 'what are you', 'introduce yourself',
            'what do you do', 'what can you do',

            # Casual chat
            'how\'s it going', 'how is it going', 'nice to meet you', 'pleasure to meet you',
            'thanks', 'thank you', 'goodbye', 'bye', 'see you later', 'talk to you later',

            # Weather/general small talk
            'how\'s the weather', 'nice day', 'beautiful day',

            # Single word greetings
            'morning', 'afternoon', 'evening'
        ]

        # Check if query starts with or contains casual patterns
        for pattern in casual_patterns:
            if query_lower.startswith(pattern) or query_lower == pattern:
                return True
            # For very short queries that are just greetings
            if len(query_lower.split()) <= 3 and pattern in query_lower:
                return True

        # Check for question marks with casual content
        if '?' in query_lower:
            casual_question_patterns = [
                'how are you', 'who are you', 'what are you', 'how\'s your day',
                'what do you do', 'tell me about yourself'
            ]
            for pattern in casual_question_patterns:
                if pattern in query_lower:
                    return True

        return False

    def _generate_casual_response(self, query: str, user_preferences: Dict[str, Any] = None) -> str:
        """Generate appropriate casual conversation responses"""
        query_lower = query.lower().strip()

        # Greetings
        if any(greeting in query_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return "Hello! I'm your Study Helper Agent. I'm here to help you with questions about your course materials, assignments, and academic content. How can I assist you with your studies today?"

        # How are you / How's your day
        if any(phrase in query_lower for phrase in ['how are you', 'how\'s your day', 'how is your day', 'how\'s it going']):
            return "I'm doing great, thank you for asking! I'm ready to help you with your studies. I can answer questions about your course materials, explain concepts, help with assignments, or provide information from your uploaded documents. What would you like to learn about today?"

        # About yourself / Who are you
        if any(phrase in query_lower for phrase in ['tell me about yourself', 'who are you', 'what are you', 'introduce yourself']):
            return """I'm your AI Study Helper Agent! Here's what I can do for you:

ACADEMIC SUPPORT:
- Answer questions about your course materials
- Explain concepts from your textbooks and lecture notes
- Help with assignments and homework
- Provide summaries of your documents

SMART SEARCH:
- Find relevant information from your uploaded materials
- Reference specific documents and sources
- Connect concepts across different courses

PERSONALIZED LEARNING:
- Adapt to your learning style and preferences
- Provide explanations at your preferred difficulty level
- Track your learning progress

Just ask me anything about your studies, and I'll search through your course materials to give you the most relevant, helpful answers!"""

        # What can you do
        if any(phrase in query_lower for phrase in ['what do you do', 'what can you do']):
            return """I can help you with your academic studies in several ways:

ANSWER QUESTIONS: Ask me about any topic from your course materials
SEARCH DOCUMENTS: I'll find relevant information from your uploaded files
EXPLAIN CONCEPTS: Get clear explanations tailored to your learning style
ASSIGNMENT HELP: Guidance based on your actual course content
STUDY SUPPORT: Summaries, examples, and step-by-step explanations

Try asking me something like:
- "Explain database normalization from my course materials"
- "What does my textbook say about data structures?"
- "Help me understand the assignment requirements"

What would you like to learn about?"""

        # Thanks
        if any(phrase in query_lower for phrase in ['thanks', 'thank you']):
            return "You're very welcome! I'm always here to help with your studies. Feel free to ask me any questions about your course materials anytime!"

        # Goodbye
        if any(phrase in query_lower for phrase in ['goodbye', 'bye', 'see you later', 'talk to you later']):
            return "Goodbye! Good luck with your studies. I'll be here whenever you need help with your course materials. Have a great day!"

        # Default casual response
        return "I'm your Study Helper Agent, ready to assist you with your academic questions! I can help you understand concepts from your course materials, find information in your documents, and support your learning. What would you like to study today?"

    def generate_rag_response(self, query: str, course_id: Optional[int] = None, user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate complete RAG response using LLM with retrieved context"""
        try:
            # Check if this is casual conversation that shouldn't use course materials
            if self._is_casual_conversation(query):
                logger.info(f"Detected casual conversation query: '{query[:50]}...' - responding without course materials")

                # Generate a friendly, personal response without course context
                casual_response = self._generate_casual_response(query, user_preferences)

                return {
                    'response': casual_response,
                    'context_used': False,
                    'sources': [],
                    'confidence': 'high',
                    'reason': 'casual_conversation'
                }

            # Generate context from relevant documents for academic questions
            context_data = self.generate_context(query, course_id)

            if not context_data['has_relevant_content']:
                # No relevant content found - provide simple guidance
                no_context_response = f"""I couldn't find specific information about "{query}" in your uploaded course materials.

This might be because:
- The topic hasn't been covered in uploaded materials yet
- Different terminology is used in your course materials
- The materials need to be processed (use /process_docs)
- Your course materials need to be synced (use /sync)

Suggestions:
- Try rephrasing your question with terms from your textbook
- Ask about topics from your enrolled courses
- Check if your instructor has uploaded relevant materials

I can provide general academic guidance, but for course-specific information, please ensure your materials are up to date."""

                return {
                    'response': no_context_response,
                    'context_used': False,
                    'sources': [],
                    'confidence': 'low',
                    'reason': 'no_relevant_context'
                }

            # Generate response using LLM with context
            response = self.llm_service.generate_response(
                query,
                context_data['context'],
                user_preferences
            )

            # Enhance response with source information
            enhanced_response = self._enhance_response_with_sources(response, context_data)

            return {
                'response': enhanced_response,
                'context_used': True,
                'sources': context_data['sources'],
                'chunks_retrieved': context_data['total_chunks'],
                'chunks_used': context_data['used_chunks'],
                'confidence': self._assess_response_confidence(context_data['sources']),
                'average_similarity': context_data.get('average_similarity', 0)
            }

        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            # Fallback to basic LLM response
            fallback_response = self.llm_service.generate_response(query, "", user_preferences)
            return {
                'response': fallback_response,
                'context_used': False,
                'sources': [],
                'confidence': 'low',
                'error': str(e)
            }
    
    def _enhance_response_with_sources(self, response: str, context_data: Dict[str, Any]) -> str:
        """Enhance LLM response with source attribution and context verification"""
        if not context_data.get('sources'):
            return response

        # Add source references to the response - simplified format to avoid Telegram parsing issues
        sources_section = "\n\nCourse Material Sources:\n"

        for i, source in enumerate(context_data['sources'][:3], 1):  # Show top 3 sources
            confidence_text = f"({source['similarity_score']*100:.0f}% match)"
            sources_section += f"{i}. {source['title']} ({source['course_code']}) {confidence_text}\n"

        if len(context_data['sources']) > 3:
            sources_section += f"...and {len(context_data['sources']) - 3} more sources\n"

        # Add quality indicator - simplified
        avg_similarity = context_data.get('average_similarity', 0)
        if avg_similarity >= 0.8:
            quality_note = "High confidence - Answer based on highly relevant course materials"
        elif avg_similarity >= 0.6:
            quality_note = "Medium confidence - Answer based on moderately relevant course materials"
        else:
            quality_note = "Lower confidence - Answer based on partially relevant course materials"

        # Combine response with source attribution - clean format
        enhanced_response = f"{response}\n{sources_section}\n{quality_note}"

        return enhanced_response

    def _assess_response_confidence(self, sources: List[Dict[str, Any]]) -> str:
        """Assess confidence level based on source quality"""
        if not sources:
            return 'low'

        avg_similarity = sum(source['similarity_score'] for source in sources) / len(sources)

        if avg_similarity >= 0.8:
            return 'high'
        elif avg_similarity >= 0.6:
            return 'medium'
        else:
            return 'low'
        
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
                'processing_rate': f"{(processed_docs/total_docs*100):.1f}%" if total_docs > 0 else "0%",
                'llm_service_available': self.llm_service is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                'total_embeddings': 0,
                'processed_documents': 0,
                'total_documents': 0,
                'processing_rate': "0%",
                'llm_service_available': False
            }