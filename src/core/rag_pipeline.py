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
from src.data.models import Document, DocumentEmbedding, DocumentAccess
from src.services.document_processor import DocumentProcessor
from src.services.llm_integration import LLMService
from src.services.adaptive_response_engine import adaptive_response_engine

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
        """Process a document and store its embeddings with batch commits to avoid timeouts"""
        BATCH_COMMIT_SIZE = 50  # Commit every 50 chunks to avoid Supabase connection timeouts

        try:
            # [FIX] Delete existing embeddings first to ensure idempotency (for reprocessing)
            with db_manager.get_session() as session:
                existing_count = session.query(DocumentEmbedding).filter(DocumentEmbedding.document_id == document_id).count()

                if existing_count > 0:
                    logger.info(f"Deleting {existing_count} existing embeddings for document ID: {document_id} (reprocessing)")
                    session.query(DocumentEmbedding).filter(DocumentEmbedding.document_id == document_id).delete(synchronize_session=False)
                    session.commit()
                    self.collection.delete(where={"document_id": document_id})
                    logger.info(f"Successfully deleted {existing_count} old entries for document ID: {document_id}")
                else:
                    logger.info(f"Processing new document ID: {document_id} (no existing embeddings)")

            # First, get document metadata in a quick session
            with db_manager.get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()

                if not document:
                    logger.error(f"Document with ID {document_id} not found")
                    return False

                if not document.file_path or not Path(document.file_path).exists():
                    logger.error(f"Document file not found: {document.file_path}")
                    return False

                # Mark as processing
                document.processing_status = "processing"
                session.commit()

                # Cache metadata before session closes
                doc_path = document.file_path
                doc_type = document.file_type
                doc_title = document.title
                doc_course_id = document.course_id
                doc_course_code = document.course.course_code if document.course else 'Unknown'

            # Process document into chunks (outside database session)
            # NOTE: user_id is NOT stored in metadata anymore - access control via DocumentAccess table
            metadata = {
                'document_id': document_id,
                'title': doc_title,
                'course_id': doc_course_id,
                'file_type': doc_type,
                'course_code': doc_course_code
            }

            chunks = self.document_processor.process_document(
                doc_path,
                doc_type,
                metadata
            )

            if not chunks:
                logger.error(f"No chunks created from document {document_id}")
                return False

            total_chunks = len(chunks)
            logger.info(f"Processing {total_chunks} chunks for document {document_id} in batches of {BATCH_COMMIT_SIZE}")

            # Process chunks in batches with separate sessions to avoid connection timeouts
            stored_count = 0
            for i in range(0, total_chunks, BATCH_COMMIT_SIZE):
                batch = chunks[i:i + BATCH_COMMIT_SIZE]
                batch_num = (i // BATCH_COMMIT_SIZE) + 1
                total_batches = (total_chunks + BATCH_COMMIT_SIZE - 1) // BATCH_COMMIT_SIZE

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks) for document {document_id}")

                # Use a fresh session for each batch
                with db_manager.get_session() as batch_session:
                    for chunk in batch:
                        if self._store_chunk_embedding(chunk, document_id, batch_session):
                            stored_count += 1

                    # Commit this batch
                    batch_session.commit()
                    logger.debug(f"✅ Batch {batch_num}/{total_batches} committed: {stored_count}/{total_chunks} chunks stored")

            # Final update: mark as completed in a final session
            with db_manager.get_session() as final_session:
                document = final_session.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.is_processed = True
                    document.processing_status = "completed"
                    document.content_text = chunks[0]['text'][:1000] + "..." if chunks else ""
                    final_session.commit()

            logger.info(f"✅ Successfully processed document {document_id}: {stored_count}/{total_chunks} chunks stored")
            return True

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            # Try to mark as failed
            try:
                with db_manager.get_session() as error_session:
                    document = error_session.query(Document).filter(Document.id == document_id).first()
                    if document:
                        document.processing_status = "failed"
                        error_session.commit()
            except Exception as mark_error:
                logger.error(f"Failed to mark document as failed: {mark_error}")
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
    def retrieve_relevant_chunks(self, query: str, user_id: int, course_id: Optional[int] = None, document_id: Optional[int] = None, top_k: int = None, min_similarity: Optional[float] = None) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a given query, filtered by user access.

        Args:
            query: Search query text
            user_id: User ID for access control filtering
            course_id: Optional course filter
            document_id: Optional document filter
            top_k: Number of results to retrieve
            min_similarity: Minimum similarity threshold (default 0.1). Set to 0 to get all chunks.
        """
        try:
            # Use configured top_k if not specified
            if top_k is None:
                top_k = settings.TOP_K_RETRIEVAL

            # Default minimum similarity threshold
            if min_similarity is None:
                min_similarity = 0.1  # 10% default threshold

            # Generate query embedding
            logger.info(f"Using embedding model: {self.embedding_model_name}")
            query_embedding = self.embedding_model.encode(query).tolist()
            logger.info(f"Query embedding shape: {len(query_embedding)}")
            logger.info(f"Query embedding sample: {query_embedding[:5]}")  # First 5 values

            # Get list of document IDs user has access to via DocumentAccess table
            with db_manager.get_session() as session:
                accessible_doc_ids = [
                    access.document_id
                    for access in session.query(DocumentAccess).filter(
                        DocumentAccess.user_id == user_id,
                        DocumentAccess.is_active == True
                    ).all()
                ]

            if not accessible_doc_ids:
                logger.warning(f"User {user_id} has no accessible documents")
                return []

            logger.info(f"User {user_id} has access to {len(accessible_doc_ids)} documents")

            # Prepare filter for document/course-specific search
            filter_conditions = []

            if document_id:
                # Verify user has access to this specific document
                if document_id not in accessible_doc_ids:
                    logger.warning(f"User {user_id} does not have access to document {document_id}")
                    return []
                filter_conditions.append({"document_id": document_id})
                logger.info(f"Filtering by document_id: {document_id} (min_similarity: {min_similarity})")
            else:
                # Filter by ALL accessible documents
                # ChromaDB uses $in operator for list filtering
                filter_conditions.append({"document_id": {"$in": accessible_doc_ids}})
                logger.info(f"Filtering by {len(accessible_doc_ids)} accessible documents")

            if course_id:
                # Additional course filter if specified
                filter_conditions.append({"course_id": course_id})
                logger.info(f"Additional filter by course_id: {course_id}")

            # Combine conditions using $and if there are multiple, otherwise use the single condition.
            where_filter = {"$and": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]

            # Search in ChromaDB with where filter
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter,
                    include=['documents', 'metadatas', 'distances']
                )
                if where_filter:
                    logger.info(f"ChromaDB query successful - filtered by {where_filter}")
                else:
                    logger.info(f"ChromaDB query successful - no filters applied")
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
                    logger.info(f"Chunk {i+1}: similarity={similarity_score:.3f}, threshold={min_similarity}")

                    # Apply similarity threshold (can be 0 for document-specific retrieval)
                    if similarity_score >= min_similarity:
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
    
    def generate_context(self, query: str, user_id: int, course_id: Optional[int] = None, document_id: Optional[int] = None, max_context_length: Optional[int] = None, min_similarity: Optional[float] = None) -> Dict[str, Any]:
        """Generate enhanced context for RAG-enhanced response with better source integration

        Args:
            query: Search query text
            user_id: User ID for filtering
            course_id: Optional course filter
            document_id: Optional document filter
            max_context_length: Maximum context length in characters
            min_similarity: Minimum similarity threshold (default 0.1). Use 0 for document-specific queries.
        """
        try:
            # Use configurable context length
            if max_context_length is None:
                max_context_length = settings.CONTEXT_MAX_LENGTH

            # Retrieve relevant chunks
            # For document-specific queries, use lower threshold to get more content
            relevant_chunks = self.retrieve_relevant_chunks(
                query,
                user_id,
                course_id,
                document_id,
                top_k=settings.TOP_K_RETRIEVAL,
                min_similarity=min_similarity
            )

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
   
    def _is_obvious_chitchat(self, query: str) -> bool:
        """Check if query is obvious chitchat/greetings that should skip RAG entirely"""
        query_lower = query.lower().strip()

        # Only skip RAG for very obvious greetings and basic chitchat
        chitchat_patterns = [
            # Greetings
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', "what's up", 'whats up', 'sup',

            # Basic questions about the bot
            'who are you', 'what are you', 'your name',

            # Thank you / goodbye
            'thank you', 'thanks', 'bye', 'goodbye', 'see you',
        ]

        # Check if query is just a greeting/chitchat
        for pattern in chitchat_patterns:
            if query_lower == pattern or query_lower.startswith(pattern + ' ') or query_lower.startswith(pattern + '?'):
                return True

        return False

    def _generate_non_educational_response(self, query: str, user_preferences: Dict[str, Any] = None) -> str:
        """Use LLM to generate natural response for non-educational queries"""
        try:
            # Use the LLM service to respond naturally without course materials
            response = self.llm_service.generate_response(query, "", user_preferences)
            return response
        except Exception as e:
            logger.error(f"Error generating non-educational response: {e}")
            # Simple fallback if LLM fails
            return "I'm here to help with your studies! Feel free to ask me about your course materials or any academic topics."

    def generate_rag_response(self, query: str, user_id: int, course_id: Optional[int] = None, document_id: Optional[int] = None, user_preferences: Dict[str, Any] = None, pre_extracted_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate complete RAG response using LLM with retrieved context

        Strategy: RAG-first approach
        1. Check for obvious non-educational queries (greetings, chitchat)
        2. Always attempt RAG retrieval for everything else
        3. If similarity < 10%, fall back to generic LLM

        Args:
            query: User's query text
            user_id: User database ID
            course_id: Optional course context
            document_id: Optional document context
            user_preferences: User preferences for personalization
            pre_extracted_topics: Optional pre-extracted topics to avoid redundant LLM calls
        """
        try:
            # Only skip RAG for obvious non-educational queries (greetings, chitchat)
            if self._is_obvious_chitchat(query):
                logger.info(f"Detected chitchat: '{query[:50]}...' - responding conversationally")

                # Use LLM to respond naturally to chitchat
                chitchat_response = self._generate_non_educational_response(query, user_preferences)

                return {
                    'response': chitchat_response,
                    'context_used': False,
                    'sources': [],
                    'confidence': 'high',
                    'reason': 'chitchat'
                }

            # For all other queries: Always try RAG first
            logger.info(f"Searching course materials for: '{query[:50]}...'")
            context_data = self.generate_context(query, user_id, course_id, document_id)

            # Check if we have relevant content based on similarity threshold
            avg_similarity = context_data.get('average_similarity', 0)
            FALLBACK_THRESHOLD = 0.10  # 10% similarity threshold

            if not context_data['has_relevant_content'] or avg_similarity < FALLBACK_THRESHOLD:
                # Similarity too low - fall back to generic LLM
                logger.info(f"Similarity {avg_similarity*100:.1f}% below threshold ({FALLBACK_THRESHOLD*100}%) - falling back to generic LLM")

                fallback_response = self._generate_non_educational_response(query, user_preferences)

                return {
                    'response': fallback_response,
                    'context_used': False,
                    'sources': [],
                    'confidence': 'low',
                    'reason': 'low_similarity_fallback'
                }

            # Use adaptive response engine to enhance prompt with personalization
            with db_manager.get_session() as session:
                try:
                    # Build base prompt with context
                    base_prompt = f"""You are an AI study assistant. Base your answer on the provided course materials.

COURSE MATERIALS:
{context_data['context'][:5000]}

STUDENT QUESTION: {query}

ANSWER:"""

                    # Get conversation history from session manager
                    from src.services.personalization_engine import session_manager
                    conversation_history = session_manager.get_conversation_history(
                        user_id=user_id,
                        db_session=session,
                        limit=3  # Last 3 conversation turns
                    )

                    # Enhance prompt with adaptive features (pass pre-extracted topics if available)
                    enhanced_prompt, metadata = adaptive_response_engine.analyze_and_enhance_prompt(
                        user_id=user_id,
                        query=query,
                        base_prompt=base_prompt,
                        session=session,
                        conversation_history=conversation_history,
                        pre_extracted_topics=pre_extracted_topics
                    )

                    logger.info(f"Adaptive response metadata: verbosity={metadata.get('verbosity_preference', 'medium')}, "
                               f"style={metadata.get('explanation_style', 'adaptive')}, "
                               f"sentiment={metadata.get('sentiment', {}).get('detected_patterns', [])}, "
                               f"topics={metadata.get('current_topics', [])}")

                except Exception as adapt_error:
                    logger.warning(f"Adaptive engine failed, using standard prompt: {adapt_error}")
                    enhanced_prompt = None

            # Generate response using LLM with enhanced prompt
            response = self.llm_service.generate_response(
                query,
                context_data['context'],
                user_preferences,
                enhanced_prompt=enhanced_prompt  # Pass enhanced prompt
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

        # Log detailed source information for developers
        avg_similarity = context_data.get('average_similarity', 0)
        logger.info(f"Response confidence: {avg_similarity*100:.1f}% average similarity")
        logger.info(f"Sources used ({len(context_data['sources'])} total):")
        for i, source in enumerate(context_data['sources'], 1):
            logger.info(f"  {i}. {source['title']} ({source['course_code']}) - {source['similarity_score']*100:.0f}% match")

        # User-facing source section (clean, no percentages or confidence levels)
        sources_section = "\n\n**📚 Sources from your course materials:**"

        for i, source in enumerate(context_data['sources'][:3], 1):  # Show top 3 sources
            sources_section += f"\n{i}. {source['title']} ({source['course_code']})"

        if len(context_data['sources']) > 3:
            sources_section += f"\n*...and {len(context_data['sources']) - 3} more sources*"

        # Combine response with clean source attribution
        enhanced_response = f"{response}{sources_section}"

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
        
    def generate_quiz_questions(
        self,
        user_id: int,
        document_id: Optional[int] = None,
        topic: Optional[str] = None,
        num_questions: int = 5,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """Generate quiz questions from document, topic, or conversation using RAG and LLM

        Args:
            user_id: ID of the user requesting quiz questions (for document filtering)
            document_id: Optional specific document to generate questions from
            topic: Optional topic to search for and generate questions about
            num_questions: Number of questions to generate (default: 5)
            conversation_history: Optional conversation history for context-based quizzes

        Returns list of question objects:
        [
            {
                'question': 'What is...?',
                'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                'correct_answer_index': 0,
                'explanation': 'The correct answer is A because...'
            },
            ...
        ]
        """
        try:
            # If conversation history is provided, use it as primary context
            if conversation_history:
                logger.info(f"Generating quiz from conversation history ({len(conversation_history)} turns)")

                # Build context from conversation
                context_parts = []
                for turn in conversation_history:
                    context_parts.append(f"Q: {turn['query']}")
                    context_parts.append(f"A: {turn['response'][:500]}")  # Truncate long responses

                context = "\n\n".join(context_parts)
                context_data = {
                    'context': context,
                    'has_relevant_content': True,
                    'sources': [{'title': 'Conversation History'}]
                }
                source_description = f"conversation about '{topic}'"

            # Retrieve relevant content from documents
            elif document_id:
                # Get content from specific document
                # Use min_similarity=0 to get ALL chunks from the document regardless of semantic match
                # This is important because quiz queries are meta-instructions, not content queries
                # Request up to 50 chunks to get comprehensive coverage of the document
                query = f"Generate comprehensive quiz questions covering the key concepts from this document"

                # First get chunks with no similarity filtering
                relevant_chunks = self.retrieve_relevant_chunks(
                    query,
                    user_id=user_id,
                    document_id=document_id,
                    top_k=50,  # Get more chunks for comprehensive quiz
                    min_similarity=0.0  # Accept all chunks from the document
                )

                if not relevant_chunks:
                    logger.warning(f"No chunks found for document ID {document_id}")
                    return []

                # Build context from chunks
                context_parts = []
                current_length = 0
                max_length = 4000  # Context limit for quiz generation

                for chunk in relevant_chunks:
                    chunk_text = chunk['text']
                    if current_length + len(chunk_text) > max_length:
                        break
                    context_parts.append(chunk_text)
                    current_length += len(chunk_text)

                context = "\n\n".join(context_parts)
                context_data = {
                    'context': context,
                    'has_relevant_content': len(context_parts) > 0,
                    'sources': [{'title': chunk['metadata'].get('title', 'Unknown')} for chunk in relevant_chunks[:3]]
                }

                source_description = f"document ID {document_id}"
                logger.info(f"Generated quiz context from {len(context_parts)} chunks ({current_length} chars) for {source_description}")

            elif topic:
                # Get content from topic search (use default similarity threshold)
                query = f"Generate quiz questions about {topic}"
                context_data = self.generate_context(query, user_id=user_id)
                source_description = f"topic '{topic}'"
            else:
                logger.error("Either document_id or topic must be provided")
                return []

            if not context_data.get('has_relevant_content'):
                logger.warning(f"No relevant content found for {source_description}")
                return []

            context = context_data['context']

            # Create prompt for question generation
            prompt = f"""Based on the following course materials, generate {num_questions} multiple-choice quiz questions to test student understanding.

COURSE MATERIALS:
{context[:4000]}

REQUIREMENTS:
1. Generate exactly {num_questions} multiple-choice questions
2. Each question should have 4 options (A, B, C, D)
3. Only ONE option should be correct
4. Include a brief explanation for why the correct answer is right
5. Questions should test understanding, not just memorization
6. Focus on key concepts and important details from the material

FORMAT YOUR RESPONSE EXACTLY LIKE THIS (do not deviate from this format):
QUESTION 1: [Your question here]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]
CORRECT: [A/B/C/D]
EXPLANATION: [Why this answer is correct]

QUESTION 2: [Your question here]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]
CORRECT: [A/B/C/D]
EXPLANATION: [Why this answer is correct]

[Continue for all {num_questions} questions...]

Now generate the questions:"""

            # Use LLM to generate questions
            llm_response = self.llm_service.generate_response(
                query="Generate quiz questions",
                context=prompt,
                user_preferences=None
            )

            # Log the raw LLM response for debugging
            logger.info(f"Raw LLM response (first 500 chars): {llm_response[:500]}")

            # Parse the LLM response into structured questions
            questions = self._parse_quiz_questions(llm_response)

            # If parsing failed completely, try a simpler prompt format
            if len(questions) == 0:
                logger.warning("First attempt failed. Trying simplified prompt format...")
                simplified_prompt = f"""Based on this material, create {num_questions} multiple choice questions.

MATERIAL:
{context[:3000]}

For each question, use this EXACT format:

1. [Question text here]
A. [Option 1]
B. [Option 2]
C. [Option 3]
D. [Option 4]
CORRECT: [A/B/C/D]
EXPLANATION: [Why this is correct]

2. [Question text here]
A. [Option 1]
B. [Option 2]
C. [Option 3]
D. [Option 4]
CORRECT: [A/B/C/D]
EXPLANATION: [Why this is correct]

Generate {num_questions} questions now:"""

                llm_response = self.llm_service.generate_response(
                    query="Generate quiz questions",
                    context=simplified_prompt,
                    user_preferences=None
                )
                logger.info(f"Simplified LLM response (first 500 chars): {llm_response[:500]}")
                questions = self._parse_quiz_questions(llm_response)

            if len(questions) < num_questions:
                logger.warning(f"Generated {len(questions)} questions, expected {num_questions}")
                if len(questions) == 0:
                    logger.error(f"LLM failed to generate parseable questions. Final response: {llm_response[:1000]}")

            logger.info(f"Successfully generated {len(questions)} quiz questions for {source_description}")
            return questions

        except Exception as e:
            logger.error(f"Error generating quiz questions: {e}")
            return []

    def _parse_quiz_questions(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured question objects - more robust parsing"""
        questions = []

        try:
            import re

            # Try multiple parsing strategies

            # Strategy 1: Standard format with "QUESTION N:"
            question_blocks = re.split(r'(?:QUESTION|Question)\s*\d+\s*:', llm_response, flags=re.IGNORECASE)[1:]

            if not question_blocks or len(question_blocks) < 2:
                # Strategy 2: Split by numbered pattern (works for inline format)
                # First, add newlines before numbers followed by periods to separate questions
                normalized = re.sub(r'(\d+)\.\s+', r'\n\1. ', llm_response)

                # Extract question blocks by number  pattern
                question_pattern = r'(\d+\.\s+.+?(?=\d+\.\s+|$))'
                potential_blocks = re.findall(question_pattern, normalized, re.DOTALL)

                # Filter blocks that look like actual questions (have options A, B, C, D)
                question_blocks = []
                for block in potential_blocks:
                    # Check if block has all required components
                    has_options = all(re.search(rf'{letter}[\.\)]', block, re.IGNORECASE) for letter in ['A', 'B', 'C', 'D'])
                    has_explanation = re.search(r'(CORRECT|EXPLANATION|Answer)', block, re.IGNORECASE)

                    if has_options and has_explanation:
                        question_blocks.append(block.strip())

                logger.info(f"Strategy 2: Found {len(question_blocks)} blocks using numbered format")

            if not question_blocks:
                logger.warning("Could not split LLM response into question blocks")
                logger.debug(f"Full LLM response for debugging: {llm_response}")
                return []

            logger.info(f"Successfully split into {len(question_blocks)} question blocks")

            for block_idx, block in enumerate(question_blocks):
                try:
                    # CRITICAL FIX: LLM often puts everything on one line
                    # Add newlines before options (A., B., C., D.) and keywords (CORRECT, EXPLANATION)
                    block = re.sub(r'\s+([A-D][\.\)])', r'\n\1', block)  # Add newline before A. B. C. D.
                    block = re.sub(r'\s+(CORRECT|Correct answer):', r'\n\1:', block, flags=re.IGNORECASE)
                    block = re.sub(r'\s+(EXPLANATION|Explanation):', r'\n\1:', block, flags=re.IGNORECASE)

                    lines = [line.strip() for line in block.strip().split('\n') if line.strip()]

                    if len(lines) < 5:  # Minimum: question + 4 options + correct (explanation optional)
                        logger.warning(f"Block {block_idx} has too few lines ({len(lines)}). Block: {block[:200]}")
                        continue

                    # Extract question text (first non-empty line, remove number prefix if present)
                    question_text = lines[0]
                    # Remove leading number like "1. " from question text
                    question_text = re.sub(r'^\d+\.\s*', '', question_text)

                    # Extract options - be more flexible with formats
                    options = []
                    # Match patterns like "A)", "A.", "A:", "a)", etc.
                    option_pattern = re.compile(r'^([A-Da-d])[\)\.\:]\s*(.+)', re.IGNORECASE)

                    for line in lines[1:]:
                        match = option_pattern.match(line)
                        if match:
                            options.append(match.group(2).strip())
                        if len(options) == 4:
                            break

                    if len(options) != 4:
                        logger.warning(f"Block {block_idx}: Found {len(options)} options (expected 4). Lines: {lines[:10]}")
                        continue

                    # Extract correct answer - try multiple formats
                    # Search in the entire block text, not line by line
                    block_text = '\n'.join(lines)
                    correct_line = None
                    correct_patterns = [
                        r'CORRECT\s*:?\s*([A-D])',  # CORRECT: A
                        r'Correct\s+answer\s*:?\s*([A-D])',  # Correct answer: A
                        r'Answer\s*:?\s*([A-D])',  # Answer: A
                        r'EXPLANATION\s*:?\s*([A-D])[\.\)]',  # EXPLANATION: B. (answer at start of explanation)
                        r'answer\s+is\s+([A-D])[\.\)]',  # "The answer is B."
                        r'option\s+([A-D])',  # "option B"
                    ]

                    for pattern in correct_patterns:
                        match = re.search(pattern, block_text, re.IGNORECASE)
                        if match:
                            correct_line = match.group(1).upper()
                            logger.debug(f"Found correct answer '{correct_line}' using pattern: {pattern}")
                            break

                    if not correct_line:
                        logger.warning(f"Block {block_idx}: No correct answer found. Block preview: {block_text[:300]}")
                        # Try to infer from explanation text which option is mentioned
                        for letter in ['A', 'B', 'C', 'D']:
                            # Look for patterns like "B is correct" or "choose B"
                            if re.search(rf'\b{letter}\b.*?correct', block_text, re.IGNORECASE) or \
                               re.search(rf'choose\s+{letter}', block_text, re.IGNORECASE):
                                correct_line = letter
                                logger.info(f"Inferred correct answer: {letter}")
                                break

                    if not correct_line:
                        # Last resort: default to B (middle option)
                        correct_line = 'B'
                        logger.warning(f"Could not find correct answer, defaulting to B")

                    correct_index = ord(correct_line) - ord('A') if correct_line in 'ABCD' else 0

                    # Extract explanation - try multiple formats
                    explanation = "No explanation provided"
                    explanation_patterns = [
                        r'EXPLANATION\s*:?\s*(.+)',
                        r'Explanation\s*:?\s*(.+)',
                        r'Because\s+(.+)',
                    ]

                    for line in lines:
                        for pattern in explanation_patterns:
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match:
                                explanation = match.group(1).strip()
                                break
                        if explanation != "No explanation provided":
                            break

                    questions.append({
                        'question': question_text,
                        'options': options,
                        'correct_answer_index': correct_index,
                        'explanation': explanation
                    })

                    logger.debug(f"Successfully parsed question {len(questions)}: {question_text[:50]}...")

                except Exception as parse_error:
                    logger.warning(f"Error parsing question block {block_idx}: {parse_error}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing quiz questions from LLM response: {e}")

        return questions

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