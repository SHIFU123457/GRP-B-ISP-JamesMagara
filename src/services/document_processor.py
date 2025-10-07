import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from docx import Document as DocxDocument
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

def safe_file_operation(operation_name: str):
    """Decorator for safe file operations with detailed error reporting"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                logger.error(f"{operation_name} failed - File not found: {e}")
                return None
            except PermissionError as e:
                logger.error(f"{operation_name} failed - Permission denied: {e}")
                return None
            except Exception as e:
                logger.error(f"{operation_name} failed - Unexpected error: {e}", exc_info=True)
                return None
        return wrapper
    return decorator

class DocumentProcessor:
    """Process documents and extract text content for RAG pipeline"""
    
    def __init__(self):
        self.chunk_size = 500  # words per chunk
        self.chunk_overlap = 50  # overlap between chunks
        
        # Download NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    @safe_file_operation("Text extraction")
    def extract_text(self, file_path: str, file_type: str) -> Optional[str]:
        """Extract text from various document formats"""
        try:
            if file_type.lower() == 'pdf':
                return self._extract_pdf_text(file_path)
            elif file_type.lower() in ['docx', 'doc']:
                return self._extract_docx_text(file_path)
            elif file_type.lower() == 'txt':
                return self._extract_txt_text(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return None
    
    @safe_file_operation("PDF text extraction")
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF files"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    @safe_file_operation("DOCX text extraction")
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from Word documents"""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    @safe_file_operation("TXT text extraction")
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from plain text files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep periods and commas
        import re
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for better retrieval"""
        # Clean the text first
        text = self.clean_text(text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            sentence_word_count = len(words)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_word_count = len(word_tokenize(' '.join(current_chunk)))
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
        
        return chunks
    
    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """Create a chunk object with text and metadata"""
        return {
            'text': text,
            'metadata': {
                **metadata,
                'chunk_index': chunk_index,
                'chunk_length': len(text.split()),
            }
        }
    
    def process_document(self, file_path: str, file_type: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Complete document processing pipeline"""
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        text = self.extract_text(file_path, file_type)
        if not text:
            logger.error(f"No text extracted from {file_path}")
            return []
        
        # Create chunks
        chunks = self.chunk_text(text, metadata)
        
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        return chunks