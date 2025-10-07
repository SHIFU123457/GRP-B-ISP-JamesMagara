# src/services/llm_service.py
import logging
import requests
import time
from typing import Optional, Dict, Any
from functools import wraps
import json
import os

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available. Install with: pip install huggingface_hub")

from config.settings import settings

logger = logging.getLogger(__name__)

def retry_on_rate_limit(max_retries: int = 3, base_delay: float = 2.0):
    """Decorator to handle rate limiting and retries"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Handle rate limiting
                    if 'rate limit' in error_msg or 'quota' in error_msg:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {delay}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(delay)
                    elif attempt < max_retries - 1:
                        time.sleep(1)  # Short delay for other errors
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator

class LLMService:
    """Service for interacting with Language Learning Models"""
    
    def __init__(self):
        # Models available in Inference Providers
        self.chat_models = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "microsoft/DialoGPT-medium",
            "HuggingFaceH4/zephyr-7b-beta"
        ]

        # Fallback models for old API
        self.model_options = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "distilgpt2",
            "gpt2",
        ]

        self.hf_api_url = "https://api-inference.huggingface.co/models"
        self.hf_token = settings.HUGGINGFACE_API_TOKEN if hasattr(settings, 'HUGGINGFACE_API_TOKEN') else None

        # Initialize inference client if available
        self.inference_client = None
        if HF_HUB_AVAILABLE and self.hf_token:
            try:
                self.inference_client = InferenceClient(token=self.hf_token)
            except Exception as e:
                logger.warning(f"Failed to initialize inference client: {e}")

        self.use_local_fallback = False

        # Initialize the service
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Hugging Face API"""
        try:
            if not self.hf_token:
                logger.warning("No Hugging Face API token found. API access may be limited.")

            # Test inference client first
            if self.inference_client:
                try:
                    test_response = self._query_chat_completion("Hello")
                    if test_response:
                        logger.info("Inference client chat completion is working")
                        return
                except Exception as e:
                    logger.warning(f"Inference client test failed: {str(e)[:100]}")
            else:
                logger.info("Inference client not available (no token or library missing)")

            # Test each model to find working ones
            working_models = []
            for model in self.model_options:
                try:
                    test_response = self._query_huggingface(
                        "Hello",
                        model_name=model
                    )

                    if test_response:
                        working_models.append(model)
                        logger.info(f"Model {model} is working")
                        break  # Stop at first working model
                    else:
                        logger.warning(f"Model {model} not responding properly")
                except Exception as e:
                    logger.warning(f"Model {model} failed test: {str(e)[:100]}")

            if working_models:
                logger.info(f"Hugging Face API connection successful with {len(working_models)} working model(s)")
                # Reorder models to put working ones first
                self.model_options = working_models + [m for m in self.model_options if m not in working_models]
            else:
                logger.warning("No working models found, will use fallback responses")
        except Exception as e:
            logger.error(f"LLM service initialization failed: {e}")
    
    @retry_on_rate_limit(max_retries=3, base_delay=2.0)
    def _query_huggingface(self, prompt: str, model_name: str, headers: Dict = None) -> Optional[str]:
        """Query Hugging Face inference API"""
        try:
            url = f"{self.hf_api_url}/{model_name}"
            
            if headers is None:
                headers = {}
            
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            
            # Adjust payload based on model type
            if "TinyLlama" in model_name:
                # For instruction-tuned chat models
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 150,
                        "temperature": 0.7,
                        "do_sample": True,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1
                    }
                }
            else:
                # For GPT-style models (gpt2, distilgpt2)
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": 256,
                        "temperature": 0.7,
                        "do_sample": True,
                        "top_p": 0.9
                    }
                }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                logger.info(f"Model {model_name} is loading, waiting...")
                time.sleep(10)
                response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            response.raise_for_status()
            result = response.json()

            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    # Check for common response fields
                    if 'generated_text' in result[0]:
                        return result[0]['generated_text']
                    elif 'answer' in result[0]:
                        return result[0]['answer']
                    elif 'text' in result[0]:
                        return result[0]['text']
                elif isinstance(result[0], str):
                    return result[0]
            elif isinstance(result, dict):
                # Single response object
                if 'generated_text' in result:
                    return result['generated_text']
                elif 'answer' in result:
                    return result['answer']
                elif 'text' in result:
                    return result['text']

            return None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Model {model_name} not found in Inference API")
            elif e.response.status_code == 429:
                logger.warning(f"Rate limited for model {model_name}")
            else:
                logger.error(f"HTTP error for model {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Hugging Face API query failed for {model_name}: {e}")
            return None

    def _query_chat_completion(self, prompt: str) -> Optional[str]:
        """Query using the InferenceClient chat completion"""
        if not self.inference_client:
            return None

        for model in self.chat_models:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful AI study assistant."},
                    {"role": "user", "content": prompt}
                ]

                response = self.inference_client.chat_completion(
                    messages=messages,
                    model=model,
                    max_tokens=150,
                    temperature=0.7
                )

                if response and response.choices:
                    content = response.choices[0].message.content
                    if content and len(content.strip()) > 10:
                        logger.info(f"Successfully used model: {model}")
                        return content.strip()

            except Exception as e:
                logger.warning(f"Model {model} failed: {str(e)[:100]}")
                continue

        return None
    
    def generate_response(self, query: str, context: str = "", user_preferences: Dict[str, Any] = None) -> str:
        """Generate response using available LLM with enhanced context integration"""
        try:
            # Prepare enhanced prompt with context
            if context.strip():
                # Use more context and make it more directive
                prompt = f"""You are an AI study assistant. IMPORTANT: Base your answer EXCLUSIVELY on the provided course materials below. Do NOT use external knowledge.

COURSE MATERIALS:
{context[:5000]}

STUDENT QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the course materials provided above
2. Quote specific parts of the course materials when relevant
3. Provide comprehensive, detailed explanations using the course content
4. Reference the source material directly in your answer
5. Give step-by-step explanations when the materials contain procedures
6. Include examples from the course materials when available
7. Be thorough and educational in your response

ANSWER:"""
            else:
                # Enhanced general prompt
                prompt = f"""You are an AI study assistant helping a student. Since no specific course materials are provided, give a helpful general academic response.

STUDENT QUESTION: {query}

INSTRUCTIONS:
1. Provide a clear, educational response
2. Suggest where the student might find more specific information (course materials, textbooks, etc.)
3. Acknowledge that you're providing general guidance without access to their specific course content

ANSWER:"""
            
            # Try chat completion API first
            try:
                response = self._query_chat_completion(prompt)
                if response:
                    cleaned_response = self._clean_response(response, prompt)
                    if len(cleaned_response.strip()) > 10:
                        return cleaned_response
            except Exception as e:
                logger.warning(f"Chat completion API failed: {e}")

            # Fallback to old model API
            for model_name in self.model_options:
                try:
                    response = self._query_huggingface(prompt, model_name)
                    if response:
                        # Clean up the response
                        cleaned_response = self._clean_response(response, prompt)
                        if len(cleaned_response.strip()) > 10:  # Ensure we got a meaningful response
                            return cleaned_response
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
                    continue
            
            # If all models fail, provide contextual fallback
            return self._generate_fallback_response(query, context)
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(query, context)
    
    def _clean_response(self, response: str, original_prompt: str) -> str:
        """Clean up the model response"""
        # Remove the original prompt if it's repeated
        if original_prompt in response:
            response = response.replace(original_prompt, "").strip()
        
        # Extract answer after "Answer:" if present
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        # Limit response length
        if len(response) > 500:
            # Try to end at a complete sentence
            sentences = response.split('. ')
            truncated = []
            char_count = 0
            
            for sentence in sentences:
                if char_count + len(sentence) > 450:
                    break
                truncated.append(sentence)
                char_count += len(sentence) + 2
            
            response = '. '.join(truncated)
            if not response.endswith('.'):
                response += '.'
        
        return response
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Enhanced fallback response when LLM is unavailable"""
        query_lower = query.lower()

        if context.strip():
            # We have context from RAG, provide a comprehensive structured response
            return f"""**Based on your course materials:**

{context[:800]}

**Direct Answer to "{query}":**
The course materials above contain relevant information that should help answer your question. Key points to focus on are the concepts and definitions mentioned in the material.

**For more specific guidance:**
- Review the complete sections these excerpts come from
- Look for related examples or practice problems
- Use /sync to ensure you have the latest course materials

*Note: Enhanced AI responses temporarily unavailable. This response is based directly on your uploaded course content.*"""
        
        # No context available, provide general guidance
        if any(word in query_lower for word in ['what is', 'define', 'explain']):
            return f"""I understand you're asking about the concept in your query. While I'm currently operating in basic mode, I recommend:

1. Checking your course materials for definitions and explanations
2. Looking for related examples in your textbooks
3. Using /sync to ensure all your course materials are up to date

Use more specific terms from your course content for better results."""
        
        elif any(word in query_lower for word in ['how to', 'how do', 'steps']):
            return f"""For procedural questions like yours, I suggest:

1. Reviewing step-by-step examples in your course materials
2. Checking if there are practice problems or tutorials available
3. Looking for similar solved examples

Try rephrasing your question with specific terms from your coursework."""
        
        else:
            return f"""I received your question about: "{query}"

While enhanced AI processing is temporarily unavailable, you can:
- Use /sync to update your course materials
- Ask more specific questions using terms from your textbooks
- Try breaking complex questions into smaller parts

Your question has been logged for when full AI capabilities are restored."""

# Update settings.py to include Hugging Face configuration
class Settings:
    # ... existing settings ...
    
    # LLM Configuration
    HUGGINGFACE_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")
    USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "False").lower() == "true"
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    
    # Fallback configuration
    ENABLE_LLM_FALLBACK: bool = os.getenv("ENABLE_LLM_FALLBACK", "True").lower() == "true"