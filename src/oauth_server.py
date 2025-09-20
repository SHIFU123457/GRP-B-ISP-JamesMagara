from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any
import os
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from src.services.oauth_manager import UserOAuthManager, oauth_manager as shared_oauth_manager

logger = logging.getLogger(__name__)

# Use shared OAuth manager configured via settings
oauth_manager = shared_oauth_manager

app = FastAPI(title="Study Helper Agent OAuth")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/oauth/callback")
async def oauth_callback(request: Request):
    """Handle Google OAuth callback"""
    try:
        # Debug: Log all incoming parameters
        logger.info(f"OAuth callback received from {request.client.host}")
        logger.info(f"Full URL: {request.url}")
        logger.info(f"Query params: {dict(request.query_params)}")

        state = request.query_params.get('state')
        code = request.query_params.get('code')
        error = request.query_params.get('error')

        if error:
            logger.error(f"OAuth error: {error}")
            return HTMLResponse(
                content=f"""
                <html>
                <head><title>Authentication Error</title></head>
                <body>
                    <h1>Authentication Failed</h1>
                    <p>Error: {error}</p>
                    <p>Please return to Telegram and try again.</p>
                </body>
                </html>
                """,
                status_code=400
            )
        
        if not state or not code:
            logger.error(f"Missing OAuth parameters - state: {bool(state)}, code: {bool(code)}")
            return HTMLResponse(
                content="""
                <html>
                <head><title>Authentication Error</title></head>
                <body>
                    <h1>Authentication Failed</h1>
                    <p>Missing required parameters.</p>
                    <p>Please return to Telegram and try again.</p>
                    <p><small>If this issue persists, please contact support.</small></p>
                </body>
                </html>
                """,
                status_code=400
            )
        
        # Log the received parameters for debugging
        logger.info(f"OAuth callback received - state: {state[:20]}..., code: {code[:20]}...")
        
        # Complete OAuth flow
        try:
            logger.info(f"Attempting OAuth flow completion for state: {state[:20]}...")
            success = oauth_manager.complete_oauth_flow(state, code)
            logger.info(f"OAuth flow completion result: {success}")
        except Exception as oauth_error:
            logger.error(f"Exception during OAuth flow completion: {oauth_error}", exc_info=True)
            success = False

            # Return more specific error for debugging
            return HTMLResponse(
                content=f"""
                <html>
                <head><title>Authentication Error</title></head>
                <body>
                    <h1>Authentication Processing Failed</h1>
                    <p>An error occurred while processing your authentication.</p>
                    <p>Error details: {str(oauth_error)[:200]}</p>
                    <p>Please return to Telegram and try again.</p>
                </body>
                </html>
                """,
                status_code=500
            )
        
        if success:
            logger.info(f"OAuth flow completed successfully for state: {state}")
            return HTMLResponse(
                content="""
                <html>
                <head>
                    <title>Authentication Successful</title>
                    <style>
                        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                        .success { color: green; }
                        .container { max-width: 500px; margin: 0 auto; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1 class="success">Google Classroom Connected!</h1>
                        <p>Your Google Classroom has been successfully connected to Study Helper Agent.</p>
                        <p><strong>You can now close this window and return to Telegram.</strong></p>
                        <p>Use the /courses command to see your connected classrooms.</p>
                    </div>
                    <script>
                        // Auto-close after 3 seconds
                        setTimeout(function() {
                            window.close();
                        }, 3000);
                    </script>
                </body>
                </html>
                """
            )
        else:
            logger.error(f"Failed to complete OAuth flow for state: {state}")
            return HTMLResponse(
                content="""
                <html>
                <head><title>Authentication Error</title></head>
                <body>
                    <h1>Authentication Failed</h1>
                    <p>Failed to complete the authentication process.</p>
                    <p>Please return to Telegram and try again.</p>
                </body>
                </html>
                """,
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return HTMLResponse(
            content=f"""
            <html>
            <head><title>System Error</title></head>
            <body>
                <h1>System Error</h1>
                <p>An unexpected error occurred during authentication.</p>
                <p>Please contact support if the issue persists.</p>
            </body>
            </html>
            """,
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "oauth"}

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return HTMLResponse(
        content="""
        <html>
        <head><title>Study Helper Agent OAuth Service</title></head>
        <body>
            <h1>Study Helper Agent OAuth Service</h1>
            <p>This service handles Google Classroom authentication.</p>
            <p>Please use the Telegram bot to initiate authentication.</p>
            <p>Status: Server is running correctly</p>
        </body>
        </html>
        """
    )

@app.get("/test")
async def test_endpoint(request: Request):
    """Test endpoint to verify server is working"""
    return {
        "status": "OK",
        "url": str(request.url),
        "client": request.client.host if request.client else None,
        "query_params": dict(request.query_params)
    }

def validate_oauth_setup():
    """Validate OAuth configuration on startup"""
    try:
        from pathlib import Path

        credentials_path = settings.GOOGLE_CLASSROOM_CREDENTIALS
        if not credentials_path:
            logger.error("GOOGLE_CLASSROOM_CREDENTIALS not configured")
            return False

        if not Path(credentials_path).exists():
            logger.error(f"Google Classroom credentials file not found: {credentials_path}")
            return False

        # Test OAuth manager initialization
        oauth_manager
        logger.info("✅ OAuth setup validation passed")
        return True

    except Exception as e:
        logger.error(f"❌ OAuth setup validation failed: {e}")
        return False

if __name__ == "__main__":
    import uvicorn

    # Validate setup before starting
    if not validate_oauth_setup():
        logger.error("OAuth setup validation failed. Please check configuration.")
        sys.exit(1)

    port = int(os.getenv("OAUTH_PORT", "8080"))
    logger.info(f"Starting OAuth server on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )