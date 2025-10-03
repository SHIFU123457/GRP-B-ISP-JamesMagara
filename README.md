[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/blswXyO9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20098947&assignment_repo_type=AssignmentRepo)
<artifact identifier="readme-study-helper" type="text/markdown" title="Study Helper Agent README">
# Study Helper Agent
An AI-powered Telegram bot that integrates with Learning Management Systems (LMS) to provide personalized academic support through intelligent document retrieval and natural language understanding.
Overview
Study Helper Agent automates academic assistance by connecting to your Google Classroom (and optionally Moodle), processing course materials, and providing context-aware responses to student queries. The bot uses Retrieval-Augmented Generation (RAG) with LLaMA language models to deliver accurate, source-grounded answers based on your actual course content.
Key Features
LMS Integration

Google Classroom: Per-user OAuth2 authentication for secure access to individual classrooms
Moodle: Server-wide API integration for institutional deployments
Automatic Sync: Scheduled synchronization of courses, assignments, and materials
Smart Notifications: Intelligent alerts for new materials with context-aware assistance prompts

AI-Powered Assistance

RAG Pipeline: Semantic search over course documents using vector embeddings
Context-Aware Responses: Answers grounded in your actual course materials with source citations
Multi-Format Support: PDF, DOCX, PPTX, and TXT document processing
Intelligent Chunking: Optimized text segmentation for better retrieval accuracy

Personalized Learning

Adaptive Responses: Tailors explanations based on learning style and difficulty preferences
Interactive Notifications: Context-sensitive help buttons for assignments, quizzes, and readings
User Profiles: Tracks preferences and interaction history
Session Analytics: Monitors engagement patterns for continuous improvement

Telegram Interface

Conversational AI: Natural language query processing
Command System: Easy-to-use commands for common tasks
Inline Buttons: Quick actions for course management and settings
Real-time Notifications: Instant alerts for new course materials

Core Components

Bot Handler (src/bot/handlers.py): Telegram message routing and command processing
LMS Integration (src/services/lms_integration.py): Google Classroom and Moodle connectors
RAG Pipeline (src/core/rag_pipeline.py): Document processing and semantic retrieval
Document Processor (src/services/document_processor.py): Multi-format text extraction
LLM Service (src/services/llm_integration.py): Hugging Face API integration
Scheduler (src/services/scheduler.py): Automated sync and notifications
OAuth Manager (src/services/oauth_manager.py): Google OAuth2 flow management

Usage
Bot Commands
/start : Initialize bot and create user profile
/help : Display help message with command list
/courses : View enrolled courses
/connect_classroom : classroomConnect Google Classroom account
/disconnect_classroom : Disconnect Google Classroom
/connections : View LMS connection status
/sync : Manually sync course materials
/process_docs : Force document processing
/status : View system status
/profile : View learning profile
/settings : Adjust preferences


Asking Questions
Simply type your question naturally:
"What are the main differences between stacks and queues?"
"Summarize today's lecture on database normalization"
"Help me with the Week 3 assignment"
The bot will:

Search your course materials
Retrieve relevant content
Generate a comprehensive answer with source citations
Provide interactive buttons for follow-up actions

Notifications
The bot automatically notifies you about:

New assignments with breakdown assistance
Quiz/test materials with study guides
Reading materials with summarization options

Each notification includes contextual help buttons tailored to the material type.