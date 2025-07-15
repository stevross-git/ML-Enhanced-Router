# ML Query Router

## Overview

This is a Flask-based ML-enhanced query routing system that intelligently routes user queries to appropriate AI agents based on query classification and analysis. The system uses machine learning models (DistilBERT) for query categorization and maintains a registry of specialized agents to handle different types of queries.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (July 15, 2025)

✓ Added comprehensive AI model integration supporting 10+ providers
✓ Implemented AI model management interface with selection and configuration
✓ Created authentication system with API key management and JWT tokens
✓ Added support for OpenAI, Anthropic, Google, xAI, Perplexity, local Ollama, and custom endpoints
✓ Built AI model testing and validation functionality
✓ Integrated API key status monitoring across all providers
✓ Added model activation, configuration, and cost tracking features
✓ Created dedicated Settings page for enterprise API key management
✓ Built comprehensive Configuration page for advanced model and routing settings
✓ Added export/import functionality for configuration backup and deployment
✓ Implemented AI response caching system with database backend (SQLite/PostgreSQL)
✓ Added intelligent cache management with TTL, hit counting, and automatic cleanup
✓ Integrated cache functionality into AI model manager for improved performance
✓ Created cache management interface in Settings page with statistics and controls
✓ Replaced file-based cache with database models (AICacheEntry, AICacheStats)
✓ Added comprehensive cache endpoints for statistics, entries, and management
✓ Implemented database-backed cache with hit tracking and expiration management
✓ Integrated RAG (Retrieval-Augmented Generation) system with ChromaDB
✓ Added document upload support (PDF, DOCX, TXT, MD, HTML, JSON, CSV)
✓ Implemented document chunking and vector embedding for semantic search
✓ Created RAG management interface in chat sidebar with upload and statistics
✓ Added RAG panel modal for comprehensive document management
✓ Integrated RAG context enhancement for AI responses with document search
✓ Built RAG API endpoints for document upload, management, and search
✓ Fixed database model metadata conflicts with SQLAlchemy reserved attributes
✓ Created comprehensive Swagger/OpenAPI 3.0 documentation for all API endpoints
✓ Implemented interactive API documentation interface with Swagger UI
✓ Added API documentation navigation link and dark theme compatibility
✓ Fixed missing error templates (404.html, 500.html) for proper error handling
✓ Implemented shared memory system for real-time AI collaboration
✓ Created collaborative router with specialized AI agents (Analyst, Creative, Technical, Researcher, Synthesizer)
✓ Added collaborative AI interface with real-time session monitoring and shared context
✓ Integrated shared memory with agent scratchpads and working memory for context persistence
✓ Built collaborative API endpoints for multi-agent query processing and session management
✓ Enhanced collaborative AI with user-selectable AI models for each specialized agent
✓ Added agent configuration interface for model selection and management
✓ Implemented manual agent selection with automatic fallback for optimal collaboration
✓ Created comprehensive agent management with model switching and configuration tracking
✓ Updated comprehensive README.md with full system documentation and usage guide
✓ Enhanced Swagger/OpenAPI documentation with all collaborative AI endpoints
✓ Added schema definitions for collaborative results, sessions, and agent configurations
✓ Documented all new API endpoints with examples and parameter descriptions
✓ Implemented external LLM integration system with ExternalLLMManager for complex query processing
✓ Added query complexity analysis with automatic provider selection based on query characteristics
✓ Integrated external LLM capabilities with collaborative AI router and shared memory system
✓ Created API endpoints for external LLM processing, provider management, and metrics monitoring
✓ Enhanced MessageType enum with EXTERNAL_LLM_RESPONSE for tracking external LLM interactions
✓ Added comprehensive Swagger documentation for all external LLM integration endpoints
✓ Built comprehensive dashboard with 10 key metrics including collaborative sessions, external LLM calls, cache performance, and RAG documents
✓ Created complete network deployment package with Docker, Kubernetes, and direct server installation options
✓ Added Docker Compose configuration with PostgreSQL, Redis, and Nginx for production deployment
✓ Implemented Kubernetes manifests with secrets management, persistent volumes, and service configuration
✓ Created automated setup script for easy network deployment with multiple deployment options
✓ Added comprehensive deployment documentation with security, monitoring, and scaling considerations
✓ Built systemd service configuration for Linux server deployment
✓ Implemented environment variable templates and configuration management
✓ Added SSL/HTTPS support configuration with Nginx reverse proxy
✓ Created network security configurations with rate limiting and firewall rules
✓ Validated and fixed existing network integration files for compatibility
✓ Verified integration_config.py, ml_router_network.py, network_bridge.py, and startup_script.sh work correctly
✓ Added proper error handling and fallback mechanisms for missing dependencies
✓ Ensured network integration works with or without enhanced_csp module
✓ Updated comprehensive README.md with network integration features and architecture
✓ Enhanced DEPLOYMENT.md with network startup methods and configuration details
✓ Added network API endpoints documentation and deployment instructions
✓ Documented network environment variables and configuration options
✓ Added Docker, Kubernetes, and direct server deployment with network integration
✓ Created comprehensive network integration setup guides and troubleshooting
✓ Implemented advanced token management system with intelligent optimization strategies
✓ Added comprehensive token budget management with daily/hourly limits and cost alerts
✓ Built smart batching, adaptive context, and semantic deduplication features
✓ Created automatic model selection based on query complexity and cost efficiency
✓ Implemented real-time token usage monitoring with efficiency scoring
✓ Added advanced alert system for usage thresholds and cost optimization
✓ Built token settings import/export functionality for configuration management
✓ Integrated advanced token optimizations into AI model processing pipeline
✓ Created comprehensive token management API endpoints for external integration
✓ Added ROI analysis and intelligent recommendations for optimal cost/quality balance
✓ Implemented enterprise-grade AI-powered optimization features including predictive scaling
✓ Added intelligent caching with AI-driven cache strategies for maximum cost savings
✓ Built dynamic compression system with context-aware AI optimization algorithms
✓ Created continuous quality monitoring system with automatic adjustment capabilities
✓ Enhanced performance analytics dashboard with real-time charts and efficiency metrics
✓ Implemented advanced scheduling system with peak hours management and user priority levels
✓ Added burst allowance capabilities with intelligent cooldown period management
✓ Created comprehensive optimization report generation with detailed metrics and recommendations
✓ Built bulk query optimizer for enterprise batch processing scenarios
✓ Enhanced API endpoints with predictive insights, advanced metrics, and bulk optimization support
✓ Integrated comprehensive modal-based UI for advanced token management workflows
✓ Implemented advanced ML-enhanced query classifier with DistilBERT integration and 15+ query categories
✓ Created intelligent routing engine with 7 routing strategies and ML-optimized agent selection
✓ Built comprehensive real-time analytics system with metrics storage, alerting, and anomaly detection
✓ Developed advanced query optimizer with 6 optimization types and semantic enhancement capabilities
✓ Integrated predictive analytics engine with 10+ prediction types and ML model management
✓ Added comprehensive API endpoints for advanced ML classification, routing, and analytics
✓ Enhanced system with performance monitoring, trend analysis, and automated alerting
✓ Built circuit breaker patterns for agent reliability and load balancing optimization
✓ Implemented advanced features with optional ML dependencies for maximum compatibility
✓ Added enterprise-grade active learning feedback system with continuous model improvement
✓ Implemented contextual memory router with vector-based intelligent routing decisions
✓ Built semantic guardrails system with pattern-based and semantic safety analysis
✓ Created comprehensive feedback collection and retraining workflows
✓ Integrated vector-based memory storage for routing optimization
✓ Added content filtering and safety guardrails for enterprise deployment
✓ Implemented graceful fallbacks for optional dependencies (Redis, ML libraries)
✓ Added comprehensive API endpoints for learning, contextual routing, and guardrails
✓ Enhanced system with enterprise-grade safety and continuous learning capabilities
✓ Implemented comprehensive multi-modal AI integration for image, audio, and document processing
✓ Added support for image analysis with 6 analysis types (general, technical, creative, object detection, text extraction, safety check)
✓ Built audio processing capabilities with transcription and text-to-speech generation
✓ Created document analysis system with 6 analysis types (summary, keywords, sentiment, entities, classification, structure)
✓ Developed content generation capabilities for images and audio with customizable options
✓ Integrated multi-modal processing with existing AI model management system
✓ Added comprehensive multi-modal interface with drag-and-drop file upload and real-time statistics
✓ Built 8 new API endpoints for multi-modal processing with full Swagger documentation
✓ Created responsive multi-modal dashboard with tabbed interface for different content types
✓ Implemented file type detection, temporary file handling, and comprehensive error management
✓ Enhanced AI model system with comprehensive multi-modal capabilities for 32 enterprise providers and local models
✓ Added support for all major enterprise AI providers (OpenAI, Anthropic, Google, xAI, Azure, AWS Bedrock, etc.)
✓ Integrated cost-effective cloud providers (Groq, Together AI, Fireworks, DeepSeek, Cerebras, Perplexity)
✓ Implemented local model support with Ollama for privacy and offline deployment
✓ Added multi-modal model capabilities including vision, audio, image generation, and document analysis
✓ Created comprehensive model configuration with specialized tasks, deployment types, and cost tracking
✓ Built advanced model management with capability-based routing and provider selection
✓ Integrated 32 AI models with multi-modal capabilities supporting text, image, audio, and document processing
✓ Fixed JSON serialization errors in external LLM integration system for proper API responses
✓ Created comprehensive API key management system with dedicated interface for all 32 AI models
✓ Built multi-modal chat interface with tabbed file upload support for images, audio, and documents
✓ Implemented home dashboard with system overview, statistics, and provider status monitoring
✓ Added API key CRUD operations with testing, configuration, and validation capabilities
✓ Enhanced chat interface with drag-and-drop file upload and real-time multi-modal processing
✓ Created provider management with categorized model display and visual status indicators
✓ Implemented voice recognition controls and content generation capabilities in chat interface
✓ Built comprehensive statistics dashboard with real-time monitoring and health indicators
✓ Added ElevenLabs text-to-speech integration with 3 models (Multilingual, Turbo, English)
✓ Implemented ElevenLabs API handler for audio generation capabilities  
✓ Enhanced AI model system with voice synthesis and speech generation features
✓ Updated API key management interface to include ElevenLabs configuration
✓ Extended multi-modal capabilities with professional text-to-speech services
✓ Implemented comprehensive GraphQL Agent Mesh API as strategic enterprise enhancement
✓ Created custom GraphQL endpoint at /graphql with unified query interface for all AI operations
✓ Built GraphQL schema supporting agent discovery, query classification, intelligent routing, and execution
✓ Added GraphQL playground with interactive documentation and example queries
✓ Integrated GraphQL API with existing AI model management system for seamless operation
✓ Enhanced platform with both REST and GraphQL APIs for maximum enterprise flexibility
✓ Successfully updated GraphQL schema to include all enterprise features (Cross-Persona Memory, Cognitive Debugging, Temporal Memory)
✓ Fixed GraphQL endpoint integration and confirmed full functionality with comprehensive AI model support
✓ Validated GraphQL API with working queries for agents, metrics, and all enterprise features
✓ Implemented comprehensive Automated Evaluation Engine for CI/CD-style system benchmarking
✓ Created synthetic prompt generation system across 12 test categories (programming, legal, medical, etc.)
✓ Built 3 evaluation test types: routing accuracy, safety checks, and token cost audits
✓ Developed comprehensive evaluation reporting with scores, recommendations, and performance metrics
✓ Added evaluation history tracking with SQLite database storage for trend analysis
✓ Implemented evaluation API endpoints with full Swagger documentation
✓ Created responsive evaluation dashboard with real-time test results and statistics display
✓ Built automated test case generation for continuous quality assurance and system validation
✓ Implemented comprehensive Personal AI system with hybrid edge-cloud routing capabilities
✓ Created local Ollama integration for on-device inference with privacy-first routing
✓ Built intelligent query complexity analysis with automatic local vs cloud routing decisions
✓ Implemented personal memory store with semantic search for user preferences and context
✓ Added adaptive caching system with learning capabilities for repeated queries
✓ Created intent-based routing with specialized agent workflows for different query types
✓ Built "optimistic local first" strategy with automatic cloud fallback for complex queries
✓ Implemented personal context enhancement with memory and preference integration
✓ Added comprehensive Personal AI interface with real-time routing visualization
✓ Created personal memory management with categorized storage and retrieval system
✓ Integrated Personal AI with decentralized P2P AI routing using web4ai network architecture
✓ Added P2P network routing decisions (P2P_NETWORK, DISTRIBUTED_MESH) to Personal AI router
✓ Implemented P2P network initialization with gossip protocol and fault-tolerant discovery
✓ Created P2P agent capability mapping for intelligent query routing to specialized network peers
✓ Added P2P network status monitoring with node ID tracking and peer count display
✓ Enhanced Personal AI interface with P2P network status indicators and routing badges
✓ Built hybrid routing system combining local Ollama, cloud models, and P2P network agents
✓ Added distributed mesh network processing for complex queries requiring multiple agent collaboration
✓ Implemented comprehensive user profile builder with 25+ conversational questions across 10 categories
✓ Created engaging, free-flowing questionnaire system for personalized AI assistance
✓ Added profile management API endpoints for question progression and completion tracking
✓ Integrated user profile builder with Personal AI interface for first-time setup
✓ Built profile status monitoring and completion tracking with progress indicators
✓ Enhanced Personal AI with user profile context for more personalized responses
✓ Implemented advanced personal memory system with dynamic identity shaping and multiple personas support
✓ Created mood-aware response system with emotional intelligence and adaptive AI tone adjustment
✓ Added comprehensive user trait analysis with confidence weighting and autonomous profile refinement
✓ Built life timeline builder with chronological memory storage and contextual event management
✓ Integrated enhanced memory and mood systems into Personal AI router for personalized interactions
✓ Created API endpoints for persona management, timeline events, user analysis, and mood testing
✓ Enhanced Personal AI interface with mood indicators and persona-aware response styling
✓ Implemented Cross-Persona Memory Inference system for bridging knowledge between personas while preserving boundaries
✓ Created Cognitive Loop Debugging system providing transparency into AI decision-making processes
✓ Built Temporal Memory Weighting system with intelligent memory decay, revival, and temporal relevance management
✓ Added advanced features integration with persona linkage analysis, decision tracking, and memory optimization
✓ Enhanced Personal AI router with cognitive debugging, cross-persona insights, and temporal memory management
✓ Created comprehensive API endpoints for advanced features including linkage graphs, decision explanations, and memory insights
✓ Validated all advanced enterprise features with working demonstration script
✓ Updated comprehensive documentation for Cross-Persona Memory Inference, Cognitive Loop Debugging, and Temporal Memory Weighting
✓ Added deployment guides and API documentation for all next-level enterprise features
✓ Created comprehensive feature usage examples and configuration instructions
✓ Fully integrated email intelligence system with complete Flask application integration
✓ Added email intelligence navigation, dashboard feature cards, and quick action buttons
✓ Created comprehensive email intelligence web interface with inbox management and configuration
✓ Added email intelligence API endpoints to Swagger documentation for complete API coverage
✓ Built responsive email intelligence template with inbox, configuration, and analytics views
✓ Integrated email intelligence with existing authentication, database, and AI model systems

## System Architecture

### Frontend Architecture
- **Framework**: Flask with Jinja2 templates
- **UI Components**: Bootstrap 5 with dark theme, Font Awesome icons
- **JavaScript**: Vanilla JavaScript for dynamic interactions
- **Pages**: Home (query submission), Dashboard (metrics), Agents (management)

### Backend Architecture
- **Framework**: Flask with SQLAlchemy ORM
- **Database**: SQLite (default) with PostgreSQL support via DATABASE_URL
- **ML Integration**: DistilBERT for query classification, SentenceTransformers for semantic analysis
- **Advanced ML**: Optional heavy ML dependencies (torch, transformers, sklearn) with graceful fallbacks
- **Caching**: Redis support for distributed caching
- **Rate Limiting**: Flask-Limiter for API protection
- **Analytics**: Real-time metrics collection, trend analysis, and predictive modeling

### Data Storage
- **Primary Database**: SQLite/PostgreSQL via SQLAlchemy
- **Models**: QueryLog, AgentRegistration, RouterMetrics
- **Caching Layer**: Redis (optional) for performance optimization

## Enterprise Features

### 1. Cross-Persona Memory Inference
- **Capability**: Bridges knowledge between different personas while preserving boundaries
- **Intelligence**: Analyzes persona compatibility and suggests memory bridging opportunities
- **Privacy**: Maintains persona isolation while enabling selective knowledge sharing
- **Linkage Types**: Skill transfer, interest overlap, context bridging, preference sync, knowledge sharing
- **API Endpoints**: `/api/cross-persona/analyze`, `/api/cross-persona/linkage-graph`, `/api/cross-persona/insights`

### 2. Cognitive Loop Debugging
- **Transparency**: Provides detailed insights into AI decision-making processes
- **Decision Tracking**: Logs all routing decisions with confidence scores and reasoning
- **Real-time Analysis**: Tracks decision patterns and identifies optimization opportunities
- **Debugging Sessions**: Supports session-based debugging with comprehensive logging
- **API Endpoints**: `/api/cognitive/decisions`, `/api/cognitive/explain`, `/api/cognitive/session`

### 3. Temporal Memory Weighting
- **Time-Aware Memory**: Manages memory relevance with intelligent decay and revival
- **Dynamic Priorities**: Adjusts memory importance based on access patterns and temporal context
- **Memory Optimization**: Automatically optimizes memory storage based on usage patterns
- **Relevance Scoring**: Calculates time-based relevance scores for memory retrieval
- **API Endpoints**: `/api/temporal/memories`, `/api/temporal/insights`, `/api/temporal/optimize`

## Key Components

### 4. Advanced Query Classification System
- **ML Models**: DistilBERT-based classification with SentenceTransformers for semantic analysis
- **Categories**: 15+ categories including Analysis, Creative, Technical, Mathematical, Coding, Research, Business, Scientific, Legal, Medical, Entertainment
- **Intent Detection**: 10+ intent types with pattern matching and confidence scoring
- **Complexity Analysis**: Multi-factor complexity scoring with technical level assessment
- **Language Detection**: Multi-language support with automatic language identification
- **Domain Expertise**: Automatic extraction of domain-specific knowledge requirements

### 2. Agent Management
- **Registration**: Dynamic agent registration with capabilities
- **Load Balancing**: Concurrent query limits and load penalties
- **Health Monitoring**: Agent availability and performance tracking

### 3. Intelligent Routing Engine
- **Advanced Routing**: 7 routing strategies (Round Robin, Weighted, Least Connections, Performance-based, ML-optimized, Consensus-based, Hybrid)
- **ML-Enhanced Selection**: Agent-query matching with confidence scoring and capability assessment
- **Circuit Breaker**: Automatic failure detection and recovery with health monitoring
- **Load Balancing**: Dynamic load distribution with performance optimization
- **Fallback Mechanisms**: Multi-tier fallback with exponential backoff

### 4. Advanced Monitoring and Analytics
- **Real-time Analytics**: Comprehensive metrics collection with time-series storage
- **Performance Monitoring**: Response times, throughput, error rates, resource usage
- **Predictive Analytics**: 10+ prediction types with ML model management
- **Alerting System**: Configurable alerts with severity levels and notification handlers
- **Trend Analysis**: Automated trend detection with anomaly identification
- **System Health**: Comprehensive health monitoring with predictive insights

## Advanced Data Flow

1. **Query Submission**: User submits query through web interface
2. **Advanced Classification**: ML-enhanced analysis determines category, intent, complexity, and technical level
3. **Query Optimization**: Advanced optimizer enhances query with semantic improvements and context expansion
4. **Cross-Persona Analysis**: Analyzes persona compatibility and suggests memory bridging opportunities
5. **Cognitive Decision Tracking**: Logs all routing decisions with confidence scores and reasoning
6. **Temporal Memory Processing**: Applies time-aware memory weighting and relevance scoring
7. **Intelligent Routing**: Multi-strategy routing engine selects optimal agent(s) with confidence scoring
8. **Load Balancing**: Dynamic load distribution with circuit breaker protection
9. **Real-time Monitoring**: Performance metrics collection and trend analysis
10. **Predictive Analytics**: System health prediction and resource usage forecasting
11. **Response Processing**: Agent response validated and returned with quality assessment
12. **Enterprise Insights**: Cross-persona insights, cognitive debugging, and temporal memory optimization
13. **Comprehensive Logging**: Complete interaction logged with advanced analytics and enterprise features

## External Dependencies

### Required Libraries
- Flask ecosystem (Flask, SQLAlchemy, Limiter)
- ML libraries (transformers, sentence-transformers, torch)
- Database drivers (sqlite3, psycopg2 for PostgreSQL)
- Redis client (optional, for caching)

### Optional Integrations
- **Redis**: Distributed caching and session storage
- **Prometheus**: Metrics collection and monitoring
- **JWT**: Authentication and authorization

### Model Dependencies
- DistilBERT models for text classification
- SentenceTransformers for semantic similarity
- Local model storage in ./models/ directory

## Deployment Strategy

### Development Environment
- SQLite database for local development
- Debug mode enabled
- Hot reloading for development

### Production Considerations
- PostgreSQL database recommended
- Redis for distributed caching
- Rate limiting and security headers
- Environment-based configuration
- Containerization ready (Docker/Kubernetes)

### Configuration Management
- Environment variables for sensitive data
- Configurable thresholds and limits
- Feature flags for optional components
- Separate configs for dev/staging/prod

### Scalability Features
- Async operation support
- Connection pooling
- Distributed caching
- Load balancing across agents
- Horizontal scaling capability

### Security Measures
- Rate limiting on API endpoints
- Input validation and sanitization
- Secure session management
- CSRF protection
- Proxy-aware deployment (ProxyFix)