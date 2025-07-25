"""
OpenAPI/Swagger specification for ML Query Router API
"""

swagger_spec = {
    "openapi": "3.0.3",
    "info": {
        "title": "ML Query Router API",
        "description": "A Flask-based ML-Enhanced Query Router with intelligent agent routing, AI model management, RAG capabilities, and enterprise features including Cross-Persona Memory Inference, Cognitive Loop Debugging, and Temporal Memory Weighting",
        "version": "2.0.0",
        "contact": {
            "name": "API Support",
            "email": "support@example.com"
        }
    },
    "servers": [
        {
            "url": "http://localhost:5000",
            "description": "Development server"
        }
    ],
    "tags": [
        {
            "name": "Query Routing",
            "description": "Query submission and routing operations"
        },
        {
            "name": "Agent Management",
            "description": "Agent registration and management"
        },
        {
            "name": "AI Models",
            "description": "AI model management and configuration"
        },
        {
            "name": "Authentication",
            "description": "User authentication and authorization"
        },
        {
            "name": "Settings",
            "description": "Application configuration and settings"
        },
        {
            "name": "Configuration",
            "description": "Advanced system configuration"
        },
        {
            "name": "Cache Management",
            "description": "AI response caching operations"
        },
        {
            "name": "Chat Interface",
            "description": "Chat messaging and session management"
        },
        {
            "name": "RAG System",
            "description": "Document upload and retrieval-augmented generation"
        },
        {
            "name": "Evaluation Engine",
            "description": "Automated evaluation and testing system"
        },
        {
            "name": "Auto Chain Generator",
            "description": "Dynamic multi-step agent chain composition"
        },
        {
            "name": "Cross-Persona Memory",
            "description": "Cross-persona memory inference and linkage analysis"
        },
        {
            "name": "Cognitive Debugging",
            "description": "AI decision tracking and cognitive loop debugging"
        },
        {
            "name": "Email Intelligence",
            "description": "AI-powered email management with smart classification and replies"
        },
        {
            "name": "Temporal Memory",
            "description": "Time-aware memory weighting and relevance management"
        },
        {
            "name": "Enterprise Features",
            "description": "Advanced enterprise AI features and analytics"
        }
    ],
    "paths": {
        "/api/submit": {
            "post": {
                "tags": ["Query Routing"],
                "summary": "Submit a query for routing",
                "description": "Submit a user query to be classified and routed to appropriate AI agent",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The user query to be processed"
                                    },
                                    "user_id": {
                                        "type": "string",
                                        "description": "Optional user identifier"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Query processed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query_id": {"type": "string"},
                                        "category": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "agent_id": {"type": "string"},
                                        "agent_name": {"type": "string"},
                                        "response": {"type": "string"},
                                        "response_time": {"type": "number"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/agents": {
            "get": {
                "tags": ["Agent Management"],
                "summary": "Get list of available agents",
                "description": "Retrieve all registered agents with their capabilities",
                "responses": {
                    "200": {
                        "description": "List of agents",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Agent"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["Agent Management"],
                "summary": "Register a new agent",
                "description": "Register a new AI agent with the system",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AgentRegistration"}
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "Agent registered successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Agent"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid agent data",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/agents/{agent_id}": {
            "delete": {
                "tags": ["Agent Management"],
                "summary": "Unregister an agent",
                "description": "Remove an agent from the system",
                "parameters": [
                    {
                        "name": "agent_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The agent ID to unregister"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Agent unregistered successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Agent not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/stats": {
            "get": {
                "tags": ["Query Routing"],
                "summary": "Get routing statistics",
                "description": "Retrieve system performance and routing statistics",
                "responses": {
                    "200": {
                        "description": "System statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_queries": {"type": "integer"},
                                        "successful_routes": {"type": "integer"},
                                        "failed_routes": {"type": "integer"},
                                        "average_response_time": {"type": "number"},
                                        "active_agents": {"type": "integer"},
                                        "categories": {
                                            "type": "object",
                                            "additionalProperties": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models": {
            "get": {
                "tags": ["AI Models"],
                "summary": "Get all AI models",
                "description": "Retrieve all configured AI models",
                "responses": {
                    "200": {
                        "description": "List of AI models",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/AIModel"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["AI Models"],
                "summary": "Create a new AI model",
                "description": "Add a new AI model configuration",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AIModelCreate"}
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "AI model created successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AIModel"}
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models/{model_id}": {
            "get": {
                "tags": ["AI Models"],
                "summary": "Get specific AI model",
                "description": "Retrieve details of a specific AI model",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "AI model details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AIModel"}
                            }
                        }
                    },
                    "404": {
                        "description": "Model not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            },
            "put": {
                "tags": ["AI Models"],
                "summary": "Update AI model",
                "description": "Update an existing AI model configuration",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AIModelUpdate"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "AI model updated successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AIModel"}
                            }
                        }
                    }
                }
            },
            "delete": {
                "tags": ["AI Models"],
                "summary": "Delete AI model",
                "description": "Remove an AI model from the system",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "AI model deleted successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models/{model_id}/activate": {
            "post": {
                "tags": ["AI Models"],
                "summary": "Activate AI model",
                "description": "Set an AI model as active for use",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "AI model activated successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "model": {"$ref": "#/components/schemas/AIModel"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models/{model_id}/test": {
            "post": {
                "tags": ["AI Models"],
                "summary": "Test AI model",
                "description": "Test an AI model with a sample query",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "system_message": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Test completed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "response": {"type": "string"},
                                        "response_time": {"type": "number"},
                                        "status": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models/active": {
            "get": {
                "tags": ["AI Models"],
                "summary": "Get active AI model",
                "description": "Retrieve the currently active AI model",
                "responses": {
                    "200": {
                        "description": "Active AI model",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AIModel"}
                            }
                        }
                    },
                    "404": {
                        "description": "No active model found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/cache/stats": {
            "get": {
                "tags": ["Cache Management"],
                "summary": "Get cache statistics",
                "description": "Retrieve AI response cache statistics",
                "responses": {
                    "200": {
                        "description": "Cache statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_entries": {"type": "integer"},
                                        "hit_rate": {"type": "number"},
                                        "miss_rate": {"type": "number"},
                                        "cache_size_mb": {"type": "number"},
                                        "expired_entries": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cache/entries": {
            "get": {
                "tags": ["Cache Management"],
                "summary": "Get cache entries",
                "description": "Retrieve cached AI responses",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer", "default": 50},
                        "description": "Maximum number of entries to return"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Cache entries",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/CacheEntry"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cache/clear": {
            "post": {
                "tags": ["Cache Management"],
                "summary": "Clear cache",
                "description": "Clear all or specific cache entries",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "model_id": {"type": "string"},
                                    "expired_only": {"type": "boolean"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Cache cleared successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "cleared_entries": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/chat/message": {
            "post": {
                "tags": ["Chat Interface"],
                "summary": "Send chat message",
                "description": "Send a message to the AI chat interface",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "model": {"type": "string"},
                                    "system_message": {"type": "string"},
                                    "enable_rag": {"type": "boolean", "default": False}
                                },
                                "required": ["message"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Message processed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "response": {"type": "string"},
                                        "model_used": {"type": "string"},
                                        "cached": {"type": "boolean"},
                                        "rag_used": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/chat/stream": {
            "post": {
                "tags": ["Chat Interface"],
                "summary": "Stream chat response",
                "description": "Stream AI response using Server-Sent Events",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "model": {"type": "string"},
                                    "system_message": {"type": "string"},
                                    "enable_rag": {"type": "boolean", "default": False}
                                },
                                "required": ["message"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Streaming response",
                        "content": {
                            "text/event-stream": {
                                "schema": {
                                    "type": "string",
                                    "description": "Server-Sent Events stream"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/upload": {
            "post": {
                "tags": ["RAG System"],
                "summary": "Upload document for RAG",
                "description": "Upload a document for retrieval-augmented generation",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "format": "binary",
                                        "description": "Document file (PDF, DOCX, TXT, MD, HTML, JSON, CSV)"
                                    }
                                },
                                "required": ["file"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Document uploaded successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "document_id": {"type": "string"},
                                        "filename": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid file or upload error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/documents": {
            "get": {
                "tags": ["RAG System"],
                "summary": "List uploaded documents",
                "description": "Get list of all uploaded documents for RAG",
                "responses": {
                    "200": {
                        "description": "List of documents",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "documents": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/RAGDocument"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/documents/{doc_id}": {
            "delete": {
                "tags": ["RAG System"],
                "summary": "Delete RAG document",
                "description": "Delete a document from the RAG system",
                "parameters": [
                    {
                        "name": "doc_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The document ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Document deleted successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Document not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/search": {
            "post": {
                "tags": ["RAG System"],
                "summary": "Search documents",
                "description": "Search uploaded documents using vector similarity",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "max_results": {"type": "integer", "default": 3}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Search results",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "results": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/RAGSearchResult"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/stats": {
            "get": {
                "tags": ["RAG System"],
                "summary": "Get RAG statistics",
                "description": "Get statistics about the RAG system",
                "responses": {
                    "200": {
                        "description": "RAG statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_documents": {"type": "integer"},
                                        "total_chunks": {"type": "integer"},
                                        "file_types": {
                                            "type": "object",
                                            "additionalProperties": {"type": "integer"}
                                        },
                                        "collection_initialized": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate": {
            "post": {
                "tags": ["Collaborative AI"],
                "summary": "Submit collaborative query",
                "description": "Submit a query for collaborative AI processing with multiple specialized agents",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The query to process collaboratively",
                                        "example": "What are the pros and cons of artificial intelligence?"
                                    },
                                    "enable_rag": {
                                        "type": "boolean",
                                        "description": "Whether to enable RAG (Retrieval-Augmented Generation) for document context",
                                        "default": False
                                    },
                                    "max_agents": {
                                        "type": "integer",
                                        "description": "Maximum number of agents to use (only applies to automatic selection)",
                                        "default": 3,
                                        "minimum": 1,
                                        "maximum": 5
                                    },
                                    "timeout": {
                                        "type": "integer",
                                        "description": "Collaboration timeout in seconds",
                                        "default": 300,
                                        "minimum": 60,
                                        "maximum": 600
                                    },
                                    "selected_agents": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                        "description": "Specific agents to use for collaboration (overrides automatic selection)",
                                        "example": ["collab_analyst", "collab_technical"],
                                        "enum": ["collab_analyst", "collab_creative", "collab_technical", "collab_researcher", "collab_synthesizer"]
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Collaborative processing completed",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CollaborativeResult"}
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate/sessions": {
            "get": {
                "tags": ["Collaborative AI"],
                "summary": "Get active collaboration sessions",
                "description": "Get information about active collaboration sessions",
                "responses": {
                    "200": {
                        "description": "Active sessions",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "sessions": {
                                            "type": "object",
                                            "additionalProperties": {"$ref": "#/components/schemas/CollaborativeSession"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate/sessions/{session_id}": {
            "get": {
                "tags": ["Collaborative AI"],
                "summary": "Get session details",
                "description": "Get detailed information about a specific collaboration session",
                "parameters": [
                    {
                        "name": "session_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Session ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Session details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CollaborativeSession"}
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate/agents": {
            "get": {
                "tags": ["Collaborative AI"],
                "summary": "Get collaborative agent configurations",
                "description": "Get current configurations for all collaborative agents",
                "responses": {
                    "200": {
                        "description": "Agent configurations",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CollaborativeAgentConfig"}
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate/agents/{agent_id}/model": {
            "put": {
                "tags": ["Collaborative AI"],
                "summary": "Update agent model",
                "description": "Update the AI model for a specific collaborative agent",
                "parameters": [
                    {
                        "name": "agent_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Agent ID"
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "model_id": {
                                        "type": "string",
                                        "description": "ID of the AI model to use"
                                    }
                                },
                                "required": ["model_id"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Agent model updated successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/shared-memory/stats": {
            "get": {
                "tags": ["Shared Memory"],
                "summary": "Get shared memory statistics",
                "description": "Get statistics about the shared memory system",
                "responses": {
                    "200": {
                        "description": "Shared memory statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_messages": {"type": "integer"},
                                        "active_sessions": {"type": "integer"},
                                        "agent_contexts": {"type": "integer"},
                                        "message_index_size": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/shared-memory/sessions/{session_id}/messages": {
            "get": {
                "tags": ["Shared Memory"],
                "summary": "Get session messages",
                "description": "Get messages from a specific collaboration session",
                "parameters": [
                    {
                        "name": "session_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Session ID"
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "integer", "default": 50},
                        "description": "Maximum number of messages to return"
                    },
                    {
                        "name": "types",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "description": "Message types to filter by"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Session messages",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "session_id": {"type": "string"},
                                        "messages": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/SharedMemoryMessage"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/shared-memory/sessions/{session_id}/context": {
            "get": {
                "tags": ["Shared Memory"],
                "summary": "Get session context",
                "description": "Get shared context for a collaboration session",
                "parameters": [
                    {
                        "name": "session_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Session ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Session context",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "additionalProperties": True
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/external-llm/analyze": {
            "post": {
                "tags": ["External LLM"],
                "summary": "Analyze query complexity",
                "description": "Analyze a query to determine its complexity and routing requirements",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "The query to analyze"}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Query complexity analysis completed",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/QueryComplexityAnalysis"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/external-llm/process": {
            "post": {
                "tags": ["External LLM"],
                "summary": "Process query with external LLM",
                "description": "Process a complex query using external LLM providers",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "The query to process"},
                                    "context": {"type": "string", "description": "Optional context for the query"},
                                    "preferred_provider": {"type": "string", "description": "Preferred external LLM provider"}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Query processed successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ExternalLLMResponse"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/external-llm/providers": {
            "get": {
                "tags": ["External LLM"],
                "summary": "Get external LLM providers",
                "description": "Get list of available external LLM providers and their configurations",
                "responses": {
                    "200": {
                        "description": "Providers retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "providers": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/ExternalProvider"}
                                        },
                                        "total_providers": {"type": "integer"},
                                        "available_providers": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    },
                    "500": {
                        "description": "Error retrieving providers",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/external-llm/metrics": {
            "get": {
                "tags": ["External LLM"],
                "summary": "Get external LLM metrics",
                "description": "Get performance metrics for external LLM providers",
                "responses": {
                    "200": {
                        "description": "Metrics retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ExternalLLMMetrics"}
                            }
                        }
                    },
                    "500": {
                        "description": "Error retrieving metrics",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/health": {
            "get": {
                "tags": ["System"],
                "summary": "Health check",
                "description": "Check system health and status",
                "responses": {
                    "200": {
                        "description": "System is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "timestamp": {"type": "string"},
                                        "version": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                    "code": {"type": "integer"}
                }
            },
            "Agent": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "endpoint": {"type": "string"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "capabilities": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "is_active": {"type": "boolean"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "last_seen": {"type": "string", "format": "date-time"}
                }
            },
            "AgentRegistration": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "endpoint": {"type": "string"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "capabilities": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["name", "endpoint", "categories"]
            },
            "AIModel": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "provider": {"type": "string"},
                    "model_id": {"type": "string"},
                    "description": {"type": "string"},
                    "config": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "is_active": {"type": "boolean"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "updated_at": {"type": "string", "format": "date-time"}
                }
            },
            "AIModelCreate": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "provider": {"type": "string"},
                    "model_id": {"type": "string"},
                    "description": {"type": "string"},
                    "config": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["name", "provider", "model_id"]
            },
            "AIModelUpdate": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "config": {
                        "type": "object",
                        "additionalProperties": True
                    }
                }
            },
            "CacheEntry": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "cache_key": {"type": "string"},
                    "query": {"type": "string"},
                    "response": {"type": "string"},
                    "model_id": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "expires_at": {"type": "string", "format": "date-time"},
                    "hit_count": {"type": "integer"},
                    "last_accessed": {"type": "string", "format": "date-time"}
                }
            },
            "RAGDocument": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "filename": {"type": "string"},
                    "file_type": {"type": "string"},
                    "chunk_count": {"type": "integer"},
                    "added_at": {"type": "string", "format": "date-time"}
                }
            },
            "RAGSearchResult": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                            "document_name": {"type": "string"},
                            "file_type": {"type": "string"},
                            "chunk_index": {"type": "integer"}
                        }
                    },
                    "similarity_score": {"type": "number"},
                    "rank": {"type": "integer"}
                }
            },
            "CollaborativeResult": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "query": {"type": "string"},
                    "enhanced_query": {"type": "string"},
                    "final_response": {"type": "string"},
                    "agents_used": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "agent_responses": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "agent_id": {"type": "string"},
                                "agent_name": {"type": "string"},
                                "specialization": {"type": "string"},
                                "response": {"type": "string"},
                                "model_used": {"type": "string"},
                                "response_time": {"type": "number"},
                                "cached": {"type": "boolean"}
                            }
                        }
                    },
                    "confidence_score": {"type": "number"},
                    "rag_used": {"type": "boolean"},
                    "rag_context": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"}
                }
            },
            "CollaborativeSession": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "query": {"type": "string"},
                    "agents": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "status": {"type": "string"},
                    "duration_minutes": {"type": "number"},
                    "created_at": {"type": "string", "format": "date-time"}
                }
            },
            "CollaborativeAgentConfig": {
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "specialization": {"type": "string"},
                                "current_model": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                        "provider": {"type": "string"}
                                    }
                                },
                                "confidence_threshold": {"type": "number"},
                                "is_active": {"type": "boolean"},
                                "current_sessions": {"type": "integer"},
                                "max_concurrent_sessions": {"type": "integer"}
                            }
                        }
                    },
                    "available_models": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "provider": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "SharedMemoryMessage": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "message_type": {"type": "string"},
                    "content": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True
                    }
                }
            },
            "QueryComplexityAnalysis": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "complexity": {"type": "string", "enum": ["simple", "moderate", "complex", "very_complex"]},
                    "domain": {"type": "string"},
                    "requires_reasoning": {"type": "boolean"},
                    "requires_creativity": {"type": "boolean"},
                    "requires_analysis": {"type": "boolean"},
                    "requires_multi_step": {"type": "boolean"},
                    "context_length": {"type": "integer"},
                    "specialized_knowledge": {"type": "array", "items": {"type": "string"}},
                    "is_complex": {"type": "boolean"},
                    "recommended_provider": {"type": "string"}
                }
            },
            "ExternalLLMResponse": {
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "provider": {"type": "string"},
                    "complexity": {"type": "string"},
                    "processing_time": {"type": "number"},
                    "success": {"type": "boolean"},
                    "error": {"type": "string"}
                }
            },
            "ExternalProvider": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "endpoint": {"type": "string"},
                    "max_tokens": {"type": "integer"},
                    "cost_per_1k_tokens": {"type": "number"},
                    "rate_limit_rpm": {"type": "integer"},
                    "specializations": {"type": "array", "items": {"type": "string"}},
                    "api_key_available": {"type": "boolean"}
                }
            },
            "ExternalLLMMetrics": {
                "type": "object",
                "properties": {
                    "total_queries": {"type": "integer"},
                    "successful_queries": {"type": "integer"},
                    "failed_queries": {"type": "integer"},
                    "average_response_time": {"type": "number"},
                    "provider_breakdown": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "total_requests": {"type": "integer"},
                                "successful_requests": {"type": "integer"},
                                "failed_requests": {"type": "integer"},
                                "avg_response_time": {"type": "number"},
                                "rate_limit_hits": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        },
        '/api/multimodal/process': {
            'post': {
                'tags': ['Multi-Modal AI'],
                'summary': 'Process multi-modal content',
                'description': 'Process image, audio, or document files with AI',
                'requestBody': {
                    'required': True,
                    'content': {
                        'multipart/form-data': {
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'file': {
                                        'type': 'string',
                                        'format': 'binary',
                                        'description': 'File to process'
                                    },
                                    'processing_type': {
                                        'type': 'string',
                                        'default': 'auto',
                                        'description': 'Type of processing to perform'
                                    },
                                    'analysis_type': {
                                        'type': 'string',
                                        'default': 'general',
                                        'description': 'Analysis type for the content'
                                    }
                                },
                                'required': ['file']
                            }
                        }
                    }
                },
                'responses': {
                    '200': {
                        'description': 'Content processed successfully',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'object',
                                    'properties': {
                                        'processing_result': {
                                            '$ref': '#/components/schemas/ProcessingResult'
                                        }
                                    }
                                }
                            }
                        }
                    },
                    '400': {'description': 'Invalid request'},
                    '500': {'description': 'Internal server error'}
                }
            }
        },
        '/api/multimodal/generate': {
            'post': {
                'tags': ['Multi-Modal AI'],
                'summary': 'Generate multi-modal content',
                'description': 'Generate image or audio content using AI',
                'requestBody': {
                    'required': True,
                    'content': {
                        'application/json': {
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'content_type': {
                                        'type': 'string',
                                        'enum': ['image', 'audio'],
                                        'description': 'Type of content to generate'
                                    },
                                    'prompt': {
                                        'type': 'string',
                                        'description': 'Prompt for content generation'
                                    },
                                    'options': {
                                        'type': 'object',
                                        'description': 'Generation options'
                                    }
                                },
                                'required': ['content_type', 'prompt']
                            }
                        }
                    }
                },
                'responses': {
                    '200': {
                        'description': 'Content generated successfully',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'object',
                                    'properties': {
                                        'generation_result': {
                                            '$ref': '#/components/schemas/ProcessingResult'
                                        }
                                    }
                                }
                            }
                        }
                    },
                    '400': {'description': 'Invalid request'},
                    '500': {'description': 'Internal server error'}
                }
            }
        },
        '/api/multimodal/analyze-image': {
            'post': {
                'tags': ['Multi-Modal AI'],
                'summary': 'Analyze image content',
                'description': 'Analyze image content with AI',
                'requestBody': {
                    'required': True,
                    'content': {
                        'multipart/form-data': {
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'file': {
                                        'type': 'string',
                                        'format': 'binary',
                                        'description': 'Image file to analyze'
                                    },
                                    'analysis_type': {
                                        'type': 'string',
                                        'enum': ['general', 'technical', 'creative', 'object_detection', 'text_extraction', 'safety_check'],
                                        'default': 'general',
                                        'description': 'Type of image analysis'
                                    }
                                },
                                'required': ['file']
                            }
                        }
                    }
                },
                'responses': {
                    '200': {
                        'description': 'Image analyzed successfully',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'object',
                                    'properties': {
                                        'image_analysis': {
                                            '$ref': '#/components/schemas/ProcessingResult'
                                        }
                                    }
                                }
                            }
                        }
                    },
                    '400': {'description': 'Invalid request'},
                    '500': {'description': 'Internal server error'}
                }
            }
        },
        '/api/multimodal/stats': {
            'get': {
                'tags': ['Multi-Modal AI'],
                'summary': 'Get multi-modal processing statistics',
                'description': 'Get comprehensive statistics for multi-modal AI processing',
                'responses': {
                    '200': {
                        'description': 'Statistics retrieved successfully',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'object',
                                    'properties': {
                                        'multimodal_stats': {
                                            'type': 'object',
                                            'properties': {
                                                'total_processed': {'type': 'integer'},
                                                'successful_processed': {'type': 'integer'},
                                                'success_rate': {'type': 'number'},
                                                'average_processing_time': {'type': 'number'},
                                                'by_type': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'image': {'type': 'integer'},
                                                        'audio': {'type': 'integer'},
                                                        'document': {'type': 'integer'}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    '500': {'description': 'Internal server error'}
                }
            }
        },
        "/api/evaluation/run": {
            "post": {
                "tags": ["Evaluation Engine"],
                "summary": "Run comprehensive evaluation",
                "description": "Execute automated evaluation tests across all test types",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "prompts_per_category": {
                                        "type": "integer",
                                        "description": "Number of prompts per category",
                                        "default": 5
                                    },
                                    "include_real_prompts": {
                                        "type": "boolean",
                                        "description": "Include real user prompts",
                                        "default": True
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Evaluation completed successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/EvaluationReport"}
                            }
                        }
                    },
                    "500": {
                        "description": "Evaluation failed",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/evaluation/history": {
            "get": {
                "tags": ["Evaluation Engine"],
                "summary": "Get evaluation history",
                "description": "Retrieve past evaluation results",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Number of results to return",
                        "schema": {"type": "integer", "default": 10}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Evaluation history retrieved",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "evaluation_history": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/EvaluationReport"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/evaluation/stats": {
            "get": {
                "tags": ["Evaluation Engine"],
                "summary": "Get evaluation statistics",
                "description": "Retrieve evaluation system statistics",
                "responses": {
                    "200": {
                        "description": "Statistics retrieved",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "evaluation_stats": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/peer-teaching/agents/register": {
            "post": {
                "tags": ["Peer Teaching"],
                "summary": "Register a new agent in the peer teaching system",
                "description": "Register a new agent with specialization and capabilities",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "agent_id": {"type": "string"},
                                    "agent_name": {"type": "string"},
                                    "specialization": {"type": "string"},
                                    "capabilities": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["agent_name", "specialization"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Agent registered successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "agent_id": {"type": "string"},
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/peer-teaching/lessons/contribute": {
            "post": {
                "tags": ["Peer Teaching"],
                "summary": "Contribute a lesson to the peer teaching system",
                "description": "Submit a lesson for other agents to learn from",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "agent_id": {"type": "string"},
                                    "lesson_type": {"type": "string"},
                                    "domain": {"type": "string"},
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "strategy_steps": {"type": "array", "items": {"type": "string"}},
                                    "effectiveness_score": {"type": "number"},
                                    "usage_context": {"type": "string"},
                                    "success_metrics": {"type": "object"}
                                },
                                "required": ["agent_id", "lesson_type", "domain", "title", "content", "strategy_steps"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Lesson contributed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "lesson_id": {"type": "string"},
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/peer-teaching/lessons/find": {
            "post": {
                "tags": ["Peer Teaching"],
                "summary": "Find relevant lessons for an agent",
                "description": "Search for lessons based on domain and type",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "agent_id": {"type": "string"},
                                    "domain": {"type": "string"},
                                    "lesson_type": {"type": "string"}
                                },
                                "required": ["agent_id", "domain"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Lessons found",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "lessons": {"type": "array", "items": {"$ref": "#/components/schemas/AgentLesson"}},
                                        "total_found": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/peer-teaching/collaborate/start": {
            "post": {
                "tags": ["Peer Teaching"],
                "summary": "Start a collaborative session",
                "description": "Initialize a multi-agent collaborative session",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "initiator_agent": {"type": "string"},
                                    "task_description": {"type": "string"},
                                    "session_type": {"type": "string"},
                                    "required_specializations": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["initiator_agent", "task_description"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Collaborative session started",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "session_id": {"type": "string"},
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/peer-teaching/debate": {
            "post": {
                "tags": ["Peer Teaching"],
                "summary": "Conduct multi-agent debate",
                "description": "Start a debate session with multiple agents",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "session_id": {"type": "string"},
                                    "question": {"type": "string"},
                                    "consensus_method": {"type": "string"}
                                },
                                "required": ["session_id", "question"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Debate completed",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "debate_result": {"$ref": "#/components/schemas/DebateResult"},
                                        "timestamp": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/peer-teaching/stats": {
            "get": {
                "tags": ["Peer Teaching"],
                "summary": "Get peer teaching system statistics",
                "description": "Retrieve comprehensive statistics about the peer teaching system",
                "responses": {
                    "200": {
                        "description": "Peer teaching statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "peer_teaching_stats": {"$ref": "#/components/schemas/PeerTeachingStats"},
                                        "timestamp": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cross-persona/analyze": {
            "post": {
                "tags": ["Cross-Persona Memory"],
                "summary": "Analyze persona compatibility",
                "description": "Analyze compatibility between two personas and generate linkage suggestions",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "persona_1": {"type": "string"},
                                    "persona_2": {"type": "string"},
                                    "persona_1_data": {"type": "object"},
                                    "persona_2_data": {"type": "object"}
                                },
                                "required": ["persona_1", "persona_2"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Persona compatibility analysis completed",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "linkages": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/CrossPersonaLinkage"}
                                        },
                                        "compatibility_score": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cross-persona/linkage-graph": {
            "get": {
                "tags": ["Cross-Persona Memory"],
                "summary": "Get persona linkage graph",
                "description": "Generate a graph representation of persona linkages",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID to get linkages for",
                        "schema": {"type": "string"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Persona linkage graph generated",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "nodes": {"type": "array", "items": {"type": "object"}},
                                        "edges": {"type": "array", "items": {"type": "object"}},
                                        "metadata": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cross-persona/insights": {
            "get": {
                "tags": ["Cross-Persona Memory"],
                "summary": "Get cross-persona insights",
                "description": "Generate insights from cross-persona analysis",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID to get insights for",
                        "schema": {"type": "string"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Cross-persona insights generated",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "insights": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/EnterpriseInsight"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cognitive/decisions": {
            "get": {
                "tags": ["Cognitive Debugging"],
                "summary": "Get cognitive decisions",
                "description": "Retrieve logged AI decisions for a user",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID to get decisions for",
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Number of decisions to return",
                        "schema": {"type": "integer", "default": 10}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Cognitive decisions retrieved",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "decisions": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/CognitiveDecision"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cognitive/explain": {
            "post": {
                "tags": ["Cognitive Debugging"],
                "summary": "Explain cognitive decision",
                "description": "Get detailed explanation of a specific decision",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "decision_id": {"type": "string"}
                                },
                                "required": ["decision_id"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Decision explanation generated",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "explanation": {"type": "object"},
                                        "decision_path": {"type": "array", "items": {"type": "string"}},
                                        "alternative_paths": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cognitive/session": {
            "post": {
                "tags": ["Cognitive Debugging"],
                "summary": "Start cognitive debugging session",
                "description": "Start a new cognitive debugging session",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string"},
                                    "session_metadata": {"type": "object"}
                                },
                                "required": ["user_id"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Debugging session started",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "session_id": {"type": "string"},
                                        "started_at": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/temporal/memories": {
            "get": {
                "tags": ["Temporal Memory"],
                "summary": "Get temporal memories",
                "description": "Retrieve time-weighted memories for a user",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID to get memories for",
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Number of memories to return",
                        "schema": {"type": "integer", "default": 10}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Temporal memories retrieved",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "memories": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/TemporalMemory"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/temporal/insights": {
            "get": {
                "tags": ["Temporal Memory"],
                "summary": "Get temporal memory insights",
                "description": "Generate insights from temporal memory analysis",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID to get insights for",
                        "schema": {"type": "string"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Temporal memory insights generated",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "insights": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/EnterpriseInsight"}
                                        },
                                        "memory_statistics": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/temporal/optimize": {
            "post": {
                "tags": ["Temporal Memory"],
                "summary": "Optimize temporal memory",
                "description": "Optimize memory storage based on temporal patterns",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string"},
                                    "optimization_type": {"type": "string"}
                                },
                                "required": ["user_id"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Memory optimization completed",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "optimization_results": {"type": "object"},
                                        "memories_optimized": {"type": "integer"},
                                        "performance_improvement": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/configure": {
            "post": {
                "tags": ["Email Intelligence"],
                "summary": "Configure email provider",
                "description": "Configure email provider settings (IMAP/SMTP, Gmail API, etc.)",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "provider": {"type": "string", "enum": ["imap_smtp", "gmail_api", "outlook_graph"]},
                                    "settings": {"$ref": "#/components/schemas/EmailConfiguration"}
                                },
                                "required": ["provider", "settings"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Email provider configured successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "message": {"type": "string"},
                                        "provider": {"type": "string"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/fetch": {
            "post": {
                "tags": ["Email Intelligence"],
                "summary": "Fetch emails from provider",
                "description": "Fetch and classify emails from configured email provider",
                "requestBody": {
                    "required": False,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "provider": {"type": "string", "enum": ["imap_smtp", "gmail_api", "outlook_graph"]}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Emails fetched and classified successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "messages": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/EmailMessage"}
                                        },
                                        "count": {"type": "integer"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/messages": {
            "get": {
                "tags": ["Email Intelligence"],
                "summary": "Get stored email messages",
                "description": "Retrieve stored email messages from database",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Number of messages to return",
                        "schema": {"type": "integer", "default": 50}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Email messages retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "messages": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/EmailMessage"}
                                        },
                                        "count": {"type": "integer"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/summary": {
            "get": {
                "tags": ["Email Intelligence"],
                "summary": "Get email summary",
                "description": "Get email summary and statistics",
                "parameters": [
                    {
                        "name": "days_back",
                        "in": "query",
                        "description": "Number of days back to analyze",
                        "schema": {"type": "integer", "default": 7}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Email summary retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "summary": {"$ref": "#/components/schemas/EmailSummary"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/classifications": {
            "get": {
                "tags": ["Email Intelligence"],
                "summary": "Get email classifications",
                "description": "Get available email classifications, intents, tones, and providers",
                "responses": {
                    "200": {
                        "description": "Email classifications retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "classifications": {
                                            "type": "object",
                                            "properties": {
                                                "classifications": {"type": "array", "items": {"type": "string"}},
                                                "intents": {"type": "array", "items": {"type": "string"}},
                                                "tones": {"type": "array", "items": {"type": "string"}},
                                                "providers": {"type": "array", "items": {"type": "string"}}
                                            }
                                        },
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/writing-style/settings": {
            "get": {
                "tags": ["Email Intelligence"],
                "summary": "Get writing style settings",
                "description": "Get writing style training settings for a user",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID",
                        "schema": {"type": "string", "default": "default"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Writing style settings retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "settings": {"$ref": "#/components/schemas/WritingStyleSettings"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["Email Intelligence"],
                "summary": "Update writing style settings",
                "description": "Update writing style training settings for a user",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID",
                        "schema": {"type": "string", "default": "default"}
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/WritingStyleSettings"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Writing style settings updated successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "message": {"type": "string"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/writing-style/training-data": {
            "get": {
                "tags": ["Email Intelligence"],
                "summary": "Get training data",
                "description": "Get writing style training data for a user",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID",
                        "schema": {"type": "string", "default": "default"}
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Number of training entries to return",
                        "schema": {"type": "integer", "default": 100}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Training data retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "training_data": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/TrainingDataEntry"}
                                        },
                                        "count": {"type": "integer"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["Email Intelligence"],
                "summary": "Add training data",
                "description": "Add writing style training data for a user",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID",
                        "schema": {"type": "string", "default": "default"}
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "email_id": {"type": "string"},
                                    "training_text": {"type": "string"},
                                    "tone": {"type": "string", "enum": ["professional", "friendly", "formal", "casual", "supportive", "assertive", "empathetic"]},
                                    "context": {"type": "string"},
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                },
                                "required": ["email_id", "training_text"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Training data added successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "message": {"type": "string"},
                                        "training_id": {"type": "integer"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/writing-style/approve/{training_id}": {
            "post": {
                "tags": ["Email Intelligence"],
                "summary": "Approve training data",
                "description": "Approve training data for use in writing style learning",
                "parameters": [
                    {
                        "name": "training_id",
                        "in": "path",
                        "required": True,
                        "description": "Training data ID",
                        "schema": {"type": "integer"}
                    },
                    {
                        "name": "user_id",
                        "in": "query",
                        "description": "User ID",
                        "schema": {"type": "string", "default": "default"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Training data approved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "message": {"type": "string"},
                                        "training_id": {"type": "integer"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/writing-style/consent": {
            "get": {
                "tags": ["Email Intelligence"],
                "summary": "Get consent options",
                "description": "Get available training consent options",
                "responses": {
                    "200": {
                        "description": "Consent options retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "consent_options": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/ConsentOption"}
                                        },
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/email/writing-style/cleanup": {
            "post": {
                "tags": ["Email Intelligence"],
                "summary": "Clean up expired training data",
                "description": "Remove expired training data from the system",
                "responses": {
                    "200": {
                        "description": "Training data cleanup completed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "message": {"type": "string"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    'components': {
        'schemas': {
            'ProcessingResult': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'processing_type': {'type': 'string'},
                    'result': {'type': 'object'},
                    'confidence': {'type': 'number'},
                    'processing_time': {'type': 'number'},
                    'error_message': {'type': 'string'},
                    'timestamp': {'type': 'string', 'format': 'date-time'}
                }
            },
            'MediaFile': {
                'type': 'object',
                'properties': {
                    'file_path': {'type': 'string'},
                    'file_type': {'type': 'string'},
                    'mime_type': {'type': 'string'},
                    'size_bytes': {'type': 'integer'},
                    'created_at': {'type': 'string', 'format': 'date-time'}
                }
            },
            'ExternalLLMResponse': {
                'type': 'object',
                'properties': {
                    'response': {'type': 'string'},
                    'provider': {'type': 'string'},
                    'complexity': {'type': 'string'},
                    'processing_time': {'type': 'number'},
                    'success': {'type': 'boolean'},
                    'error': {'type': 'string'}
                }
            },
            'ExternalProvider': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'name': {'type': 'string'},
                    'endpoint': {'type': 'string'},
                    'max_tokens': {'type': 'integer'},
                    'cost_per_1k_tokens': {'type': 'number'},
                    'rate_limit_rpm': {'type': 'integer'},
                    'specializations': {'type': 'array', 'items': {'type': 'string'}},
                    'api_key_available': {'type': 'boolean'}
                }
            },
            'ExternalLLMMetrics': {
                'type': 'object',
                'properties': {
                    'total_queries': {'type': 'integer'},
                    'successful_queries': {'type': 'integer'},
                    'failed_queries': {'type': 'integer'},
                    'average_response_time': {'type': 'number'},
                    'provider_breakdown': {
                        'type': 'object',
                        'additionalProperties': {
                            'type': 'object',
                            'properties': {
                                'total_requests': {'type': 'integer'},
                                'successful_requests': {'type': 'integer'},
                                'failed_requests': {'type': 'integer'},
                                'avg_response_time': {'type': 'number'},
                                'rate_limit_hits': {'type': 'integer'}
                            }
                        }
                    }
                }
            },
            'EvaluationReport': {
                'type': 'object',
                'properties': {
                    'test_session_id': {'type': 'string'},
                    'total_tests': {'type': 'integer'},
                    'passed_tests': {'type': 'integer'},
                    'failed_tests': {'type': 'integer'},
                    'overall_score': {'type': 'number'},
                    'routing_accuracy': {'type': 'number'},
                    'safety_score': {'type': 'number'},
                    'cost_efficiency': {'type': 'number'},
                    'average_latency': {'type': 'number'},
                    'test_results': {
                        'type': 'array',
                        'items': {'$ref': '#/components/schemas/TestResult'}
                    },
                    'recommendations': {
                        'type': 'array',
                        'items': {'type': 'string'}
                    },
                    'timestamp': {'type': 'string', 'format': 'date-time'}
                }
            },
            'TestResult': {
                'type': 'object',
                'properties': {
                    'test_id': {'type': 'string'},
                    'test_type': {'type': 'string'},
                    'prompt': {'type': 'string'},
                    'category': {'type': 'string'},
                    'success': {'type': 'boolean'},
                    'score': {'type': 'number'},
                    'execution_time': {'type': 'number'},
                    'cost': {'type': 'number'},
                    'error_message': {'type': 'string'},
                    'timestamp': {'type': 'string', 'format': 'date-time'}
                }
            },
            'AgentLesson': {
                'type': 'object',
                'properties': {
                    'lesson_id': {'type': 'string'},
                    'title': {'type': 'string'},
                    'content': {'type': 'string'},
                    'lesson_type': {'type': 'string'},
                    'domain': {'type': 'string'},
                    'effectiveness_score': {'type': 'number'},
                    'adoption_count': {'type': 'integer'},
                    'strategy_steps': {'type': 'array', 'items': {'type': 'string'}},
                    'created_at': {'type': 'string', 'format': 'date-time'}
                }
            },
            'DebateResult': {
                'type': 'object',
                'properties': {
                    'session_id': {'type': 'string'},
                    'consensus_result': {'type': 'object'},
                    'agent_positions': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'agent_id': {'type': 'string'},
                                'specialization': {'type': 'string'},
                                'position': {'type': 'string'},
                                'confidence': {'type': 'number'},
                                'reasoning': {'type': 'string'}
                            }
                        }
                    },
                    'lessons_generated': {'type': 'array', 'items': {'type': 'string'}}
                }
            },
            'PeerTeachingStats': {
                'type': 'object',
                'properties': {
                    'total_agents': {'type': 'integer'},
                    'total_lessons': {'type': 'integer'},
                    'active_sessions': {'type': 'integer'},
                    'knowledge_contributions': {'type': 'integer'},
                    'lesson_adoption_rate': {'type': 'number'},
                    'collaboration_success_rate': {'type': 'number'},
                    'specialization_distribution': {'type': 'object'},
                    'top_lesson_contributors': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'agent_id': {'type': 'string'},
                                'agent_name': {'type': 'string'},
                                'lessons_contributed': {'type': 'integer'}
                            }
                        }
                    }
                }
            },
            'CrossPersonaLinkage': {
                'type': 'object',
                'properties': {
                    'persona_1': {'type': 'string'},
                    'persona_2': {'type': 'string'},
                    'linkage_type': {'type': 'string'},
                    'confidence': {'type': 'number'},
                    'description': {'type': 'string'},
                    'created_at': {'type': 'string', 'format': 'date-time'},
                    'usage_count': {'type': 'integer'},
                    'user_approved': {'type': 'boolean'}
                }
            },
            'CognitiveDecision': {
                'type': 'object',
                'properties': {
                    'decision_id': {'type': 'string'},
                    'decision_type': {'type': 'string'},
                    'decision_made': {'type': 'string'},
                    'reasoning': {'type': 'string'},
                    'confidence': {'type': 'number'},
                    'context': {'type': 'object'},
                    'alternatives': {'type': 'array', 'items': {'type': 'string'}},
                    'timestamp': {'type': 'string', 'format': 'date-time'}
                }
            },
            'TemporalMemory': {
                'type': 'object',
                'properties': {
                    'memory_id': {'type': 'string'},
                    'content': {'type': 'string'},
                    'confidence': {'type': 'number'},
                    'priority': {'type': 'string'},
                    'access_count': {'type': 'integer'},
                    'last_accessed': {'type': 'string', 'format': 'date-time'},
                    'created_at': {'type': 'string', 'format': 'date-time'},
                    'temporal_weight': {'type': 'number'},
                    'relevance_score': {'type': 'number'}
                }
            },
            'EnterpriseInsight': {
                'type': 'object',
                'properties': {
                    'insight_type': {'type': 'string'},
                    'description': {'type': 'string'},
                    'confidence': {'type': 'number'},
                    'involved_entities': {'type': 'array', 'items': {'type': 'string'}},
                    'evidence': {'type': 'array', 'items': {'type': 'string'}},
                    'recommendation': {'type': 'string'},
                    'timestamp': {'type': 'string', 'format': 'date-time'}
                }
            },
            'EmailMessage': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'subject': {'type': 'string'},
                    'sender': {'type': 'string'},
                    'recipient': {'type': 'string'},
                    'body': {'type': 'string'},
                    'timestamp': {'type': 'string', 'format': 'date-time'},
                    'thread_id': {'type': 'string'},
                    'classification': {'type': 'string'},
                    'intent': {'type': 'string'},
                    'confidence': {'type': 'number'},
                    'extracted_entities': {'type': 'array', 'items': {'type': 'object'}},
                    'action_items': {'type': 'array', 'items': {'type': 'string'}},
                    'metadata': {'type': 'object'}
                }
            },
            'EmailReply': {
                'type': 'object',
                'properties': {
                    'recipient': {'type': 'string'},
                    'subject': {'type': 'string'},
                    'body': {'type': 'string'},
                    'tone': {'type': 'string'},
                    'persona': {'type': 'string'},
                    'confidence': {'type': 'number'},
                    'requires_review': {'type': 'boolean'},
                    'metadata': {'type': 'object'}
                }
            },
            'EmailConfiguration': {
                'type': 'object',
                'properties': {
                    'provider': {'type': 'string'},
                    'host': {'type': 'string'},
                    'port': {'type': 'integer'},
                    'username': {'type': 'string'},
                    'password': {'type': 'string'},
                    'use_tls': {'type': 'boolean'},
                    'use_ssl': {'type': 'boolean'}
                }
            },
            'EmailSummary': {
                'type': 'object',
                'properties': {
                    'total_messages': {'type': 'integer'},
                    'classification_breakdown': {'type': 'object'},
                    'most_active_senders': {'type': 'array', 'items': {'type': 'string'}},
                    'total_replies_generated': {'type': 'integer'},
                    'reply_accuracy': {'type': 'number'},
                    'processing_time': {'type': 'number'},
                    'date_range': {'type': 'string'}
                }
            },
            'WritingStyleSettings': {
                'type': 'object',
                'properties': {
                    'consent': {'type': 'string', 'enum': ['enabled', 'disabled', 'ask_each_time']},
                    'auto_learn_from_sent': {'type': 'boolean'},
                    'learn_from_manual_edits': {'type': 'boolean'},
                    'preserve_privacy': {'type': 'boolean'},
                    'training_data_retention_days': {'type': 'integer'},
                    'min_confidence_threshold': {'type': 'number'},
                    'user_approval_required': {'type': 'boolean'}
                }
            },
            'TrainingDataEntry': {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string'},
                    'tone': {'type': 'string'},
                    'context': {'type': 'string'},
                    'confidence': {'type': 'number'},
                    'created_at': {'type': 'string', 'format': 'date-time'}
                }
            },
            'ConsentOption': {
                'type': 'object',
                'properties': {
                    'value': {'type': 'string'},
                    'label': {'type': 'string'},
                    'description': {'type': 'string'}
                }
            }
        }
    }
}