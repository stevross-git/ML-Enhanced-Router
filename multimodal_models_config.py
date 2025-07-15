"""
Comprehensive Multi-Modal AI Models Configuration
Defines all major enterprise AI providers and local models with their capabilities
"""

from ai_models import AIModel, AIProvider, ModelCapability

def get_comprehensive_multimodal_models():
    """Get comprehensive list of multi-modal AI models from all major providers"""
    
    return [
        # ===== OPENAI MODELS =====
        AIModel(
            id="gpt-4o",
            name="GPT-4o (Multi-modal)",
            provider=AIProvider.OPENAI,
            model_name="gpt-4o",
            endpoint="https://api.openai.com/v1/chat/completions",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            context_window=128000,
            cost_per_1k_tokens=0.03,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.MULTIMODAL,
                ModelCapability.REASONING,
                ModelCapability.FUNCTION_CALLING
            ],
            supports_vision=True,
            supports_functions=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            specialized_tasks=["vision", "analysis", "reasoning", "code_generation"]
        ),
        
        AIModel(
            id="gpt-4-turbo",
            name="GPT-4 Turbo (Vision)",
            provider=AIProvider.OPENAI,
            model_name="gpt-4-turbo",
            endpoint="https://api.openai.com/v1/chat/completions",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            context_window=128000,
            cost_per_1k_tokens=0.01,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.FUNCTION_CALLING
            ],
            supports_vision=True,
            supports_functions=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"]
        ),
        
        AIModel(
            id="dall-e-3",
            name="DALL-E 3 (Image Generation)",
            provider=AIProvider.OPENAI,
            model_name="dall-e-3",
            endpoint="https://api.openai.com/v1/images/generations",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4000,
            context_window=4000,
            cost_per_1k_tokens=0.04,
            capabilities=[ModelCapability.IMAGE_GENERATION],
            model_type="image",
            input_modalities=["text"],
            output_modalities=["image"],
            specialized_tasks=["image_creation", "artistic_generation"]
        ),
        
        AIModel(
            id="whisper-1",
            name="Whisper (Audio Transcription)",
            provider=AIProvider.OPENAI,
            model_name="whisper-1",
            endpoint="https://api.openai.com/v1/audio/transcriptions",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            context_window=25000,
            cost_per_1k_tokens=0.006,
            capabilities=[ModelCapability.AUDIO_TRANSCRIPTION],
            supports_audio=True,
            model_type="audio",
            input_modalities=["audio"],
            output_modalities=["text"],
            specialized_tasks=["transcription", "speech_to_text"]
        ),
        
        AIModel(
            id="tts-1",
            name="TTS-1 (Text to Speech)",
            provider=AIProvider.OPENAI,
            model_name="tts-1",
            endpoint="https://api.openai.com/v1/audio/speech",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            context_window=4096,
            cost_per_1k_tokens=0.015,
            capabilities=[ModelCapability.AUDIO_GENERATION],
            supports_audio=True,
            model_type="audio",
            input_modalities=["text"],
            output_modalities=["audio"],
            specialized_tasks=["speech_generation", "text_to_speech"]
        ),
        
        # ===== ANTHROPIC MODELS =====
        AIModel(
            id="claude-sonnet-4",
            name="Claude Sonnet 4 (Multi-modal)",
            provider=AIProvider.ANTHROPIC,
            model_name="claude-sonnet-4-20250514",
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=4096,
            context_window=200000,
            cost_per_1k_tokens=0.015,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.MULTIMODAL,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            specialized_tasks=["reasoning", "analysis", "code_generation", "writing"]
        ),
        
        AIModel(
            id="claude-3-5-sonnet",
            name="Claude 3.5 Sonnet (Vision)",
            provider=AIProvider.ANTHROPIC,
            model_name="claude-3-5-sonnet-20241022",
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=4096,
            context_window=200000,
            cost_per_1k_tokens=0.003,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"]
        ),
        
        # ===== GOOGLE MODELS =====
        AIModel(
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash (Multi-modal)",
            provider=AIProvider.GOOGLE,
            model_name="gemini-2.5-flash",
            endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            api_key_env="GEMINI_API_KEY",
            max_tokens=8192,
            context_window=1048576,
            cost_per_1k_tokens=0.000125,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.AUDIO_TRANSCRIPTION,
                ModelCapability.VIDEO_ANALYSIS,
                ModelCapability.MULTIMODAL,
                ModelCapability.FUNCTION_CALLING
            ],
            supports_vision=True,
            supports_audio=True,
            supports_video=True,
            supports_functions=True,
            model_type="multimodal",
            input_modalities=["text", "image", "audio", "video"],
            output_modalities=["text"],
            specialized_tasks=["multimodal_analysis", "fast_processing", "function_calling"]
        ),
        
        AIModel(
            id="gemini-2.5-pro",
            name="Gemini 2.5 Pro (Multi-modal)",
            provider=AIProvider.GOOGLE,
            model_name="gemini-2.5-pro",
            endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            api_key_env="GEMINI_API_KEY",
            max_tokens=8192,
            context_window=2097152,
            cost_per_1k_tokens=0.00125,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.AUDIO_TRANSCRIPTION,
                ModelCapability.VIDEO_ANALYSIS,
                ModelCapability.MULTIMODAL,
                ModelCapability.REASONING,
                ModelCapability.FUNCTION_CALLING
            ],
            supports_vision=True,
            supports_audio=True,
            supports_video=True,
            supports_functions=True,
            model_type="multimodal",
            input_modalities=["text", "image", "audio", "video"],
            output_modalities=["text"],
            specialized_tasks=["advanced_reasoning", "multimodal_analysis", "complex_tasks"]
        ),
        
        AIModel(
            id="gemini-2.0-flash-image",
            name="Gemini 2.0 Flash (Image Generation)",
            provider=AIProvider.GOOGLE,
            model_name="gemini-2.0-flash-preview-image-generation",
            endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-preview:generateContent",
            api_key_env="GEMINI_API_KEY",
            max_tokens=8192,
            context_window=1048576,
            cost_per_1k_tokens=0.000125,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_GENERATION,
                ModelCapability.MULTIMODAL
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text"],
            output_modalities=["text", "image"],
            specialized_tasks=["image_generation", "creative_content"]
        ),
        
        # ===== XAI MODELS =====
        AIModel(
            id="grok-2-vision",
            name="Grok 2 Vision",
            provider=AIProvider.XAI,
            model_name="grok-2-vision-1212",
            endpoint="https://api.x.ai/v1/chat/completions",
            api_key_env="XAI_API_KEY",
            max_tokens=4096,
            context_window=8192,
            cost_per_1k_tokens=0.002,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.MULTIMODAL,
                ModelCapability.REASONING
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            specialized_tasks=["real_time_analysis", "web_search", "current_events"]
        ),
        
        AIModel(
            id="grok-2",
            name="Grok 2",
            provider=AIProvider.XAI,
            model_name="grok-2-1212",
            endpoint="https://api.x.ai/v1/chat/completions",
            api_key_env="XAI_API_KEY",
            max_tokens=4096,
            context_window=131072,
            cost_per_1k_tokens=0.002,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            specialized_tasks=["reasoning", "analysis", "real_time_info"]
        ),
        
        # ===== AZURE OPENAI MODELS =====
        AIModel(
            id="azure-gpt-4o",
            name="Azure GPT-4o (Multi-modal)",
            provider=AIProvider.AZURE,
            model_name="gpt-4o",
            endpoint="https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions",
            api_key_env="AZURE_OPENAI_API_KEY",
            max_tokens=4096,
            context_window=128000,
            cost_per_1k_tokens=0.03,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.MULTIMODAL,
                ModelCapability.FUNCTION_CALLING
            ],
            supports_vision=True,
            supports_functions=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            deployment_type="cloud",
            region="eastus",
            specialized_tasks=["enterprise_ready", "compliance", "security"]
        ),
        
        # ===== AWS BEDROCK MODELS =====
        AIModel(
            id="bedrock-claude-3-sonnet",
            name="Bedrock Claude 3 Sonnet",
            provider=AIProvider.AWS_BEDROCK,
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            endpoint="https://bedrock-runtime.{region}.amazonaws.com/model/{model-id}/invoke",
            api_key_env="AWS_ACCESS_KEY_ID",
            max_tokens=4096,
            context_window=200000,
            cost_per_1k_tokens=0.003,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.REASONING
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            deployment_type="cloud",
            region="us-east-1",
            specialized_tasks=["enterprise_aws", "compliance", "security"]
        ),
        
        # ===== REPLICATE MODELS =====
        AIModel(
            id="replicate-llava",
            name="Replicate LLaVA (Vision)",
            provider=AIProvider.REPLICATE,
            model_name="yorickvp/llava-13b",
            endpoint="https://api.replicate.com/v1/predictions",
            api_key_env="REPLICATE_API_TOKEN",
            max_tokens=4096,
            context_window=2048,
            cost_per_1k_tokens=0.0005,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.MULTIMODAL
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["open_source", "cost_effective"]
        ),
        
        AIModel(
            id="replicate-sdxl",
            name="Replicate SDXL (Image Generation)",
            provider=AIProvider.REPLICATE,
            model_name="stability-ai/sdxl",
            endpoint="https://api.replicate.com/v1/predictions",
            api_key_env="REPLICATE_API_TOKEN",
            max_tokens=512,
            context_window=512,
            cost_per_1k_tokens=0.0027,
            capabilities=[ModelCapability.IMAGE_GENERATION],
            model_type="image",
            input_modalities=["text"],
            output_modalities=["image"],
            deployment_type="cloud",
            specialized_tasks=["artistic_generation", "high_quality_images"]
        ),
        
        # ===== HUGGING FACE MODELS =====
        AIModel(
            id="hf-llava-1.5",
            name="Hugging Face LLaVA 1.5",
            provider=AIProvider.HUGGINGFACE,
            model_name="llava-hf/llava-1.5-7b-hf",
            endpoint="https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf",
            api_key_env="HUGGINGFACE_API_KEY",
            max_tokens=2048,
            context_window=2048,
            cost_per_1k_tokens=0.0002,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.MULTIMODAL
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["open_source", "research", "experimentation"]
        ),
        
        AIModel(
            id="hf-whisper-large",
            name="Hugging Face Whisper Large",
            provider=AIProvider.HUGGINGFACE,
            model_name="openai/whisper-large-v3",
            endpoint="https://api-inference.huggingface.co/models/openai/whisper-large-v3",
            api_key_env="HUGGINGFACE_API_KEY",
            max_tokens=448,
            context_window=30000,
            cost_per_1k_tokens=0.0001,
            capabilities=[ModelCapability.AUDIO_TRANSCRIPTION],
            supports_audio=True,
            model_type="audio",
            input_modalities=["audio"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["transcription", "multilingual", "open_source"]
        ),
        
        # ===== GROQ MODELS =====
        AIModel(
            id="groq-llama-3.1-70b",
            name="Groq Llama 3.1 70B (Vision)",
            provider=AIProvider.GROQ,
            model_name="llama-3.1-70b-versatile",
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key_env="GROQ_API_KEY",
            max_tokens=4096,
            context_window=131072,
            cost_per_1k_tokens=0.00059,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["ultra_fast_inference", "cost_effective", "open_source"]
        ),
        
        AIModel(
            id="groq-llava-1.5-7b",
            name="Groq LLaVA 1.5 7B (Vision)",
            provider=AIProvider.GROQ,
            model_name="llava-v1.5-7b-4096-preview",
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key_env="GROQ_API_KEY",
            max_tokens=4096,
            context_window=4096,
            cost_per_1k_tokens=0.00059,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.MULTIMODAL
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["ultra_fast_inference", "vision_analysis", "open_source"]
        ),
        
        # ===== TOGETHER AI MODELS =====
        AIModel(
            id="together-llama-3.1-70b",
            name="Together Llama 3.1 70B (Vision)",
            provider=AIProvider.TOGETHER,
            model_name="meta-llama/Llama-3.1-70B-Instruct-Turbo",
            endpoint="https://api.together.xyz/v1/chat/completions",
            api_key_env="TOGETHER_API_KEY",
            max_tokens=4096,
            context_window=131072,
            cost_per_1k_tokens=0.0009,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["open_source", "customizable", "fast_inference"]
        ),
        
        AIModel(
            id="together-llava-next",
            name="Together LLaVA-NeXT (Vision)",
            provider=AIProvider.TOGETHER,
            model_name="NousResearch/Nous-Hermes-2-Vision-Alpha",
            endpoint="https://api.together.xyz/v1/chat/completions",
            api_key_env="TOGETHER_API_KEY",
            max_tokens=4096,
            context_window=4096,
            cost_per_1k_tokens=0.0008,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.MULTIMODAL
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["open_source", "vision_analysis", "customizable"]
        ),
        
        # ===== FIREWORKS AI MODELS =====
        AIModel(
            id="fireworks-llama-3.1-70b",
            name="Fireworks Llama 3.1 70B",
            provider=AIProvider.FIREWORKS,
            model_name="accounts/fireworks/models/llama-v3p1-70b-instruct",
            endpoint="https://api.fireworks.ai/inference/v1/chat/completions",
            api_key_env="FIREWORKS_API_KEY",
            max_tokens=4096,
            context_window=131072,
            cost_per_1k_tokens=0.0009,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["fast_inference", "open_source", "cost_effective"]
        ),
        
        # ===== OLLAMA LOCAL MODELS =====
        AIModel(
            id="ollama-llama3.1",
            name="Ollama Llama 3.1 (Local)",
            provider=AIProvider.OLLAMA,
            model_name="llama3.1",
            endpoint="http://localhost:11434/api/generate",
            api_key_env="",
            max_tokens=4096,
            context_window=131072,
            cost_per_1k_tokens=0.0,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="local",
            specialized_tasks=["privacy", "offline", "no_cost", "local_deployment"]
        ),
        
        AIModel(
            id="ollama-llava",
            name="Ollama LLaVA (Local Vision)",
            provider=AIProvider.OLLAMA,
            model_name="llava",
            endpoint="http://localhost:11434/api/generate",
            api_key_env="",
            max_tokens=4096,
            context_window=4096,
            cost_per_1k_tokens=0.0,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.MULTIMODAL
            ],
            supports_vision=True,
            model_type="multimodal",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            deployment_type="local",
            specialized_tasks=["privacy", "offline", "no_cost", "vision_analysis"]
        ),
        
        AIModel(
            id="ollama-mistral",
            name="Ollama Mistral (Local)",
            provider=AIProvider.OLLAMA,
            model_name="mistral",
            endpoint="http://localhost:11434/api/generate",
            api_key_env="",
            max_tokens=4096,
            context_window=32768,
            cost_per_1k_tokens=0.0,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="local",
            specialized_tasks=["privacy", "offline", "no_cost", "multilingual"]
        ),
        
        AIModel(
            id="ollama-codellama",
            name="Ollama CodeLlama (Local)",
            provider=AIProvider.OLLAMA,
            model_name="codellama",
            endpoint="http://localhost:11434/api/generate",
            api_key_env="",
            max_tokens=4096,
            context_window=16384,
            cost_per_1k_tokens=0.0,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="local",
            specialized_tasks=["privacy", "offline", "no_cost", "code_generation"]
        ),
        
        # ===== DEEPSEEK MODELS =====
        AIModel(
            id="deepseek-chat",
            name="DeepSeek Chat",
            provider=AIProvider.DEEPSEEK,
            model_name="deepseek-chat",
            endpoint="https://api.deepseek.com/v1/chat/completions",
            api_key_env="DEEPSEEK_API_KEY",
            max_tokens=4096,
            context_window=32768,
            cost_per_1k_tokens=0.00014,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["reasoning", "mathematics", "code_generation", "cost_effective"]
        ),
        
        AIModel(
            id="deepseek-coder",
            name="DeepSeek Coder",
            provider=AIProvider.DEEPSEEK,
            model_name="deepseek-coder",
            endpoint="https://api.deepseek.com/v1/chat/completions",
            api_key_env="DEEPSEEK_API_KEY",
            max_tokens=4096,
            context_window=16384,
            cost_per_1k_tokens=0.00014,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["code_generation", "programming", "debugging", "cost_effective"]
        ),
        
        # ===== CEREBRAS MODELS =====
        AIModel(
            id="cerebras-llama3.1-70b",
            name="Cerebras Llama 3.1 70B",
            provider=AIProvider.CEREBRAS,
            model_name="llama3.1-70b",
            endpoint="https://api.cerebras.ai/v1/chat/completions",
            api_key_env="CEREBRAS_API_KEY",
            max_tokens=4096,
            context_window=131072,
            cost_per_1k_tokens=0.0006,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["ultra_fast_inference", "high_throughput", "cost_effective"]
        ),
        
        # ===== PERPLEXITY MODELS =====
        AIModel(
            id="perplexity-sonar-small",
            name="Perplexity Sonar Small (Online)",
            provider=AIProvider.PERPLEXITY,
            model_name="llama-3.1-sonar-small-128k-online",
            endpoint="https://api.perplexity.ai/chat/completions",
            api_key_env="PERPLEXITY_API_KEY",
            max_tokens=4096,
            context_window=131072,
            cost_per_1k_tokens=0.0002,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["real_time_search", "current_information", "web_search"]
        ),
        
        AIModel(
            id="perplexity-sonar-large",
            name="Perplexity Sonar Large (Online)",
            provider=AIProvider.PERPLEXITY,
            model_name="llama-3.1-sonar-large-128k-online",
            endpoint="https://api.perplexity.ai/chat/completions",
            api_key_env="PERPLEXITY_API_KEY",
            max_tokens=4096,
            context_window=131072,
            cost_per_1k_tokens=0.001,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING
            ],
            model_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            deployment_type="cloud",
            specialized_tasks=["real_time_search", "current_information", "web_search", "advanced_reasoning"]
        ),
        
        # ===== ELEVENLABS MODELS =====
        AIModel(
            id="elevenlabs-tts-multilingual",
            name="ElevenLabs TTS Multilingual",
            provider=AIProvider.ELEVENLABS,
            model_name="eleven_multilingual_v2",
            endpoint="https://api.elevenlabs.io/v1/text-to-speech",
            api_key_env="ELEVENLABS_API_KEY",
            max_tokens=5000,
            context_window=5000,
            cost_per_1k_tokens=0.18,
            capabilities=[
                ModelCapability.AUDIO_GENERATION
            ],
            supports_audio=True,
            model_type="audio",
            input_modalities=["text"],
            output_modalities=["audio"],
            deployment_type="cloud",
            specialized_tasks=["speech_synthesis", "voice_cloning", "multilingual_tts"]
        ),
        
        AIModel(
            id="elevenlabs-tts-turbo",
            name="ElevenLabs TTS Turbo",
            provider=AIProvider.ELEVENLABS,
            model_name="eleven_turbo_v2",
            endpoint="https://api.elevenlabs.io/v1/text-to-speech",
            api_key_env="ELEVENLABS_API_KEY",
            max_tokens=5000,
            context_window=5000,
            cost_per_1k_tokens=0.20,
            capabilities=[
                ModelCapability.AUDIO_GENERATION
            ],
            supports_audio=True,
            model_type="audio",
            input_modalities=["text"],
            output_modalities=["audio"],
            deployment_type="cloud",
            specialized_tasks=["fast_speech_synthesis", "low_latency_tts"]
        ),
        
        AIModel(
            id="elevenlabs-tts-english",
            name="ElevenLabs TTS English",
            provider=AIProvider.ELEVENLABS,
            model_name="eleven_monolingual_v1",
            endpoint="https://api.elevenlabs.io/v1/text-to-speech",
            api_key_env="ELEVENLABS_API_KEY",
            max_tokens=5000,
            context_window=5000,
            cost_per_1k_tokens=0.15,
            capabilities=[
                ModelCapability.AUDIO_GENERATION
            ],
            supports_audio=True,
            model_type="audio",
            input_modalities=["text"],
            output_modalities=["audio"],
            deployment_type="cloud",
            specialized_tasks=["english_tts", "high_quality_speech"]
        ),
    ]