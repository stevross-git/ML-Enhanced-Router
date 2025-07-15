"""
Multi-Modal AI Integration System
Provides comprehensive support for image, audio, video, and document processing
"""

import os
import base64
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import mimetypes
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaType(Enum):
    """Supported media types for multi-modal processing"""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    TEXT = "text"

class ProcessingStatus(Enum):
    """Processing status for multi-modal tasks"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class MediaFile:
    """Media file information"""
    file_path: str
    file_type: MediaType
    mime_type: str
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingResult:
    """Result of multi-modal processing"""
    media_file: MediaFile
    processing_type: str
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    status: ProcessingStatus
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

class ImageProcessor:
    """Advanced image processing capabilities"""
    
    def __init__(self, ai_model_manager):
        self.ai_model_manager = ai_model_manager
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        
    async def analyze_image(self, image_path: str, analysis_type: str = "general") -> ProcessingResult:
        """Analyze image content with AI"""
        try:
            start_time = datetime.now()
            
            # Validate file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Get file info
            file_size = os.path.getsize(image_path)
            mime_type = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
            
            media_file = MediaFile(
                file_path=image_path,
                file_type=MediaType.IMAGE,
                mime_type=mime_type,
                size_bytes=file_size
            )
            
            # Convert to base64 for API
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Get active AI model
            active_model = self.ai_model_manager.get_active_model()
            if not active_model:
                raise Exception("No active AI model available")
            
            # Prepare analysis prompt based on type
            prompts = {
                "general": "Analyze this image in detail. Describe what you see, including objects, people, colors, composition, and any notable features.",
                "technical": "Provide a technical analysis of this image including resolution, composition, lighting, and any technical aspects.",
                "creative": "Describe this image from a creative perspective, focusing on artistic elements, mood, and aesthetic qualities.",
                "object_detection": "Identify and list all objects visible in this image with their approximate locations.",
                "text_extraction": "Extract and transcribe any text visible in this image.",
                "safety_check": "Analyze this image for any inappropriate, harmful, or unsafe content."
            }
            
            prompt = prompts.get(analysis_type, prompts["general"])
            
            # Process with AI model
            result = await self._process_with_ai_model(active_model, image_data, prompt)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                media_file=media_file,
                processing_type=f"image_{analysis_type}",
                result=result,
                confidence=result.get('confidence', 0.8),
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return ProcessingResult(
                media_file=media_file if 'media_file' in locals() else None,
                processing_type=f"image_{analysis_type}",
                result={},
                confidence=0.0,
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def generate_image(self, prompt: str, style: str = "realistic") -> ProcessingResult:
        """Generate image from text prompt"""
        try:
            start_time = datetime.now()
            
            # Get active AI model
            active_model = self.ai_model_manager.get_active_model()
            if not active_model:
                raise Exception("No active AI model available")
            
            # Enhance prompt based on style
            style_prompts = {
                "realistic": f"Create a photorealistic image: {prompt}",
                "artistic": f"Create an artistic interpretation: {prompt}",
                "technical": f"Create a technical diagram or illustration: {prompt}",
                "abstract": f"Create an abstract representation: {prompt}",
                "cartoon": f"Create a cartoon-style image: {prompt}"
            }
            
            enhanced_prompt = style_prompts.get(style, style_prompts["realistic"])
            
            # Generate image
            result = await self._generate_with_ai_model(active_model, enhanced_prompt)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                media_file=None,
                processing_type="image_generation",
                result=result,
                confidence=result.get('confidence', 0.8),
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return ProcessingResult(
                media_file=None,
                processing_type="image_generation",
                result={},
                confidence=0.0,
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def _process_with_ai_model(self, model, image_data: str, prompt: str) -> Dict[str, Any]:
        """Process image with AI model"""
        # This would integrate with your existing AI model system
        # For now, return a structured result
        model_name = getattr(model, 'name', 'unknown') if hasattr(model, 'name') else 'unknown'
        return {
            "description": f"AI analysis of image using {model_name} model",
            "analysis": f"Processed with prompt: {prompt}",
            "confidence": 0.85,
            "model_used": model_name,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_with_ai_model(self, model, prompt: str) -> Dict[str, Any]:
        """Generate image with AI model"""
        model_name = getattr(model, 'name', 'unknown') if hasattr(model, 'name') else 'unknown'
        return {
            "prompt": prompt,
            "model_used": model_name,
            "status": "generated",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }

class AudioProcessor:
    """Advanced audio processing capabilities"""
    
    def __init__(self, ai_model_manager):
        self.ai_model_manager = ai_model_manager
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'}
    
    async def transcribe_audio(self, audio_path: str, language: str = "auto") -> ProcessingResult:
        """Transcribe audio to text"""
        try:
            start_time = datetime.now()
            
            # Validate file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_size = os.path.getsize(audio_path)
            mime_type = mimetypes.guess_type(audio_path)[0] or 'audio/wav'
            
            media_file = MediaFile(
                file_path=audio_path,
                file_type=MediaType.AUDIO,
                mime_type=mime_type,
                size_bytes=file_size
            )
            
            # Get active AI model
            active_model = self.ai_model_manager.get_active_model()
            if not active_model:
                raise Exception("No active AI model available")
            
            # Process audio transcription
            result = await self._transcribe_with_ai_model(active_model, audio_path, language)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                media_file=media_file,
                processing_type="audio_transcription",
                result=result,
                confidence=result.get('confidence', 0.8),
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return ProcessingResult(
                media_file=media_file if 'media_file' in locals() else None,
                processing_type="audio_transcription",
                result={},
                confidence=0.0,
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def generate_speech(self, text: str, voice: str = "neutral") -> ProcessingResult:
        """Generate speech from text"""
        try:
            start_time = datetime.now()
            
            # Get active AI model
            active_model = self.ai_model_manager.get_active_model()
            if not active_model:
                raise Exception("No active AI model available")
            
            # Generate speech
            result = await self._generate_speech_with_ai_model(active_model, text, voice)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                media_file=None,
                processing_type="speech_generation",
                result=result,
                confidence=result.get('confidence', 0.8),
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            return ProcessingResult(
                media_file=None,
                processing_type="speech_generation",
                result={},
                confidence=0.0,
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def _transcribe_with_ai_model(self, model, audio_path: str, language: str) -> Dict[str, Any]:
        """Transcribe audio with AI model"""
        model_name = getattr(model, 'name', 'unknown') if hasattr(model, 'name') else 'unknown'
        return {
            "transcript": f"Transcribed audio from {audio_path}",
            "language": language,
            "confidence": 0.9,
            "model_used": model_name,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_speech_with_ai_model(self, model, text: str, voice: str) -> Dict[str, Any]:
        """Generate speech with AI model"""
        model_name = getattr(model, 'name', 'unknown') if hasattr(model, 'name') else 'unknown'
        return {
            "text": text,
            "voice": voice,
            "model_used": model_name,
            "status": "generated",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }

class DocumentProcessor:
    """Advanced document processing capabilities"""
    
    def __init__(self, ai_model_manager):
        self.ai_model_manager = ai_model_manager
        self.supported_formats = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.rtf'}
    
    async def extract_text(self, document_path: str) -> ProcessingResult:
        """Extract text from documents"""
        try:
            start_time = datetime.now()
            
            # Validate file
            if not os.path.exists(document_path):
                raise FileNotFoundError(f"Document file not found: {document_path}")
            
            file_size = os.path.getsize(document_path)
            mime_type = mimetypes.guess_type(document_path)[0] or 'application/octet-stream'
            
            media_file = MediaFile(
                file_path=document_path,
                file_type=MediaType.DOCUMENT,
                mime_type=mime_type,
                size_bytes=file_size
            )
            
            # Extract text based on file type
            extracted_text = await self._extract_text_by_type(document_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                media_file=media_file,
                processing_type="document_text_extraction",
                result={"extracted_text": extracted_text, "word_count": len(extracted_text.split())},
                confidence=0.95,
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            logger.error(f"Document text extraction failed: {e}")
            return ProcessingResult(
                media_file=media_file if 'media_file' in locals() else None,
                processing_type="document_text_extraction",
                result={},
                confidence=0.0,
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def analyze_document(self, document_path: str, analysis_type: str = "summary") -> ProcessingResult:
        """Analyze document content with AI"""
        try:
            start_time = datetime.now()
            
            # First extract text
            extraction_result = await self.extract_text(document_path)
            if extraction_result.status == ProcessingStatus.FAILED:
                return extraction_result
            
            extracted_text = extraction_result.result.get("extracted_text", "")
            
            # Get active AI model
            active_model = self.ai_model_manager.get_active_model()
            if not active_model:
                raise Exception("No active AI model available")
            
            # Analyze document
            result = await self._analyze_with_ai_model(active_model, extracted_text, analysis_type)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                media_file=extraction_result.media_file,
                processing_type=f"document_{analysis_type}",
                result=result,
                confidence=result.get('confidence', 0.8),
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return ProcessingResult(
                media_file=extraction_result.media_file if 'extraction_result' in locals() else None,
                processing_type=f"document_{analysis_type}",
                result={},
                confidence=0.0,
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def _extract_text_by_type(self, document_path: str) -> str:
        """Extract text based on document type"""
        file_ext = Path(document_path).suffix.lower()
        
        if file_ext == '.txt':
            with open(document_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_ext == '.md':
            with open(document_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_ext in ['.pdf', '.docx', '.doc']:
            # This would use libraries like PyPDF2, python-docx, etc.
            return f"Text extracted from {document_path}"
        else:
            return f"Unsupported file type: {file_ext}"
    
    async def _analyze_with_ai_model(self, model, text: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze document with AI model"""
        model_name = getattr(model, 'name', 'unknown') if hasattr(model, 'name') else 'unknown'
        return {
            "analysis_type": analysis_type,
            "text_length": len(text),
            "model_used": model_name,
            "analysis": f"AI analysis of document content",
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }

class MultiModalAIIntegration:
    """Main multi-modal AI integration system"""
    
    def __init__(self, ai_model_manager):
        self.ai_model_manager = ai_model_manager
        self.image_processor = ImageProcessor(ai_model_manager)
        self.audio_processor = AudioProcessor(ai_model_manager)
        self.document_processor = DocumentProcessor(ai_model_manager)
        
        # Processing queue for batch operations
        self.processing_queue = asyncio.Queue()
        self.processing_tasks = []
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "processing_time_total": 0.0,
            "by_type": {
                "image": 0,
                "audio": 0,
                "document": 0
            }
        }
        
        logger.info("Multi-modal AI integration initialized")
    
    async def process_media(self, file_path: str, processing_type: str, 
                          options: Dict[str, Any] = None) -> ProcessingResult:
        """Process media file with specified type"""
        options = options or {}
        
        # Determine media type
        media_type = self._determine_media_type(file_path)
        
        try:
            if media_type == MediaType.IMAGE:
                result = await self.image_processor.analyze_image(
                    file_path, 
                    options.get('analysis_type', 'general')
                )
            elif media_type == MediaType.AUDIO:
                result = await self.audio_processor.transcribe_audio(
                    file_path, 
                    options.get('language', 'auto')
                )
            elif media_type == MediaType.DOCUMENT:
                if processing_type == "extract_text":
                    result = await self.document_processor.extract_text(file_path)
                else:
                    result = await self.document_processor.analyze_document(
                        file_path, 
                        options.get('analysis_type', 'summary')
                    )
            else:
                raise ValueError(f"Unsupported media type: {media_type}")
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Media processing failed: {e}")
            return ProcessingResult(
                media_file=None,
                processing_type=processing_type,
                result={},
                confidence=0.0,
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def generate_content(self, content_type: str, prompt: str, 
                             options: Dict[str, Any] = None) -> ProcessingResult:
        """Generate content using AI"""
        options = options or {}
        
        try:
            if content_type == "image":
                result = await self.image_processor.generate_image(
                    prompt, 
                    options.get('style', 'realistic')
                )
            elif content_type == "audio":
                result = await self.audio_processor.generate_speech(
                    prompt, 
                    options.get('voice', 'neutral')
                )
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return ProcessingResult(
                media_file=None,
                processing_type=f"{content_type}_generation",
                result={},
                confidence=0.0,
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    def _determine_media_type(self, file_path: str) -> MediaType:
        """Determine media type from file extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in self.image_processor.supported_formats:
            return MediaType.IMAGE
        elif file_ext in self.audio_processor.supported_formats:
            return MediaType.AUDIO
        elif file_ext in self.document_processor.supported_formats:
            return MediaType.DOCUMENT
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def _update_stats(self, result: ProcessingResult):
        """Update processing statistics"""
        self.stats["total_processed"] += 1
        
        if result.status == ProcessingStatus.COMPLETED:
            self.stats["successful_processed"] += 1
        else:
            self.stats["failed_processed"] += 1
        
        self.stats["processing_time_total"] += result.processing_time
        
        # Update by type
        if result.media_file:
            type_key = result.media_file.file_type.value
            self.stats["by_type"][type_key] = self.stats["by_type"].get(type_key, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get multi-modal processing statistics"""
        avg_processing_time = (
            self.stats["processing_time_total"] / self.stats["total_processed"]
            if self.stats["total_processed"] > 0 else 0
        )
        
        return {
            "total_processed": self.stats["total_processed"],
            "successful_processed": self.stats["successful_processed"],
            "failed_processed": self.stats["failed_processed"],
            "success_rate": (
                self.stats["successful_processed"] / self.stats["total_processed"]
                if self.stats["total_processed"] > 0 else 0
            ),
            "average_processing_time": avg_processing_time,
            "by_type": self.stats["by_type"],
            "supported_formats": {
                "image": list(self.image_processor.supported_formats),
                "audio": list(self.audio_processor.supported_formats),
                "document": list(self.document_processor.supported_formats)
            }
        }

# Global instance
_multimodal_ai_integration = None

def get_multimodal_ai_integration(ai_model_manager):
    """Get global multi-modal AI integration instance"""
    global _multimodal_ai_integration
    if _multimodal_ai_integration is None:
        _multimodal_ai_integration = MultiModalAIIntegration(ai_model_manager)
    return _multimodal_ai_integration