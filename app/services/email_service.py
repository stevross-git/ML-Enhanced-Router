"""
Email Service
Handles email processing, intelligence extraction, and communication
"""

import os
import re
import json
import email
import imaplib
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from flask import current_app
from sqlalchemy import func

from app.extensions import db
from app.models.user import User
from app.utils.exceptions import EmailError, ValidationError, ServiceError


class EmailService:
    """Service for handling email operations and intelligence"""
    
    def __init__(self):
        self.smtp_server = None
        self.smtp_port = 587
        self.smtp_username = None
        self.smtp_password = None
        self.imap_server = None
        self.imap_port = 993
        self.imap_username = None
        self.imap_password = None
        self.use_tls = True
        self.initialized = False
    
    def initialize(self, app):
        """Initialize the email service with app configuration"""
        try:
            self.smtp_server = app.config.get('SMTP_SERVER')
            self.smtp_port = app.config.get('SMTP_PORT', 587)
            self.smtp_username = app.config.get('SMTP_USERNAME')
            self.smtp_password = app.config.get('SMTP_PASSWORD')
            
            self.imap_server = app.config.get('IMAP_SERVER')
            self.imap_port = app.config.get('IMAP_PORT', 993)
            self.imap_username = app.config.get('IMAP_USERNAME')
            self.imap_password = app.config.get('IMAP_PASSWORD')
            
            self.use_tls = app.config.get('EMAIL_USE_TLS', True)
            
            if not all([self.smtp_server, self.smtp_username, self.smtp_password]):
                current_app.logger.warning("SMTP configuration incomplete - email sending disabled")
            
            if not all([self.imap_server, self.imap_username, self.imap_password]):
                current_app.logger.warning("IMAP configuration incomplete - email reading disabled")
            
            self.initialized = True
            current_app.logger.info("Email service initialized successfully")
            
        except Exception as e:
            current_app.logger.error(f"Email service initialization failed: {e}")
            self.initialized = False
    
    def send_email(self, to_addresses: List[str], subject: str, body: str, 
                   html_body: str = None, attachments: List[str] = None,
                   from_address: str = None) -> bool:
        """
        Send email to recipients
        
        Args:
            to_addresses: List of recipient email addresses
            subject: Email subject
            body: Plain text body
            html_body: HTML body (optional)
            attachments: List of file paths to attach (optional)
            from_address: Sender address (optional, uses default)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            EmailError: If sending fails
            ValidationError: If parameters are invalid
        """
        if not self.initialized:
            raise EmailError("Email service not initialized")
        
        if not all([self.smtp_server, self.smtp_username, self.smtp_password]):
            raise EmailError("SMTP configuration incomplete")
        
        try:
            if not to_addresses:
                raise ValidationError("No recipients specified")
            
            if not subject or not body:
                raise ValidationError("Subject and body are required")
            
            for email_addr in to_addresses:
                if not self._is_valid_email(email_addr):
                    raise ValidationError(f"Invalid email address: {email_addr}")
            
            from_address = from_address or self.smtp_username
            
            msg = MIMEMultipart('alternative')
            msg['From'] = from_address
            msg['To'] = ', '.join(to_addresses)
            msg['Subject'] = subject
            
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        self._add_attachment(msg, file_path)
                    else:
                        current_app.logger.warning(f"Attachment not found: {file_path}")
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            current_app.logger.info(f"Email sent successfully to {len(to_addresses)} recipients")
            return True
            
        except Exception as e:
            current_app.logger.error(f"Email sending error: {e}")
            if isinstance(e, (EmailError, ValidationError)):
                raise
            raise EmailError(f"Failed to send email: {str(e)}")
    
    def extract_email_intelligence(self, email_content: str) -> Dict[str, Any]:
        """
        Extract intelligence from email content
        
        Args:
            email_content: Email content to analyze
            
        Returns:
            Dict containing extracted intelligence
        """
        try:
            intelligence = {
                'sentiment': self._analyze_sentiment(email_content),
                'entities': self._extract_entities(email_content),
                'keywords': self._extract_keywords(email_content),
                'intent': self._classify_intent(email_content),
                'urgency': self._assess_urgency(email_content),
                'language': self._detect_language(email_content),
                'topics': self._extract_topics(email_content),
                'action_items': self._extract_action_items(email_content)
            }
            
            return intelligence
            
        except Exception as e:
            current_app.logger.error(f"Email intelligence extraction error: {e}")
            return {'error': str(e)}
    
    def process_training_consent(self, email_content: str, sender_email: str) -> Dict[str, Any]:
        """
        Process training consent from email
        
        Args:
            email_content: Email content
            sender_email: Sender's email address
            
        Returns:
            Dict containing consent processing results
        """
        try:
            consent_data = {
                'sender_email': sender_email,
                'timestamp': datetime.utcnow(),
                'consent_given': False,
                'consent_type': None,
                'extracted_data': {}
            }
            
            content_lower = email_content.lower()
            
            consent_keywords = ['consent', 'agree', 'permission', 'authorize', 'approve']
            training_keywords = ['training', 'learn', 'improve', 'data', 'model']
            
            has_consent = any(keyword in content_lower for keyword in consent_keywords)
            has_training = any(keyword in content_lower for keyword in training_keywords)
            
            if has_consent and has_training:
                consent_data['consent_given'] = True
                consent_data['consent_type'] = 'training'
            
            consent_data['extracted_data'] = {
                'full_content': email_content,
                'intelligence': self.extract_email_intelligence(email_content),
                'consent_keywords_found': [kw for kw in consent_keywords if kw in content_lower],
                'training_keywords_found': [kw for kw in training_keywords if kw in content_lower]
            }
            
            current_app.logger.info(f"Training consent processed: {sender_email} -> {consent_data['consent_given']}")
            return consent_data
            
        except Exception as e:
            current_app.logger.error(f"Training consent processing error: {e}")
            return {'error': str(e)}
    
    def get_email_stats(self) -> Dict[str, Any]:
        """Get email service statistics"""
        try:
            stats = {
                'smtp_configured': bool(self.smtp_server and self.smtp_username),
                'imap_configured': bool(self.imap_server and self.imap_username),
                'service_initialized': self.initialized,
                'recent_activity': {
                    'emails_sent': 0,
                    'emails_read': 0,
                    'notifications_sent': 0
                }
            }
            
            return stats
            
        except Exception as e:
            current_app.logger.error(f"Email stats error: {e}")
            return {'error': str(e)}
    
    def _is_valid_email(self, email_address: str) -> bool:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email_address) is not None
    
    def _add_attachment(self, msg: MIMEMultipart, file_path: str):
        """Add file attachment to email message"""
        try:
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(file_path)}'
            )
            
            msg.attach(part)
            
        except Exception as e:
            current_app.logger.warning(f"Failed to add attachment {file_path}: {e}")
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        positive_words = ['good', 'great', 'excellent', 'happy', 'pleased', 'satisfied']
        negative_words = ['bad', 'terrible', 'awful', 'angry', 'disappointed', 'frustrated']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = 0.7
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = 0.3
        else:
            sentiment = 'neutral'
            score = 0.5
        
        return {
            'sentiment': sentiment,
            'score': score,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count
        }
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        entities = []
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email_addr in emails:
            entities.append({'type': 'email', 'value': email_addr})
        
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            entities.append({'type': 'phone', 'value': phone})
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        import re
        from collections import Counter
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in words if word not in stop_words]
        
        word_counts = Counter(words)
        keywords = [word for word, count in word_counts.most_common(10)]
        
        return keywords
    
    def _classify_intent(self, text: str) -> str:
        """Classify intent of text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['question', 'help', 'how', 'what', 'why', 'when']):
            return 'inquiry'
        elif any(word in text_lower for word in ['complaint', 'problem', 'issue', 'wrong', 'error']):
            return 'complaint'
        elif any(word in text_lower for word in ['request', 'please', 'need', 'want', 'require']):
            return 'request'
        elif any(word in text_lower for word in ['thank', 'thanks', 'appreciate', 'grateful']):
            return 'appreciation'
        else:
            return 'general'
    
    def _assess_urgency(self, text: str) -> str:
        """Assess urgency of text"""
        text_lower = text.lower()
        
        urgent_words = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'important']
        if any(word in text_lower for word in urgent_words):
            return 'high'
        
        medium_words = ['soon', 'quickly', 'priority', 'needed']
        if any(word in text_lower for word in medium_words):
            return 'medium'
        
        return 'low'
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        return 'en'
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        topics = []
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['technical', 'code', 'software', 'system']):
            topics.append('technical')
        
        if any(word in text_lower for word in ['business', 'sales', 'marketing', 'revenue']):
            topics.append('business')
        
        if any(word in text_lower for word in ['support', 'help', 'assistance', 'service']):
            topics.append('support')
        
        return topics or ['general']
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from text"""
        action_items = []
        
        action_patterns = [
            r'please\s+([^.!?]+)',
            r'need\s+to\s+([^.!?]+)',
            r'should\s+([^.!?]+)',
            r'must\s+([^.!?]+)'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            action_items.extend(matches)
        
        return action_items[:5]


_email_service = None

def get_email_service() -> EmailService:
    """Get singleton email service instance"""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
        if current_app:
            _email_service.initialize(current_app)
    return _email_service

def init_email_service(app):
    """Initialize email service with Flask app"""
    service = get_email_service()
    service.initialize(app)
    return service
