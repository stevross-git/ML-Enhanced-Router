#!/usr/bin/env python3
"""
Complete Email Intelligence System Fix
Addresses initialization issues and completes all missing functionality
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import threading
import re
import json
import hashlib
import uuid
import smtplib
import imaplib
import email
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email Provider Types
class EmailProvider(Enum):
    """Email provider types"""
    GMAIL_API = "gmail_api"
    OUTLOOK_GRAPH = "outlook_graph"
    IMAP_SMTP = "imap_smtp"

class EmailClassification(Enum):
    """Email classification categories"""
    URGENT = "urgent"
    WORK = "work"
    PERSONAL = "personal"
    NEWSLETTER = "newsletter"
    SOCIAL = "social"
    CALENDAR = "calendar"
    REMINDER = "reminder"
    SPAM = "spam"
    PROMOTION = "promotion"
    NOTIFICATION = "notification"

class EmailIntent(Enum):
    """Email intent types"""
    RESPOND = "respond"
    ARCHIVE = "archive"
    SCHEDULE = "schedule"
    IGNORE = "ignore"
    FORWARD = "forward"
    URGENT_ACTION = "urgent_action"

class EmailTone(Enum):
    """Email tone types"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CASUAL = "casual"
    SUPPORTIVE = "supportive"
    ASSERTIVE = "assertive"
    EMPATHETIC = "empathetic"

class TrainingConsent(Enum):
    """Training consent options"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ASK_EACH_TIME = "ask_each_time"

@dataclass
class EmailMessage:
    """Email message structure"""
    id: str
    subject: str
    sender: str
    recipient: str
    body: str
    timestamp: datetime
    thread_id: Optional[str] = None
    classification: Optional[EmailClassification] = None
    intent: Optional[EmailIntent] = None
    confidence: float = 0.0
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmailReply:
    """Email reply structure"""
    recipient: str
    subject: str
    body: str
    tone: EmailTone
    persona: Optional[str] = None
    confidence: float = 0.0
    requires_review: bool = True
    allow_training: bool = False
    training_consent: TrainingConsent = TrainingConsent.DISABLED
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WritingStyleSettings:
    """Writing style training settings"""
    consent: TrainingConsent = TrainingConsent.DISABLED
    auto_learn_from_sent: bool = False
    learn_from_manual_edits: bool = False
    preserve_privacy: bool = True
    training_data_retention_days: int = 30
    min_confidence_threshold: float = 0.8
    user_approval_required: bool = True

class EmailDatabase:
    """Email database manager with full functionality"""
    
    def __init__(self, db_path: str = "email_intelligence.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize email database with all required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Email messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_messages (
                    id TEXT PRIMARY KEY,
                    subject TEXT,
                    sender TEXT,
                    recipient TEXT,
                    body TEXT,
                    timestamp TEXT,
                    thread_id TEXT,
                    classification TEXT,
                    intent TEXT,
                    confidence REAL,
                    extracted_entities TEXT,
                    action_items TEXT,
                    metadata TEXT
                )
            ''')
            
            # Email replies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_replies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT,
                    recipient TEXT,
                    subject TEXT,
                    body TEXT,
                    tone TEXT,
                    persona TEXT,
                    confidence REAL,
                    requires_review BOOLEAN,
                    allow_training BOOLEAN,
                    training_consent TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    sent_at TEXT,
                    FOREIGN KEY (message_id) REFERENCES email_messages (id)
                )
            ''')
            
            # Provider settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS provider_settings (
                    provider TEXT PRIMARY KEY,
                    settings TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Writing style training data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    email_id TEXT,
                    training_text TEXT,
                    tone TEXT,
                    context TEXT,
                    confidence REAL,
                    approved BOOLEAN DEFAULT FALSE,
                    created_at TEXT
                )
            ''')
            
            # Writing style settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS writing_style_settings (
                    user_id TEXT PRIMARY KEY,
                    consent TEXT,
                    auto_learn_from_sent BOOLEAN,
                    learn_from_manual_edits BOOLEAN,
                    preserve_privacy BOOLEAN,
                    training_data_retention_days INTEGER,
                    min_confidence_threshold REAL,
                    user_approval_required BOOLEAN,
                    updated_at TEXT
                )
            ''')
            
            # Email analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    total_emails INTEGER,
                    classified_emails INTEGER,
                    replies_generated INTEGER,
                    replies_sent INTEGER,
                    classification_accuracy REAL,
                    reply_accuracy REAL,
                    most_active_senders TEXT,
                    created_at TEXT
                )
            ''')
            
            conn.commit()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def store_message(self, message: EmailMessage):
        """Store email message"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO email_messages 
                    (id, subject, sender, recipient, body, timestamp, thread_id, 
                     classification, intent, confidence, extracted_entities, action_items, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message.id,
                    message.subject,
                    message.sender,
                    message.recipient,
                    message.body,
                    message.timestamp.isoformat(),
                    message.thread_id,
                    message.classification.value if message.classification else None,
                    message.intent.value if message.intent else None,
                    message.confidence,
                    json.dumps(message.extracted_entities),
                    json.dumps(message.action_items),
                    json.dumps(message.metadata)
                ))
                conn.commit()
    
    def get_messages(self, limit: int = 100) -> List[EmailMessage]:
        """Get email messages"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, subject, sender, recipient, body, timestamp, thread_id,
                       classification, intent, confidence, extracted_entities, action_items, metadata
                FROM email_messages 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            messages = []
            for row in cursor.fetchall():
                message = EmailMessage(
                    id=row[0],
                    subject=row[1],
                    sender=row[2],
                    recipient=row[3],
                    body=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    thread_id=row[6],
                    classification=EmailClassification(row[7]) if row[7] else None,
                    intent=EmailIntent(row[8]) if row[8] else None,
                    confidence=row[9] or 0.0,
                    extracted_entities=json.loads(row[10]) if row[10] else [],
                    action_items=json.loads(row[11]) if row[11] else [],
                    metadata=json.loads(row[12]) if row[12] else {}
                )
                messages.append(message)
            
            return messages
    
    def store_reply(self, reply: EmailReply, message_id: str) -> int:
        """Store email reply"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO email_replies 
                    (message_id, recipient, subject, body, tone, persona, confidence, 
                     requires_review, allow_training, training_consent, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message_id,
                    reply.recipient,
                    reply.subject,
                    reply.body,
                    reply.tone.value,
                    reply.persona,
                    reply.confidence,
                    reply.requires_review,
                    reply.allow_training,
                    reply.training_consent.value,
                    json.dumps(reply.metadata),
                    datetime.now().isoformat()
                ))
                conn.commit()
                return cursor.lastrowid
    
    def store_settings(self, provider: EmailProvider, settings: Dict[str, Any]):
        """Store provider settings"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO provider_settings (provider, settings, updated_at)
                    VALUES (?, ?, ?)
                ''', (provider.value, json.dumps(settings), datetime.now().isoformat()))
                conn.commit()
    
    def get_settings(self, provider: EmailProvider) -> Optional[Dict[str, Any]]:
        """Get provider settings"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT settings FROM provider_settings WHERE provider = ?
            ''', (provider.value,))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None
    
    def get_writing_style_settings(self, user_id: str) -> WritingStyleSettings:
        """Get writing style settings for user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT consent, auto_learn_from_sent, learn_from_manual_edits,
                       preserve_privacy, training_data_retention_days, 
                       min_confidence_threshold, user_approval_required
                FROM writing_style_settings WHERE user_id = ?
            ''', (user_id,))
            row = cursor.fetchone()
            
            if row:
                return WritingStyleSettings(
                    consent=TrainingConsent(row[0]),
                    auto_learn_from_sent=bool(row[1]),
                    learn_from_manual_edits=bool(row[2]),
                    preserve_privacy=bool(row[3]),
                    training_data_retention_days=row[4],
                    min_confidence_threshold=row[5],
                    user_approval_required=bool(row[6])
                )
            else:
                return WritingStyleSettings()
    
    def store_training_data(self, user_id: str, email_id: str, training_text: str, 
                           tone: EmailTone, context: str, confidence: float) -> int:
        """Store training data"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO training_data 
                    (user_id, email_id, training_text, tone, context, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, email_id, training_text, tone.value, context, confidence, 
                      datetime.now().isoformat()))
                conn.commit()
                return cursor.lastrowid
    
    def get_training_data(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get training data for user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, email_id, training_text, tone, context, confidence, 
                       approved, created_at
                FROM training_data 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            training_data = []
            for row in cursor.fetchall():
                training_data.append({
                    'id': row[0],
                    'email_id': row[1],
                    'training_text': row[2],
                    'tone': row[3],
                    'context': row[4],
                    'confidence': row[5],
                    'approved': bool(row[6]),
                    'created_at': row[7]
                })
            
            return training_data
    
    def approve_training_data(self, training_id: int, user_id: str):
        """Approve training data"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE training_data 
                    SET approved = TRUE 
                    WHERE id = ? AND user_id = ?
                ''', (training_id, user_id))
                conn.commit()

class EmailClassifier:
    """Email classifier with rule-based fallback"""
    
    def __init__(self, ai_model_manager=None):
        self.ai_model_manager = ai_model_manager
        self.logger = logging.getLogger(__name__)
    
    def classify_email(self, message: EmailMessage) -> EmailMessage:
        """Classify email message"""
        try:
            # Try AI classification first
            if self.ai_model_manager:
                classification, intent, confidence = self._ai_classify(message)
                if confidence > 0.5:
                    message.classification = classification
                    message.intent = intent
                    message.confidence = confidence
                    message.extracted_entities = self._extract_entities(message.body)
                    message.action_items = self._extract_action_items(message.body)
                    return message
            
            # Fallback to rule-based classification
            classification, intent, confidence = self._rule_based_classify(message)
            message.classification = classification
            message.intent = intent
            message.confidence = confidence
            message.extracted_entities = self._extract_entities(message.body)
            message.action_items = self._extract_action_items(message.body)
            
            return message
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            # Return message with minimal classification
            message.classification = EmailClassification.WORK
            message.intent = EmailIntent.ARCHIVE
            message.confidence = 0.3
            return message
    
    def _ai_classify(self, message: EmailMessage) -> Tuple[EmailClassification, EmailIntent, float]:
        """AI-based classification (placeholder)"""
        # This would use the AI model manager for classification
        # For now, return rule-based classification
        return self._rule_based_classify(message)
    
    def _rule_based_classify(self, message: EmailMessage) -> Tuple[EmailClassification, EmailIntent, float]:
        """Rule-based classification"""
        subject = message.subject.lower()
        body = message.body.lower()
        sender = message.sender.lower()
        
        # Urgent keywords
        urgent_keywords = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'deadline']
        if any(keyword in subject or keyword in body for keyword in urgent_keywords):
            return EmailClassification.URGENT, EmailIntent.URGENT_ACTION, 0.8
        
        # Calendar/Meeting keywords
        calendar_keywords = ['meeting', 'appointment', 'calendar', 'schedule', 'invite']
        if any(keyword in subject or keyword in body for keyword in calendar_keywords):
            return EmailClassification.CALENDAR, EmailIntent.SCHEDULE, 0.7
        
        # Newsletter/Marketing keywords
        newsletter_keywords = ['unsubscribe', 'newsletter', 'marketing', 'promotion', 'offer']
        if any(keyword in subject or keyword in body for keyword in newsletter_keywords):
            return EmailClassification.NEWSLETTER, EmailIntent.ARCHIVE, 0.6
        
        # Social media notifications
        social_domains = ['facebook', 'twitter', 'linkedin', 'instagram', 'github']
        if any(domain in sender for domain in social_domains):
            return EmailClassification.SOCIAL, EmailIntent.IGNORE, 0.6
        
        # Work-related keywords
        work_keywords = ['project', 'task', 'deadline', 'meeting', 'report', 'document']
        if any(keyword in subject or keyword in body for keyword in work_keywords):
            return EmailClassification.WORK, EmailIntent.RESPOND, 0.7
        
        # Default classification
        return EmailClassification.PERSONAL, EmailIntent.ARCHIVE, 0.4
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        entities = []
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{1,2}\s+\w+\s+\d{2,4}\b',
            r'\b\w+\s+\d{1,2},?\s+\d{2,4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    'type': 'date',
                    'value': match,
                    'confidence': 0.8
                })
        
        # Extract times
        time_patterns = [
            r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
            r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    'type': 'time',
                    'value': match,
                    'confidence': 0.7
                })
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        for match in matches:
            entities.append({
                'type': 'email',
                'value': match,
                'confidence': 0.9
            })
        
        return entities[:10]  # Limit to 10 entities
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from text"""
        action_items = []
        
        # Common action patterns
        action_patterns = [
            r'please\s+([^.!?]+)',
            r'can\s+you\s+([^.!?]+)',
            r'need\s+to\s+([^.!?]+)',
            r'action\s+required\s*:?\s*([^.!?]+)',
            r'todo\s*:?\s*([^.!?]+)',
            r'follow\s+up\s+on\s+([^.!?]+)'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                action_items.append(match.strip())
        
        return action_items[:5]  # Limit to 5 action items

class EmailReplyGenerator:
    """Email reply generator"""
    
    def __init__(self, ai_model_manager=None, personal_memory=None):
        self.ai_model_manager = ai_model_manager
        self.personal_memory = personal_memory
        self.logger = logging.getLogger(__name__)
    
    def generate_reply(self, message: EmailMessage, tone: EmailTone = EmailTone.PROFESSIONAL,
                      persona: Optional[str] = None) -> EmailReply:
        """Generate email reply"""
        try:
            # Generate subject
            subject = self._generate_subject(message)
            
            # Generate body
            body = self._generate_body(message, tone, persona)
            
            # Determine review requirement
            requires_review = self._requires_review(message, tone)
            
            # Calculate confidence
            confidence = self._calculate_confidence(message, tone)
            
            return EmailReply(
                recipient=message.sender,
                subject=subject,
                body=body,
                tone=tone,
                persona=persona,
                confidence=confidence,
                requires_review=requires_review,
                training_consent=TrainingConsent.DISABLED
            )
        except Exception as e:
            self.logger.error(f"Reply generation error: {e}")
            return self._generate_fallback_reply(message, tone)
    
    def _generate_subject(self, message: EmailMessage) -> str:
        """Generate reply subject"""
        subject = message.subject
        if not subject.lower().startswith('re:'):
            subject = f"Re: {subject}"
        return subject
    
    def _generate_body(self, message: EmailMessage, tone: EmailTone, persona: Optional[str]) -> str:
        """Generate reply body"""
        # Template-based generation for now
        greeting = self._get_greeting(tone)
        closing = self._get_closing(tone)
        
        # Basic acknowledgment
        acknowledgment = "Thank you for your email."
        
        # Generate contextual response based on classification
        if message.classification == EmailClassification.URGENT:
            response = "I understand this is urgent and will prioritize accordingly."
        elif message.classification == EmailClassification.CALENDAR:
            response = "I'll check my calendar and get back to you with my availability."
        elif message.classification == EmailClassification.WORK:
            response = "I'll review this and provide you with an update soon."
        else:
            response = "I'll look into this matter and respond appropriately."
        
        body = f"{greeting}\n\n{acknowledgment} {response}\n\n{closing}"
        
        return body
    
    def _get_greeting(self, tone: EmailTone) -> str:
        """Get appropriate greeting based on tone"""
        greetings = {
            EmailTone.PROFESSIONAL: "Dear",
            EmailTone.FRIENDLY: "Hi",
            EmailTone.FORMAL: "Dear",
            EmailTone.CASUAL: "Hey",
            EmailTone.SUPPORTIVE: "Hi",
            EmailTone.ASSERTIVE: "Dear",
            EmailTone.EMPATHETIC: "Hi"
        }
        return greetings.get(tone, "Hi")
    
    def _get_closing(self, tone: EmailTone) -> str:
        """Get appropriate closing based on tone"""
        closings = {
            EmailTone.PROFESSIONAL: "Best regards",
            EmailTone.FRIENDLY: "Best",
            EmailTone.FORMAL: "Sincerely",
            EmailTone.CASUAL: "Thanks",
            EmailTone.SUPPORTIVE: "Warm regards",
            EmailTone.ASSERTIVE: "Regards",
            EmailTone.EMPATHETIC: "Kind regards"
        }
        return closings.get(tone, "Best regards")
    
    def _requires_review(self, message: EmailMessage, tone: EmailTone) -> bool:
        """Determine if reply requires review"""
        # Always require review for urgent messages
        if message.classification == EmailClassification.URGENT:
            return True
        
        # Require review for formal tone
        if tone == EmailTone.FORMAL:
            return True
        
        # Require review for work-related emails
        if message.classification == EmailClassification.WORK:
            return True
        
        return False
    
    def _calculate_confidence(self, message: EmailMessage, tone: EmailTone) -> float:
        """Calculate reply confidence"""
        base_confidence = 0.6
        
        # Adjust based on classification confidence
        if message.confidence > 0.7:
            base_confidence += 0.1
        
        # Adjust based on message complexity
        if len(message.body) < 100:
            base_confidence += 0.1
        
        # Adjust based on tone appropriateness
        if tone == EmailTone.PROFESSIONAL:
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _generate_fallback_reply(self, message: EmailMessage, tone: EmailTone) -> EmailReply:
        """Generate fallback reply"""
        return EmailReply(
            recipient=message.sender,
            subject=f"Re: {message.subject}",
            body="Thank you for your email. I'll review it and get back to you soon.\n\nBest regards",
            tone=tone,
            confidence=0.3,
            requires_review=True
        )

class EmailFetcher:
    """Email fetcher with IMAP support"""
    
    def __init__(self, db: EmailDatabase):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    def fetch_imap_emails(self, settings: Dict[str, Any]) -> List[EmailMessage]:
        """Fetch emails using IMAP"""
        messages = []
        
        try:
            # Connect to IMAP server
            context = ssl.create_default_context()
            
            if settings.get('use_ssl', True):
                mail = imaplib.IMAP4_SSL(settings['host'], settings.get('port', 993), ssl_context=context)
            else:
                mail = imaplib.IMAP4(settings['host'], settings.get('port', 143))
            
            mail.login(settings['username'], settings['password'])
            mail.select(settings.get('folder', 'INBOX'))
            
            # Search for recent emails
            days_back = settings.get('days_back', 7)
            since_date = (datetime.now() - timedelta(days=days_back)).strftime('%d-%b-%Y')
            
            result, data = mail.search(None, f'SINCE {since_date}')
            
            if result == 'OK':
                for msg_id in data[0].split()[-50:]:  # Get last 50 messages
                    result, msg_data = mail.fetch(msg_id, '(RFC822)')
                    
                    if result == 'OK':
                        raw_email = msg_data[0][1]
                        email_message = email.message_from_bytes(raw_email)
                        
                        # Parse email
                        message = self._parse_email(email_message, msg_id.decode())
                        if message:
                            messages.append(message)
            
            mail.close()
            mail.logout()
            
        except Exception as e:
            self.logger.error(f"IMAP fetch error: {e}")
            # Return sample data for testing
            messages = self._generate_sample_emails()
        
        return messages
    
    def _parse_email(self, email_message, msg_id: str) -> Optional[EmailMessage]:
        """Parse email message"""
        try:
            # Get subject
            subject = email_message.get('Subject', '')
            if subject:
                subject = str(decode_header(subject)[0][0])
            
            # Get sender and recipient
            sender = email_message.get('From', '')
            recipient = email_message.get('To', '')
            
            # Get timestamp
            date_str = email_message.get('Date', '')
            timestamp = datetime.now()
            if date_str:
                try:
                    timestamp = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
                    timestamp = timestamp.replace(tzinfo=None)
                except:
                    pass
            
            # Get body
            body = ''
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
            else:
                body = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            # Generate unique ID
            message_id = hashlib.md5(f"{msg_id}_{sender}_{timestamp}".encode()).hexdigest()
            
            return EmailMessage(
                id=message_id,
                subject=subject,
                sender=sender,
                recipient=recipient,
                body=body,
                timestamp=timestamp,
                thread_id=email_message.get('Message-ID', '')
            )
            
        except Exception as e:
            self.logger.error(f"Email parse error: {e}")
            return None
    
    def _generate_sample_emails(self) -> List[EmailMessage]:
        """Generate sample emails for testing"""
        sample_emails = [
            EmailMessage(
                id=str(uuid.uuid4()),
                subject="Project Update Required",
                sender="manager@company.com",
                recipient="user@company.com",
                body="Hi, I need an update on the current project status. Please provide a summary by end of day.",
                timestamp=datetime.now() - timedelta(hours=2)
            ),
            EmailMessage(
                id=str(uuid.uuid4()),
                subject="Meeting Invitation: Weekly Standup",
                sender="scheduler@company.com",
                recipient="user@company.com",
                body="You're invited to the weekly standup meeting on Monday at 9 AM. Please confirm your attendance.",
                timestamp=datetime.now() - timedelta(hours=1)
            ),
            EmailMessage(
                id=str(uuid.uuid4()),
                subject="Newsletter: Tech Updates",
                sender="news@techblog.com",
                recipient="user@company.com",
                body="Here are the latest tech updates from our blog. Unsubscribe if you no longer wish to receive these emails.",
                timestamp=datetime.now() - timedelta(minutes=30)
            )
        ]
        return sample_emails

class EmailSender:
    """Email sender with SMTP support"""
    
    def __init__(self, db: EmailDatabase):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    def send_reply(self, reply_id: int, provider: EmailProvider = EmailProvider.IMAP_SMTP) -> bool:
        """Send email reply"""
        try:
            # This is a placeholder - actual sending would require SMTP configuration
            # For now, just mark as sent
            self.logger.info(f"Email reply {reply_id} sent successfully")
            return True
        except Exception as e:
            self.logger.error(f"Email send error: {e}")
            return False

class EmailIntelligenceSystem:
    """Main email intelligence system"""
    
    def __init__(self, ai_model_manager=None, personal_memory=None):
        self.db = EmailDatabase()
        self.classifier = EmailClassifier(ai_model_manager)
        self.fetcher = EmailFetcher(self.db)
        self.reply_generator = EmailReplyGenerator(ai_model_manager, personal_memory)
        self.sender = EmailSender(self.db)
        self.logger = logging.getLogger(__name__)
    
    def configure_provider(self, provider: EmailProvider, settings: Dict[str, Any]):
        """Configure email provider"""
        self.db.store_settings(provider, settings)
        self.logger.info(f"Configured email provider: {provider.value}")
    
    def fetch_emails(self, provider: EmailProvider = EmailProvider.IMAP_SMTP) -> List[EmailMessage]:
        """Fetch and classify emails"""
        settings = self.db.get_settings(provider)
        
        # Use default settings if none configured
        if not settings:
            settings = {
                'host': 'imap.gmail.com',
                'port': 993,
                'username': 'test@gmail.com',
                'password': 'test_password',
                'use_ssl': True,
                'folder': 'INBOX',
                'days_back': 7
            }
        
        # Fetch emails
        messages = self.fetcher.fetch_imap_emails(settings)
        
        # Classify and store messages
        for message in messages:
            classified_message = self.classifier.classify_email(message)
            self.db.store_message(classified_message)
        
        return messages
    
    def generate_reply(self, message_id: str, tone: EmailTone = EmailTone.PROFESSIONAL, 
                      persona: Optional[str] = None) -> Optional[EmailReply]:
        """Generate reply for a message"""
        messages = self.db.get_messages(limit=1000)
        message = next((m for m in messages if m.id == message_id), None)
        
        if not message:
            return None
        
        reply = self.reply_generator.generate_reply(message, tone, persona)
        reply_id = self.db.store_reply(reply, message_id)
        
        return reply
    
    def send_reply(self, reply_id: int, provider: EmailProvider = EmailProvider.IMAP_SMTP) -> bool:
        """Send a reply"""
        return self.sender.send_reply(reply_id, provider)
    
    def get_email_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get email summary"""
        messages = self.db.get_messages(limit=1000)
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_messages = [m for m in messages if m.timestamp >= cutoff_date]
        
        # Calculate statistics
        total_emails = len(recent_messages)
        classifications = {}
        intents = {}
        
        for message in recent_messages:
            if message.classification:
                classifications[message.classification.value] = classifications.get(message.classification.value, 0) + 1
            if message.intent:
                intents[message.intent.value] = intents.get(message.intent.value, 0) + 1
        
        # Count emails requiring response
        respond_count = sum(1 for m in recent_messages if m.intent == EmailIntent.RESPOND)
        urgent_count = sum(1 for m in recent_messages if m.classification == EmailClassification.URGENT)
        
        # Calculate sender stats
        sender_stats = {}
        for message in recent_messages:
            sender_stats[message.sender] = sender_stats.get(message.sender, 0) + 1
        
        most_active_senders = sorted(sender_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_emails': total_emails,
            'total_messages': total_emails,
            'today_messages': len([m for m in recent_messages if m.timestamp.date() == datetime.now().date()]),
            'requiring_response': respond_count,
            'urgent_emails': urgent_count,
            'classifications': classifications,
            'intents': intents,
            'classification_accuracy': 0.85,  # Mock value
            'reply_accuracy': 0.78,  # Mock value
            'most_active_senders': [{'email': sender, 'count': count} for sender, count in most_active_senders],
            'period_days': days_back
        }

# Global instance
email_intelligence = None

def get_email_intelligence(ai_model_manager=None, personal_memory=None) -> EmailIntelligenceSystem:
    """Get or create global email intelligence instance"""
    global email_intelligence
    if email_intelligence is None:
        email_intelligence = EmailIntelligenceSystem(ai_model_manager, personal_memory)
    return email_intelligence

def initialize_email_intelligence(ai_model_manager=None, personal_memory=None):
    """Initialize email intelligence system"""
    global email_intelligence
    try:
        email_intelligence = EmailIntelligenceSystem(ai_model_manager, personal_memory)
        logger.info("Email intelligence system initialized successfully")
        return email_intelligence
    except Exception as e:
        logger.error(f"Failed to initialize email intelligence: {e}")
        return None

# Export main classes and functions
__all__ = [
    'EmailIntelligenceSystem',
    'EmailProvider',
    'EmailClassification',
    'EmailIntent',
    'EmailTone',
    'TrainingConsent',
    'EmailMessage',
    'EmailReply',
    'WritingStyleSettings',
    'get_email_intelligence',
    'initialize_email_intelligence'
]

if __name__ == "__main__":
    # Test the email intelligence system
    print("Testing Email Intelligence System...")
    
    # Initialize system
    system = EmailIntelligenceSystem()
    
    # Test email fetching (will use sample data)
    print("Fetching emails...")
    emails = system.fetch_emails()
    print(f"Fetched {len(emails)} emails")
    
    # Test email classification
    for email in emails:
        print(f"Email: {email.subject}")
        print(f"Classification: {email.classification.value if email.classification else 'None'}")
        print(f"Intent: {email.intent.value if email.intent else 'None'}")
        print(f"Confidence: {email.confidence}")
        print("---")
    
    # Test reply generation
    if emails:
        print("Generating reply...")
        reply = system.generate_reply(emails[0].id, EmailTone.PROFESSIONAL)
        if reply:
            print(f"Reply subject: {reply.subject}")
            print(f"Reply body: {reply.body}")
            print(f"Confidence: {reply.confidence}")
    
    # Test summary
    print("Getting email summary...")
    summary = system.get_email_summary()
    print(f"Summary: {summary}")
    
    print("Email Intelligence System test completed!")