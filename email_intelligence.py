"""
Email Intelligence System for Personal AI
Provides email ingestion, classification, and intelligent reply generation
"""

import os
import json
import logging
import smtplib
import imaplib
import email
import sqlite3
import threading
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
import ssl

# Try to import optional dependencies
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GMAIL_API_AVAILABLE = True
except ImportError:
    GMAIL_API_AVAILABLE = False

try:
    import requests
    from msal import ConfidentialClientApplication
    GRAPH_API_AVAILABLE = True
except ImportError:
    GRAPH_API_AVAILABLE = False


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
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmailDatabase:
    """Email database manager"""
    
    def __init__(self, db_path: str = "email_intelligence.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize email database"""
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
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Email replies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_replies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_message_id TEXT,
                    recipient TEXT,
                    subject TEXT,
                    body TEXT,
                    tone TEXT,
                    persona TEXT,
                    confidence REAL,
                    requires_review INTEGER,
                    sent INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (original_message_id) REFERENCES email_messages (id)
                )
            ''')
            
            # Email settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    settings TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def get_connection(self):
        """Get database connection with proper locking"""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def store_message(self, message: EmailMessage):
        """Store an email message"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO email_messages 
                    (id, subject, sender, recipient, body, timestamp, thread_id, 
                     classification, intent, confidence, extracted_entities, 
                     action_items, metadata)
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
    
    def get_messages(self, limit: int = 50) -> List[EmailMessage]:
        """Get recent email messages"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM email_messages 
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
    
    def store_reply(self, reply: EmailReply, original_message_id: str):
        """Store an email reply draft"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO email_replies 
                    (original_message_id, recipient, subject, body, tone, persona, 
                     confidence, requires_review)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    original_message_id,
                    reply.recipient,
                    reply.subject,
                    reply.body,
                    reply.tone.value,
                    reply.persona,
                    reply.confidence,
                    1 if reply.requires_review else 0
                ))
                conn.commit()
                return cursor.lastrowid
    
    def get_settings(self, provider: EmailProvider) -> Optional[Dict[str, Any]]:
        """Get email settings for a provider"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT settings FROM email_settings 
                WHERE provider = ? AND is_active = 1
                ORDER BY created_at DESC
                LIMIT 1
            ''', (provider.value,))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
    
    def store_settings(self, provider: EmailProvider, settings: Dict[str, Any]):
        """Store email settings"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO email_settings (provider, settings)
                    VALUES (?, ?)
                ''', (provider.value, json.dumps(settings)))
                conn.commit()


class EmailClassifier:
    """Email classification and analysis engine"""
    
    def __init__(self, ai_model_manager=None):
        self.ai_model_manager = ai_model_manager
        self.logger = logging.getLogger(__name__)
        
        # Classification keywords
        self.classification_keywords = {
            EmailClassification.URGENT: [
                'urgent', 'asap', 'immediately', 'emergency', 'critical',
                'deadline', 'due today', 'time sensitive', 'rush'
            ],
            EmailClassification.WORK: [
                'meeting', 'project', 'deadline', 'report', 'team',
                'office', 'manager', 'colleague', 'client', 'proposal'
            ],
            EmailClassification.PERSONAL: [
                'family', 'friend', 'personal', 'home', 'vacation',
                'birthday', 'anniversary', 'dinner', 'weekend'
            ],
            EmailClassification.NEWSLETTER: [
                'newsletter', 'unsubscribe', 'weekly', 'monthly',
                'digest', 'update', 'news', 'bulletin'
            ],
            EmailClassification.SOCIAL: [
                'facebook', 'twitter', 'linkedin', 'instagram',
                'social', 'follow', 'like', 'share', 'comment'
            ],
            EmailClassification.CALENDAR: [
                'calendar', 'meeting', 'appointment', 'schedule',
                'event', 'invite', 'invitation', 'rsvp'
            ],
            EmailClassification.REMINDER: [
                'reminder', 'remember', 'don\'t forget', 'due',
                'expiring', 'renewal', 'payment', 'bill'
            ]
        }
        
        # Intent keywords
        self.intent_keywords = {
            EmailIntent.RESPOND: [
                'question', 'reply', 'response', 'feedback',
                'thoughts', 'opinion', 'can you', 'please'
            ],
            EmailIntent.URGENT_ACTION: [
                'urgent', 'asap', 'immediately', 'critical',
                'emergency', 'action required', 'please confirm'
            ],
            EmailIntent.SCHEDULE: [
                'schedule', 'meeting', 'appointment', 'calendar',
                'available', 'free time', 'book', 'reserve'
            ]
        }
    
    def classify_email(self, message: EmailMessage) -> EmailMessage:
        """Classify an email message"""
        text = f"{message.subject} {message.body}".lower()
        
        # Find best classification
        best_classification = EmailClassification.NOTIFICATION
        best_score = 0.0
        
        for classification, keywords in self.classification_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_classification = classification
        
        # Find best intent
        best_intent = EmailIntent.ARCHIVE
        best_intent_score = 0.0
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_intent_score:
                best_intent_score = score
                best_intent = intent
        
        # Extract entities and action items
        entities = self._extract_entities(text)
        action_items = self._extract_action_items(text)
        
        # Update message
        message.classification = best_classification
        message.intent = best_intent
        message.confidence = min(1.0, (best_score + best_intent_score) / 10.0)
        message.extracted_entities = entities
        message.action_items = action_items
        
        return message
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from email text"""
        entities = []
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}\/\d{1,2}\/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'date',
                    'value': match,
                    'confidence': 0.8
                })
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        for match in email_matches:
            entities.append({
                'type': 'email',
                'value': match,
                'confidence': 0.9
            })
        
        # Extract phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phone_matches = re.findall(phone_pattern, text)
        for match in phone_matches:
            entities.append({
                'type': 'phone',
                'value': match,
                'confidence': 0.7
            })
        
        return entities
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from email text"""
        action_items = []
        
        # Action keywords
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


class EmailFetcher:
    """Email fetching from various providers"""
    
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
                        email_message = email.message_from_bytes(msg_data[0][1])
                        
                        # Parse email
                        subject = self._decode_header(email_message.get('Subject', ''))
                        sender = self._decode_header(email_message.get('From', ''))
                        recipient = self._decode_header(email_message.get('To', ''))
                        
                        # Get body
                        body = self._get_email_body(email_message)
                        
                        # Get timestamp
                        timestamp = email.utils.parsedate_to_datetime(email_message.get('Date'))
                        
                        message = EmailMessage(
                            id=f"imap_{msg_id.decode()}",
                            subject=subject,
                            sender=sender,
                            recipient=recipient,
                            body=body,
                            timestamp=timestamp,
                            metadata={'provider': 'imap', 'message_id': msg_id.decode()}
                        )
                        
                        messages.append(message)
            
            mail.close()
            mail.logout()
            
        except Exception as e:
            self.logger.error(f"Error fetching IMAP emails: {e}")
        
        return messages
    
    def _decode_header(self, header: str) -> str:
        """Decode email header"""
        if not header:
            return ""
        
        decoded = decode_header(header)
        return ''.join([
            part.decode(encoding or 'utf-8') if isinstance(part, bytes) else part
            for part, encoding in decoded
        ])
    
    def _get_email_body(self, email_message) -> str:
        """Extract email body"""
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            body = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        return body


class EmailReplyGenerator:
    """Generate intelligent email replies"""
    
    def __init__(self, ai_model_manager=None, personal_memory=None):
        self.ai_model_manager = ai_model_manager
        self.personal_memory = personal_memory
        self.logger = logging.getLogger(__name__)
    
    def generate_reply(self, message: EmailMessage, tone: EmailTone = EmailTone.PROFESSIONAL, 
                      persona: Optional[str] = None) -> EmailReply:
        """Generate a reply to an email message"""
        
        # Get context from personal memory
        context = self._get_personal_context(message, persona)
        
        # Generate reply using AI
        reply_text = self._generate_reply_text(message, tone, context)
        
        # Generate subject
        subject = self._generate_subject(message.subject)
        
        reply = EmailReply(
            recipient=message.sender,
            subject=subject,
            body=reply_text,
            tone=tone,
            persona=persona,
            confidence=0.8,
            requires_review=True
        )
        
        return reply
    
    def _get_personal_context(self, message: EmailMessage, persona: Optional[str]) -> Dict[str, Any]:
        """Get personal context for reply generation"""
        context = {
            'sender': message.sender,
            'subject': message.subject,
            'classification': message.classification.value if message.classification else None,
            'intent': message.intent.value if message.intent else None,
            'action_items': message.action_items,
            'entities': message.extracted_entities
        }
        
        if self.personal_memory:
            # Get memories related to sender
            memories = self.personal_memory.search_memories(message.sender)
            context['memories'] = memories[:3]  # Top 3 relevant memories
            
            # Get preferences
            preferences = self.personal_memory.get_preferences('communication')
            context['preferences'] = preferences
        
        return context
    
    def _generate_reply_text(self, message: EmailMessage, tone: EmailTone, context: Dict[str, Any]) -> str:
        """Generate reply text using AI"""
        
        # Tone-based templates
        tone_templates = {
            EmailTone.PROFESSIONAL: "Thank you for your email. ",
            EmailTone.FRIENDLY: "Hi! Thanks for reaching out. ",
            EmailTone.FORMAL: "Dear {sender}, I acknowledge receipt of your message. ",
            EmailTone.CASUAL: "Hey! Got your message. ",
            EmailTone.SUPPORTIVE: "I understand your concern and I'm here to help. ",
            EmailTone.ASSERTIVE: "I've reviewed your request. ",
            EmailTone.EMPATHETIC: "I can see this is important to you. "
        }
        
        # Start with tone template
        reply_start = tone_templates.get(tone, "Thank you for your email. ")
        
        # Generate contextual response
        if message.classification == EmailClassification.URGENT:
            reply_text = reply_start + "I understand this is urgent and I'll prioritize this accordingly."
        elif message.intent == EmailIntent.RESPOND:
            reply_text = reply_start + "I'll review this and get back to you with a detailed response."
        elif message.intent == EmailIntent.SCHEDULE:
            reply_text = reply_start + "I'll check my calendar and propose some available times."
        else:
            reply_text = reply_start + "I've received your message and will follow up as needed."
        
        # Add action items if present
        if message.action_items:
            reply_text += f"\n\nRegarding the items you mentioned, I'll address: {', '.join(message.action_items[:2])}"
        
        # Add closing based on tone
        if tone == EmailTone.PROFESSIONAL:
            reply_text += "\n\nBest regards"
        elif tone == EmailTone.FRIENDLY:
            reply_text += "\n\nThanks again!"
        elif tone == EmailTone.FORMAL:
            reply_text += "\n\nSincerely"
        else:
            reply_text += "\n\nThanks!"
        
        return reply_text
    
    def _generate_subject(self, original_subject: str) -> str:
        """Generate reply subject"""
        if original_subject.lower().startswith('re:'):
            return original_subject
        else:
            return f"Re: {original_subject}"


class EmailSender:
    """Email sending functionality"""
    
    def __init__(self, db: EmailDatabase):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    def send_smtp_email(self, reply: EmailReply, settings: Dict[str, Any]) -> bool:
        """Send email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = settings['username']
            msg['To'] = reply.recipient
            msg['Subject'] = reply.subject
            
            # Add body
            msg.attach(MIMEText(reply.body, 'plain'))
            
            # Connect to SMTP server
            if settings.get('use_ssl', True):
                server = smtplib.SMTP_SSL(settings['host'], settings.get('port', 587))
            else:
                server = smtplib.SMTP(settings['host'], settings.get('port', 587))
                server.starttls()
            
            server.login(settings['username'], settings['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email sent successfully to {reply.recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
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
        if not settings:
            raise ValueError(f"No settings found for provider: {provider.value}")
        
        # Fetch emails
        if provider == EmailProvider.IMAP_SMTP:
            messages = self.fetcher.fetch_imap_emails(settings)
        else:
            raise NotImplementedError(f"Provider {provider.value} not implemented yet")
        
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
        # Implementation would fetch reply from database and send
        # This is a placeholder for the actual implementation
        return True
    
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
        
        return {
            'total_emails': total_emails,
            'requiring_response': respond_count,
            'urgent_emails': urgent_count,
            'classifications': classifications,
            'intents': intents,
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