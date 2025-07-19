#!/usr/bin/env python3
"""
Office 365 Email Intelligence Integration
Connects Office 365 OAuth2 authentication with the email intelligence system
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
from flask import Flask, request, jsonify, session, redirect, url_for

# Import your existing email intelligence system
from email_intelligence import (
    EmailIntelligenceSystem, 
    EmailMessage, 
    EmailProvider,
    EmailClassification,
    EmailIntent,
    EmailTone
)

# Import Office 365 authentication system
from office365_auth_system import (
    Office365AuthManager,
    Office365GraphClient,
    Office365Config,
    create_office365_auth
)

logger = logging.getLogger(__name__)

class Office365EmailProvider:
    """Office 365 email provider for the email intelligence system"""
    
    def __init__(self, graph_client: Office365GraphClient, auth_manager: Office365AuthManager):
        self.graph_client = graph_client
        self.auth_manager = auth_manager
        self.logger = logging.getLogger(__name__)
    
    def fetch_emails(self, user_id: str, limit: int = 50) -> List[EmailMessage]:
        """Fetch emails from Office 365 using Microsoft Graph API"""
        try:
            # Get messages from Graph API
            messages_data = self.graph_client.get_messages(user_id, limit)
            
            # Convert to EmailMessage objects
            email_messages = []
            for msg_data in messages_data:
                email_message = self._convert_graph_message_to_email_message(msg_data)
                if email_message:
                    email_messages.append(email_message)
            
            self.logger.info(f"Fetched {len(email_messages)} emails from Office 365")
            return email_messages
            
        except Exception as e:
            self.logger.error(f"Error fetching emails from Office 365: {e}")
            return []
    
    def _convert_graph_message_to_email_message(self, msg_data: Dict[str, Any]) -> Optional[EmailMessage]:
        """Convert Microsoft Graph message to EmailMessage"""
        try:
            # Extract sender email
            sender_email = ""
            if msg_data.get('sender') and msg_data['sender'].get('emailAddress'):
                sender_email = msg_data['sender']['emailAddress'].get('address', '')
            
            # Extract recipient email
            recipient_email = ""
            if msg_data.get('toRecipients') and len(msg_data['toRecipients']) > 0:
                recipient_email = msg_data['toRecipients'][0]['emailAddress'].get('address', '')
            
            # Extract body content
            body_content = ""
            if msg_data.get('body'):
                body_content = msg_data['body'].get('content', '')
                # Remove HTML tags for plain text processing
                import re
                body_content = re.sub(r'<[^>]+>', '', body_content)
            
            # Parse timestamp
            timestamp = datetime.now()
            if msg_data.get('receivedDateTime'):
                timestamp = datetime.fromisoformat(msg_data['receivedDateTime'].replace('Z', '+00:00'))
                timestamp = timestamp.replace(tzinfo=None)  # Remove timezone for consistency
            
            # Create EmailMessage
            email_message = EmailMessage(
                id=msg_data.get('id', ''),
                subject=msg_data.get('subject', ''),
                sender=sender_email,
                recipient=recipient_email,
                body=body_content,
                timestamp=timestamp,
                thread_id=msg_data.get('conversationId', ''),
                metadata={
                    'importance': msg_data.get('importance', 'normal'),
                    'is_read': msg_data.get('isRead', False),
                    'has_attachments': msg_data.get('hasAttachments', False),
                    'internet_message_id': msg_data.get('internetMessageId', ''),
                    'web_link': msg_data.get('webLink', '')
                }
            )
            
            return email_message
            
        except Exception as e:
            self.logger.error(f"Error converting Graph message to EmailMessage: {e}")
            return None
    
    def send_email(self, user_id: str, to_email: str, subject: str, body: str) -> bool:
        """Send email using Office 365"""
        try:
            return self.graph_client.send_message(user_id, to_email, subject, body)
        except Exception as e:
            self.logger.error(f"Error sending email via Office 365: {e}")
            return False
    
    def get_user_profile(self, user_id: str):
        """Get user profile from Office 365"""
        try:
            return self.graph_client.get_user_profile(user_id)
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            return None

class Office365EmailIntelligence:
    """Enhanced email intelligence system with Office 365 integration"""
    
    def __init__(self, email_intelligence: EmailIntelligenceSystem, 
                 office365_provider: Office365EmailProvider):
        self.email_intelligence = email_intelligence
        self.office365_provider = office365_provider
        self.logger = logging.getLogger(__name__)
    
    def fetch_and_classify_emails(self, user_id: str, limit: int = 50) -> List[EmailMessage]:
        """Fetch emails from Office 365 and classify them"""
        try:
            # Fetch emails from Office 365
            emails = self.office365_provider.fetch_emails(user_id, limit)
            
            # Classify each email
            classified_emails = []
            for email in emails:
                classified_email = self.email_intelligence.classifier.classify_email(email)
                self.email_intelligence.db.store_message(classified_email)
                classified_emails.append(classified_email)
            
            self.logger.info(f"Fetched and classified {len(classified_emails)} emails")
            return classified_emails
            
        except Exception as e:
            self.logger.error(f"Error fetching and classifying emails: {e}")
            return []
    
    def generate_and_send_reply(self, user_id: str, message_id: str, 
                               tone: EmailTone = EmailTone.PROFESSIONAL) -> bool:
        """Generate reply using AI and send via Office 365"""
        try:
            # Generate reply using email intelligence
            reply = self.email_intelligence.generate_reply(message_id, tone)
            if not reply:
                self.logger.error(f"Failed to generate reply for message {message_id}")
                return False
            
            # Send reply via Office 365
            success = self.office365_provider.send_email(
                user_id, 
                reply.recipient, 
                reply.subject, 
                reply.body
            )
            
            if success:
                self.logger.info(f"Successfully sent AI-generated reply for message {message_id}")
                return True
            else:
                self.logger.error(f"Failed to send reply for message {message_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error generating and sending reply: {e}")
            return False
    
    def get_email_analytics(self, user_id: str, days_back: int = 7) -> Dict[str, Any]:
        """Get email analytics enhanced with Office 365 data"""
        try:
            # Get base analytics from email intelligence
            analytics = self.email_intelligence.get_email_summary(days_back)
            
            # Add Office 365 specific metrics
            user_profile = self.office365_provider.get_user_profile(user_id)
            if user_profile:
                analytics['office365_user'] = {
                    'email': user_profile.email,
                    'display_name': user_profile.display_name,
                    'job_title': user_profile.job_title,
                    'office_location': user_profile.office_location
                }
            
            # Add provider information
            analytics['provider'] = 'Office 365'
            analytics['provider_features'] = {
                'real_time_sync': True,
                'calendar_integration': True,
                'contacts_integration': True,
                'advanced_search': True
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting email analytics: {e}")
            return {}

def create_office365_email_integration(app: Flask, email_intelligence: EmailIntelligenceSystem):
    """Create Office 365 email intelligence integration"""
    
    # Office 365 configuration from environment variables
    CLIENT_ID = os.getenv('OFFICE365_CLIENT_ID', '')
    CLIENT_SECRET = os.getenv('OFFICE365_CLIENT_SECRET', '')
    REDIRECT_URI = os.getenv('OFFICE365_REDIRECT_URI', 'http://localhost:5000/auth/office365/callback')
    TENANT_ID = os.getenv('OFFICE365_TENANT_ID', None)
    
    # Create Office 365 authentication system
    office365_auth, auth_manager = create_office365_auth(
        app, CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, TENANT_ID
    )
    
    # Create Office 365 email provider
    graph_client = Office365GraphClient(auth_manager)
    office365_provider = Office365EmailProvider(graph_client, auth_manager)
    
    # Create integrated email intelligence system
    office365_email_intelligence = Office365EmailIntelligence(email_intelligence, office365_provider)
    
    # Register additional routes for Office 365 email intelligence
    @app.route('/api/office365/email/fetch', methods=['POST'])
    def fetch_office365_emails():
        """Fetch emails from Office 365 and classify them"""
        if not session.get('office365_authenticated'):
            return jsonify({'error': 'Not authenticated with Office 365'}), 401
        
        user_id = session.get('office365_user_id')
        if not user_id:
            return jsonify({'error': 'User ID not found in session'}), 400
        
        data = request.get_json() or {}
        limit = data.get('limit', 50)
        
        try:
            emails = office365_email_intelligence.fetch_and_classify_emails(user_id, limit)
            
            # Convert to JSON serializable format
            emails_dict = []
            for email in emails:
                emails_dict.append({
                    'id': email.id,
                    'subject': email.subject,
                    'sender': email.sender,
                    'recipient': email.recipient,
                    'body': email.body[:500] + '...' if len(email.body) > 500 else email.body,
                    'timestamp': email.timestamp.isoformat(),
                    'thread_id': email.thread_id,
                    'classification': email.classification.value if email.classification else None,
                    'intent': email.intent.value if email.intent else None,
                    'confidence': email.confidence,
                    'extracted_entities': email.extracted_entities,
                    'action_items': email.action_items,
                    'metadata': email.metadata
                })
            
            return jsonify({
                'status': 'success',
                'provider': 'Office 365',
                'messages': emails_dict,
                'count': len(emails_dict),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error fetching Office 365 emails: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/office365/email/reply', methods=['POST'])
    def generate_office365_reply():
        """Generate and send reply via Office 365"""
        if not session.get('office365_authenticated'):
            return jsonify({'error': 'Not authenticated with Office 365'}), 401
        
        user_id = session.get('office365_user_id')
        if not user_id:
            return jsonify({'error': 'User ID not found in session'}), 400
        
        data = request.get_json()
        if not data or 'message_id' not in data:
            return jsonify({'error': 'Message ID is required'}), 400
        
        message_id = data['message_id']
        tone = data.get('tone', 'professional')
        auto_send = data.get('auto_send', False)
        
        try:
            # Validate tone
            if tone not in [t.value for t in EmailTone]:
                return jsonify({'error': f'Invalid tone: {tone}'}), 400
            
            email_tone = EmailTone(tone)
            
            if auto_send:
                # Generate and send reply automatically
                success = office365_email_intelligence.generate_and_send_reply(
                    user_id, message_id, email_tone
                )
                
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': 'Reply generated and sent successfully',
                        'message_id': message_id,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'Failed to generate and send reply'}), 500
            else:
                # Just generate reply for review
                reply = email_intelligence.generate_reply(message_id, email_tone)
                
                if not reply:
                    return jsonify({'error': 'Failed to generate reply'}), 500
                
                return jsonify({
                    'status': 'success',
                    'reply': {
                        'recipient': reply.recipient,
                        'subject': reply.subject,
                        'body': reply.body,
                        'tone': reply.tone.value,
                        'confidence': reply.confidence,
                        'requires_review': reply.requires_review
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error generating Office 365 reply: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/office365/email/send', methods=['POST'])
    def send_office365_email():
        """Send email via Office 365"""
        if not session.get('office365_authenticated'):
            return jsonify({'error': 'Not authenticated with Office 365'}), 401
        
        user_id = session.get('office365_user_id')
        if not user_id:
            return jsonify({'error': 'User ID not found in session'}), 400
        
        data = request.get_json()
        if not data or not all(k in data for k in ['to', 'subject', 'body']):
            return jsonify({'error': 'Missing required fields: to, subject, body'}), 400
        
        try:
            success = office365_provider.send_email(
                user_id,
                data['to'],
                data['subject'],
                data['body']
            )
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': 'Email sent successfully via Office 365',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Failed to send email'}), 500
                
        except Exception as e:
            logger.error(f"Error sending Office 365 email: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/office365/email/analytics', methods=['GET'])
    def get_office365_email_analytics():
        """Get email analytics with Office 365 data"""
        if not session.get('office365_authenticated'):
            return jsonify({'error': 'Not authenticated with Office 365'}), 401
        
        user_id = session.get('office365_user_id')
        if not user_id:
            return jsonify({'error': 'User ID not found in session'}), 400
        
        days_back = request.args.get('days_back', 7, type=int)
        
        try:
            analytics = office365_email_intelligence.get_email_analytics(user_id, days_back)
            
            return jsonify({
                'status': 'success',
                'analytics': analytics,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting Office 365 email analytics: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/office365/email/configure', methods=['POST'])
    def configure_office365_email():
        """Configure Office 365 email settings"""
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Configuration data required'}), 400
        
        try:
            # Store Office 365 specific settings
            settings = {
                'provider': 'office365',
                'auto_sync': data.get('auto_sync', True),
                'sync_interval': data.get('sync_interval', 300),  # 5 minutes
                'enable_calendar_integration': data.get('enable_calendar_integration', True),
                'enable_contacts_integration': data.get('enable_contacts_integration', True),
                'classification_threshold': data.get('classification_threshold', 0.7),
                'auto_reply_threshold': data.get('auto_reply_threshold', 0.8)
            }
            
            # Store in email intelligence database
            email_intelligence.db.store_settings(EmailProvider.OUTLOOK_GRAPH, settings)
            
            return jsonify({
                'status': 'success',
                'message': 'Office 365 email configuration saved',
                'settings': settings,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error configuring Office 365 email: {e}")
            return jsonify({'error': str(e)}), 500
    
    return office365_email_intelligence, office365_auth, auth_manager

# Flask app factory with Office 365 integration
def create_app_with_office365():
    """Create Flask app with Office 365 email intelligence integration"""
    app = Flask(__name__)
    app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
    
    # Initialize email intelligence system
    email_intelligence = EmailIntelligenceSystem()
    
    # Create Office 365 integration
    office365_integration, office365_auth, auth_manager = create_office365_email_integration(
        app, email_intelligence
    )
    
    # Enhanced email intelligence page with Office 365 support
    @app.route('/email-intelligence')
    def email_intelligence_page():
        """Enhanced email intelligence page with Office 365 integration"""
        return render_template('email_intelligence_office365.html')
    
    # Main dashboard showing authentication status
    @app.route('/')
    def index():
        """Main dashboard"""
        office365_status = session.get('office365_authenticated', False)
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Email Intelligence with Office 365</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 50px; }}
                .status {{ padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .connected {{ background: #d4edda; color: #155724; }}
                .disconnected {{ background: #f8d7da; color: #721c24; }}
                .btn {{ padding: 10px 20px; margin: 10px; text-decoration: none; 
                        color: white; background: #007bff; border-radius: 3px; }}
                .btn:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <h1>Email Intelligence with Office 365</h1>
            
            <div class="status {'connected' if office365_status else 'disconnected'}">
                <h3>Office 365 Status: {'Connected' if office365_status else 'Not Connected'}</h3>
                {'<p>Your Office 365 account is connected and ready for intelligent email processing.</p>' if office365_status else '<p>Connect your Office 365 account to enable intelligent email processing.</p>'}
            </div>
            
            <div>
                <h3>Available Actions:</h3>
                {f'<a href="/email-intelligence" class="btn">Open Email Intelligence Dashboard</a>' if office365_status else ''}
                {f'<a href="/auth/office365/logout" class="btn">Disconnect Office 365</a>' if office365_status else '<a href="/auth/office365/login" class="btn">Connect Office 365</a>'}
                <a href="/auth/office365/config" class="btn">Configure Office 365</a>
            </div>
            
            <div>
                <h3>API Endpoints:</h3>
                <ul>
                    <li><code>POST /api/office365/email/fetch</code> - Fetch and classify emails</li>
                    <li><code>POST /api/office365/email/reply</code> - Generate AI replies</li>
                    <li><code>POST /api/office365/email/send</code> - Send emails</li>
                    <li><code>GET /api/office365/email/analytics</code> - Get email analytics</li>
                    <li><code>GET /api/office365/status</code> - Check authentication status</li>
                </ul>
            </div>
        </body>
        </html>
        '''
    
    return app, office365_integration

# Example usage
if __name__ == "__main__":
    # Set up environment variables
    os.environ.setdefault('OFFICE365_CLIENT_ID', 'your-client-id')
    os.environ.setdefault('OFFICE365_CLIENT_SECRET', 'your-client-secret')
    os.environ.setdefault('OFFICE365_REDIRECT_URI', 'http://localhost:5000/auth/office365/callback')
    os.environ.setdefault('SECRET_KEY', 'your-secret-key-change-this')
    
    # Create app with Office 365 integration
    app, office365_integration = create_app_with_office365()
    
    print("Office 365 Email Intelligence Integration")
    print("=======================================")
    print("1. Configure Office 365: http://localhost:5000/auth/office365/config")
    print("2. Connect Office 365: http://localhost:5000/auth/office365/login")
    print("3. Email Intelligence: http://localhost:5000/email-intelligence")
    print("")
    print("Required Environment Variables:")
    print("- OFFICE365_CLIENT_ID")
    print("- OFFICE365_CLIENT_SECRET")
    print("- OFFICE365_REDIRECT_URI")
    print("- SECRET_KEY")
    print("")
    print("Azure AD App Registration Requirements:")
    print("- Application Type: Web")
    print("- Redirect URI: http://localhost:5000/auth/office365/callback")
    print("- API Permissions: Mail.ReadWrite, Mail.Send, User.Read, offline_access")
    
    app.run(debug=True, host='0.0.0.0', port=5000)