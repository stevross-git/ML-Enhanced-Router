#!/usr/bin/env python3
"""
Office 365 OAuth2 Authentication System
Provides secure authentication and token management for Office 365 integration
"""

import os
import json
import logging
import secrets
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import threading
from urllib.parse import urlencode, parse_qs, urlparse
import requests
from flask import Flask, request, redirect, url_for, session, jsonify, render_template_string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Office 365 Configuration
class Office365Config:
    """Office 365 OAuth2 configuration"""
    
    # Microsoft Graph API endpoints
    AUTHORITY = "https://login.microsoftonline.com/common"
    GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
    
    # OAuth2 endpoints
    AUTHORIZATION_URL = f"{AUTHORITY}/oauth2/v2.0/authorize"
    TOKEN_URL = f"{AUTHORITY}/oauth2/v2.0/token"
    
    # Required scopes for email access
    EMAIL_SCOPES = [
        "https://graph.microsoft.com/Mail.ReadWrite",
        "https://graph.microsoft.com/Mail.Send",
        "https://graph.microsoft.com/User.Read",
        "https://graph.microsoft.com/MailboxSettings.ReadWrite",
        "offline_access"  # For refresh tokens
    ]
    
    # Optional scopes for enhanced features
    ENHANCED_SCOPES = [
        "https://graph.microsoft.com/Calendars.ReadWrite",
        "https://graph.microsoft.com/Contacts.ReadWrite",
        "https://graph.microsoft.com/Files.ReadWrite.All"
    ]

@dataclass
class Office365Token:
    """Office 365 OAuth2 token data"""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    scope: str
    id_token: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired"""
        expiry_time = self.created_at + timedelta(seconds=self.expires_in - 300)  # 5 min buffer
        return datetime.now() > expiry_time
    
    @property
    def expires_at(self) -> datetime:
        """Get token expiration time"""
        return self.created_at + timedelta(seconds=self.expires_in)

@dataclass
class Office365User:
    """Office 365 user information"""
    user_id: str
    email: str
    display_name: str
    given_name: Optional[str] = None
    surname: Optional[str] = None
    job_title: Optional[str] = None
    office_location: Optional[str] = None
    preferred_language: Optional[str] = None
    user_principal_name: Optional[str] = None

class Office365AuthState(Enum):
    """OAuth2 authentication states"""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ERROR = "error"

class Office365Database:
    """Database for Office 365 authentication data"""
    
    def __init__(self, db_path: str = "office365_auth.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize authentication database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # OAuth2 tokens table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS oauth2_tokens (
                    user_id TEXT PRIMARY KEY,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    token_type TEXT NOT NULL,
                    expires_in INTEGER NOT NULL,
                    scope TEXT NOT NULL,
                    id_token TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    given_name TEXT,
                    surname TEXT,
                    job_title TEXT,
                    office_location TEXT,
                    preferred_language TEXT,
                    user_principal_name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # OAuth2 state table (for CSRF protection)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS oauth2_states (
                    state TEXT PRIMARY KEY,
                    user_session_id TEXT,
                    code_verifier TEXT,
                    scopes TEXT,
                    redirect_uri TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            ''')
            
            # App registrations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS app_registrations (
                    app_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    client_secret TEXT NOT NULL,
                    tenant_id TEXT,
                    redirect_uri TEXT NOT NULL,
                    scopes TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def store_token(self, user_id: str, token: Office365Token):
        """Store OAuth2 token"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO oauth2_tokens 
                    (user_id, access_token, refresh_token, token_type, expires_in, 
                     scope, id_token, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    token.access_token,
                    token.refresh_token,
                    token.token_type,
                    token.expires_in,
                    token.scope,
                    token.id_token,
                    token.created_at.isoformat(),
                    now
                ))
                conn.commit()
    
    def get_token(self, user_id: str) -> Optional[Office365Token]:
        """Get OAuth2 token for user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT access_token, refresh_token, token_type, expires_in, 
                       scope, id_token, created_at
                FROM oauth2_tokens 
                WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if row:
                return Office365Token(
                    access_token=row[0],
                    refresh_token=row[1],
                    token_type=row[2],
                    expires_in=row[3],
                    scope=row[4],
                    id_token=row[5],
                    created_at=datetime.fromisoformat(row[6])
                )
            return None
    
    def store_user_profile(self, user: Office365User):
        """Store user profile"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_profiles 
                    (user_id, email, display_name, given_name, surname, job_title,
                     office_location, preferred_language, user_principal_name, 
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user.user_id,
                    user.email,
                    user.display_name,
                    user.given_name,
                    user.surname,
                    user.job_title,
                    user.office_location,
                    user.preferred_language,
                    user.user_principal_name,
                    now,
                    now
                ))
                conn.commit()
    
    def get_user_profile(self, user_id: str) -> Optional[Office365User]:
        """Get user profile"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, email, display_name, given_name, surname, 
                       job_title, office_location, preferred_language, user_principal_name
                FROM user_profiles 
                WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if row:
                return Office365User(
                    user_id=row[0],
                    email=row[1],
                    display_name=row[2],
                    given_name=row[3],
                    surname=row[4],
                    job_title=row[5],
                    office_location=row[6],
                    preferred_language=row[7],
                    user_principal_name=row[8]
                )
            return None
    
    def store_oauth2_state(self, state: str, user_session_id: str, code_verifier: str,
                          scopes: List[str], redirect_uri: str):
        """Store OAuth2 state for CSRF protection"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now()
                expires_at = now + timedelta(minutes=10)  # State expires in 10 minutes
                
                cursor.execute('''
                    INSERT OR REPLACE INTO oauth2_states 
                    (state, user_session_id, code_verifier, scopes, redirect_uri, 
                     created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    state,
                    user_session_id,
                    code_verifier,
                    json.dumps(scopes),
                    redirect_uri,
                    now.isoformat(),
                    expires_at.isoformat()
                ))
                conn.commit()
    
    def get_oauth2_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Get OAuth2 state data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_session_id, code_verifier, scopes, redirect_uri, expires_at
                FROM oauth2_states 
                WHERE state = ?
            ''', (state,))
            
            row = cursor.fetchone()
            if row:
                expires_at = datetime.fromisoformat(row[4])
                if datetime.now() > expires_at:
                    # State expired, clean up
                    self.delete_oauth2_state(state)
                    return None
                
                return {
                    'user_session_id': row[0],
                    'code_verifier': row[1],
                    'scopes': json.loads(row[2]),
                    'redirect_uri': row[3],
                    'expires_at': expires_at
                }
            return None
    
    def delete_oauth2_state(self, state: str):
        """Delete OAuth2 state"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM oauth2_states WHERE state = ?', (state,))
                conn.commit()
    
    def store_app_registration(self, app_id: str, client_id: str, client_secret: str,
                              tenant_id: Optional[str], redirect_uri: str, scopes: List[str]):
        """Store app registration"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO app_registrations 
                    (app_id, client_id, client_secret, tenant_id, redirect_uri, 
                     scopes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    app_id,
                    client_id,
                    client_secret,
                    tenant_id,
                    redirect_uri,
                    json.dumps(scopes),
                    now,
                    now
                ))
                conn.commit()
    
    def get_app_registration(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get app registration"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT client_id, client_secret, tenant_id, redirect_uri, scopes
                FROM app_registrations 
                WHERE app_id = ?
            ''', (app_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'client_id': row[0],
                    'client_secret': row[1],
                    'tenant_id': row[2],
                    'redirect_uri': row[3],
                    'scopes': json.loads(row[4])
                }
            return None

class Office365AuthManager:
    """Office 365 OAuth2 authentication manager"""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, 
                 tenant_id: Optional[str] = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.tenant_id = tenant_id or "common"
        self.db = Office365Database()
        self.logger = logging.getLogger(__name__)
        
        # Store app registration
        self.db.store_app_registration(
            app_id="default",
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            redirect_uri=redirect_uri,
            scopes=Office365Config.EMAIL_SCOPES
        )
    
    def generate_pkce_challenge(self) -> Tuple[str, str]:
        """Generate PKCE code verifier and challenge"""
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        return code_verifier, code_challenge
    
    def get_authorization_url(self, user_session_id: str, scopes: Optional[List[str]] = None) -> str:
        """Generate authorization URL for OAuth2 flow"""
        scopes = scopes or Office365Config.EMAIL_SCOPES
        
        # Generate PKCE challenge
        code_verifier, code_challenge = self.generate_pkce_challenge()
        
        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        
        # Store state in database
        self.db.store_oauth2_state(
            state=state,
            user_session_id=user_session_id,
            code_verifier=code_verifier,
            scopes=scopes,
            redirect_uri=self.redirect_uri
        )
        
        # Build authorization URL
        auth_params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(scopes),
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256',
            'response_mode': 'query'
        }
        
        authority = Office365Config.AUTHORITY
        if self.tenant_id != "common":
            authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        
        auth_url = f"{authority}/oauth2/v2.0/authorize?" + urlencode(auth_params)
        
        self.logger.info(f"Generated authorization URL for user session: {user_session_id}")
        return auth_url
    
    def exchange_code_for_token(self, code: str, state: str) -> Optional[Office365Token]:
        """Exchange authorization code for access token"""
        # Validate state
        state_data = self.db.get_oauth2_state(state)
        if not state_data:
            self.logger.error(f"Invalid or expired state: {state}")
            return None
        
        # Clean up state
        self.db.delete_oauth2_state(state)
        
        # Exchange code for token
        token_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code',
            'code_verifier': state_data['code_verifier']
        }
        
        authority = Office365Config.AUTHORITY
        if self.tenant_id != "common":
            authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        
        try:
            response = requests.post(
                f"{authority}/oauth2/v2.0/token",
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                token_response = response.json()
                
                token = Office365Token(
                    access_token=token_response['access_token'],
                    refresh_token=token_response.get('refresh_token', ''),
                    token_type=token_response.get('token_type', 'Bearer'),
                    expires_in=token_response.get('expires_in', 3600),
                    scope=token_response.get('scope', ''),
                    id_token=token_response.get('id_token')
                )
                
                self.logger.info("Successfully exchanged code for token")
                return token
            else:
                self.logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Token exchange error: {e}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[Office365Token]:
        """Refresh access token using refresh token"""
        token_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        }
        
        authority = Office365Config.AUTHORITY
        if self.tenant_id != "common":
            authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        
        try:
            response = requests.post(
                f"{authority}/oauth2/v2.0/token",
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                token_response = response.json()
                
                token = Office365Token(
                    access_token=token_response['access_token'],
                    refresh_token=token_response.get('refresh_token', refresh_token),
                    token_type=token_response.get('token_type', 'Bearer'),
                    expires_in=token_response.get('expires_in', 3600),
                    scope=token_response.get('scope', ''),
                    id_token=token_response.get('id_token')
                )
                
                self.logger.info("Successfully refreshed token")
                return token
            else:
                self.logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Token refresh error: {e}")
            return None
    
    def get_valid_token(self, user_id: str) -> Optional[Office365Token]:
        """Get valid access token for user, refreshing if necessary"""
        token = self.db.get_token(user_id)
        if not token:
            return None
        
        if token.is_expired:
            self.logger.info(f"Token expired for user {user_id}, refreshing...")
            new_token = self.refresh_token(token.refresh_token)
            if new_token:
                self.db.store_token(user_id, new_token)
                return new_token
            else:
                self.logger.error(f"Failed to refresh token for user {user_id}")
                return None
        
        return token
    
    def revoke_token(self, user_id: str) -> bool:
        """Revoke user's token"""
        token = self.db.get_token(user_id)
        if not token:
            return False
        
        # Microsoft Graph doesn't have a standard revoke endpoint
        # We'll just delete the token from our database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM oauth2_tokens WHERE user_id = ?', (user_id,))
            conn.commit()
        
        self.logger.info(f"Revoked token for user {user_id}")
        return True

class Office365GraphClient:
    """Microsoft Graph API client"""
    
    def __init__(self, auth_manager: Office365AuthManager):
        self.auth_manager = auth_manager
        self.logger = logging.getLogger(__name__)
    
    def _get_headers(self, user_id: str) -> Optional[Dict[str, str]]:
        """Get authorization headers for API requests"""
        token = self.auth_manager.get_valid_token(user_id)
        if not token:
            return None
        
        return {
            'Authorization': f'{token.token_type} {token.access_token}',
            'Content-Type': 'application/json'
        }
    
    def get_user_profile(self, user_id: str) -> Optional[Office365User]:
        """Get user profile from Microsoft Graph"""
        headers = self._get_headers(user_id)
        if not headers:
            return None
        
        try:
            response = requests.get(
                f"{Office365Config.GRAPH_API_BASE}/me",
                headers=headers
            )
            
            if response.status_code == 200:
                user_data = response.json()
                
                user = Office365User(
                    user_id=user_data['id'],
                    email=user_data.get('mail') or user_data.get('userPrincipalName'),
                    display_name=user_data.get('displayName', ''),
                    given_name=user_data.get('givenName'),
                    surname=user_data.get('surname'),
                    job_title=user_data.get('jobTitle'),
                    office_location=user_data.get('officeLocation'),
                    preferred_language=user_data.get('preferredLanguage'),
                    user_principal_name=user_data.get('userPrincipalName')
                )
                
                # Store user profile
                self.auth_manager.db.store_user_profile(user)
                
                self.logger.info(f"Retrieved user profile for {user.email}")
                return user
            else:
                self.logger.error(f"Failed to get user profile: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            return None
    
    def get_messages(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's email messages"""
        headers = self._get_headers(user_id)
        if not headers:
            return []
        
        try:
            response = requests.get(
                f"{Office365Config.GRAPH_API_BASE}/me/messages",
                headers=headers,
                params={'$top': limit, '$orderby': 'receivedDateTime desc'}
            )
            
            if response.status_code == 200:
                messages_data = response.json()
                return messages_data.get('value', [])
            else:
                self.logger.error(f"Failed to get messages: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting messages: {e}")
            return []
    
    def send_message(self, user_id: str, to_email: str, subject: str, body: str) -> bool:
        """Send email message"""
        headers = self._get_headers(user_id)
        if not headers:
            return False
        
        message_data = {
            'message': {
                'subject': subject,
                'body': {
                    'contentType': 'HTML',
                    'content': body
                },
                'toRecipients': [
                    {
                        'emailAddress': {
                            'address': to_email
                        }
                    }
                ]
            }
        }
        
        try:
            response = requests.post(
                f"{Office365Config.GRAPH_API_BASE}/me/sendMail",
                headers=headers,
                json=message_data
            )
            
            if response.status_code == 202:
                self.logger.info(f"Message sent successfully to {to_email}")
                return True
            else:
                self.logger.error(f"Failed to send message: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False

class Office365AuthFlask:
    """Flask integration for Office 365 authentication"""
    
    def __init__(self, app: Flask, auth_manager: Office365AuthManager):
        self.app = app
        self.auth_manager = auth_manager
        self.graph_client = Office365GraphClient(auth_manager)
        self.register_routes()
    
    def register_routes(self):
        """Register Flask routes for Office 365 authentication"""
        
        @self.app.route('/auth/office365/login')
        def office365_login():
            """Initiate Office 365 OAuth2 login"""
            user_session_id = session.get('user_id', secrets.token_urlsafe(16))
            session['user_id'] = user_session_id
            
            # Get requested scopes
            scopes = request.args.getlist('scope')
            if not scopes:
                scopes = Office365Config.EMAIL_SCOPES
            
            auth_url = self.auth_manager.get_authorization_url(user_session_id, scopes)
            return redirect(auth_url)
        
        @self.app.route('/auth/office365/callback')
        def office365_callback():
            """Handle Office 365 OAuth2 callback"""
            code = request.args.get('code')
            state = request.args.get('state')
            error = request.args.get('error')
            
            if error:
                logger.error(f"OAuth2 error: {error}")
                return jsonify({'error': error}), 400
            
            if not code or not state:
                return jsonify({'error': 'Missing code or state parameter'}), 400
            
            # Exchange code for token
            token = self.auth_manager.exchange_code_for_token(code, state)
            if not token:
                return jsonify({'error': 'Failed to exchange code for token'}), 400
            
            # Get user profile
            user_profile = self.graph_client.get_user_profile(token.access_token)
            if not user_profile:
                return jsonify({'error': 'Failed to get user profile'}), 400
            
            # Store token with user ID
            self.auth_manager.db.store_token(user_profile.user_id, token)
            
            # Store user session
            session['office365_user_id'] = user_profile.user_id
            session['office365_authenticated'] = True
            
            return redirect(url_for('office365_success'))
        
        @self.app.route('/auth/office365/success')
        def office365_success():
            """Office 365 authentication success page"""
            if not session.get('office365_authenticated'):
                return redirect(url_for('office365_login'))
            
            user_id = session.get('office365_user_id')
            user_profile = self.auth_manager.db.get_user_profile(user_id)
            
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Office 365 Authentication Success</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 50px; }
                    .success { color: green; }
                    .info { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1 class="success">✓ Office 365 Authentication Successful</h1>
                <div class="info">
                    <h3>User Information:</h3>
                    <p><strong>Name:</strong> {{ user_profile.display_name }}</p>
                    <p><strong>Email:</strong> {{ user_profile.email }}</p>
                    <p><strong>Job Title:</strong> {{ user_profile.job_title or 'N/A' }}</p>
                    <p><strong>Office:</strong> {{ user_profile.office_location or 'N/A' }}</p>
                </div>
                <p><a href="/auth/office365/logout">Logout</a></p>
                <p><a href="/email-intelligence">Go to Email Intelligence</a></p>
            </body>
            </html>
            ''', user_profile=user_profile)
        
        @self.app.route('/auth/office365/logout')
        def office365_logout():
            """Office 365 logout"""
            user_id = session.get('office365_user_id')
            if user_id:
                self.auth_manager.revoke_token(user_id)
            
            session.pop('office365_user_id', None)
            session.pop('office365_authenticated', None)
            
            return jsonify({'message': 'Logged out successfully'})
        
        @self.app.route('/api/office365/status')
        def office365_status():
            """Get Office 365 authentication status"""
            if not session.get('office365_authenticated'):
                return jsonify({
                    'authenticated': False,
                    'user': None
                })
            
            user_id = session.get('office365_user_id')
            user_profile = self.auth_manager.db.get_user_profile(user_id)
            token = self.auth_manager.get_valid_token(user_id)
            
            return jsonify({
                'authenticated': True,
                'user': {
                    'id': user_profile.user_id,
                    'email': user_profile.email,
                    'display_name': user_profile.display_name,
                    'job_title': user_profile.job_title
                } if user_profile else None,
                'token_expires_at': token.expires_at.isoformat() if token else None
            })
        
        @self.app.route('/api/office365/messages')
        def office365_messages():
            """Get Office 365 messages"""
            if not session.get('office365_authenticated'):
                return jsonify({'error': 'Not authenticated'}), 401
            
            user_id = session.get('office365_user_id')
            limit = request.args.get('limit', 50, type=int)
            
            messages = self.graph_client.get_messages(user_id, limit)
            return jsonify({
                'messages': messages,
                'count': len(messages)
            })
        
        @self.app.route('/api/office365/send-message', methods=['POST'])
        def office365_send_message():
            """Send Office 365 message"""
            if not session.get('office365_authenticated'):
                return jsonify({'error': 'Not authenticated'}), 401
            
            user_id = session.get('office365_user_id')
            data = request.get_json()
            
            if not data or not all(k in data for k in ['to', 'subject', 'body']):
                return jsonify({'error': 'Missing required fields'}), 400
            
            success = self.graph_client.send_message(
                user_id, 
                data['to'], 
                data['subject'], 
                data['body']
            )
            
            if success:
                return jsonify({'message': 'Message sent successfully'})
            else:
                return jsonify({'error': 'Failed to send message'}), 500

# Factory function for easy integration
def create_office365_auth(app: Flask, client_id: str, client_secret: str, 
                         redirect_uri: str, tenant_id: Optional[str] = None):
    """Create Office 365 authentication system"""
    auth_manager = Office365AuthManager(client_id, client_secret, redirect_uri, tenant_id)
    office365_auth = Office365AuthFlask(app, auth_manager)
    
    # Add configuration endpoint
    @app.route('/auth/office365/config', methods=['GET', 'POST'])
    def office365_config():
        """Configure Office 365 authentication"""
        if request.method == 'GET':
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Office 365 Configuration</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 50px; }
                    .form-group { margin: 15px 0; }
                    label { display: block; margin-bottom: 5px; }
                    input, select { width: 100%; padding: 8px; margin-bottom: 10px; }
                    button { background: #0078d4; color: white; padding: 10px 20px; border: none; border-radius: 3px; }
                    .info { background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <h1>Office 365 Configuration</h1>
                <div class="info">
                    <h3>Setup Instructions:</h3>
                    <ol>
                        <li>Go to <a href="https://portal.azure.com/" target="_blank">Azure Portal</a></li>
                        <li>Navigate to "Azure Active Directory" > "App registrations"</li>
                        <li>Click "New registration"</li>
                        <li>Set redirect URI to: <code>{{ redirect_uri }}</code></li>
                        <li>Copy the Client ID and create a Client Secret</li>
                        <li>Add API permissions for Microsoft Graph (Mail.ReadWrite, Mail.Send, User.Read)</li>
                    </ol>
                </div>
                <form method="post">
                    <div class="form-group">
                        <label>Client ID:</label>
                        <input type="text" name="client_id" required>
                    </div>
                    <div class="form-group">
                        <label>Client Secret:</label>
                        <input type="password" name="client_secret" required>
                    </div>
                    <div class="form-group">
                        <label>Tenant ID (optional):</label>
                        <input type="text" name="tenant_id" placeholder="Leave empty for multi-tenant">
                    </div>
                    <button type="submit">Save Configuration</button>
                </form>
            </body>
            </html>
            ''', redirect_uri=redirect_uri)
        
        elif request.method == 'POST':
            client_id = request.form.get('client_id')
            client_secret = request.form.get('client_secret')
            tenant_id = request.form.get('tenant_id') or None
            
            if not client_id or not client_secret:
                return jsonify({'error': 'Client ID and Client Secret are required'}), 400
            
            # Update configuration
            auth_manager.client_id = client_id
            auth_manager.client_secret = client_secret
            auth_manager.tenant_id = tenant_id or "common"
            
            # Store in database
            auth_manager.db.store_app_registration(
                app_id="default",
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id,
                redirect_uri=redirect_uri,
                scopes=Office365Config.EMAIL_SCOPES
            )
            
            return jsonify({'message': 'Configuration saved successfully'})
    
    return office365_auth, auth_manager

# Example usage
if __name__ == "__main__":
    from flask import Flask
    
    app = Flask(__name__)
    app.secret_key = 'your-secret-key-here'
    
    # Configure Office 365 authentication
    CLIENT_ID = "your-client-id"
    CLIENT_SECRET = "your-client-secret"
    REDIRECT_URI = "http://localhost:5000/auth/office365/callback"
    TENANT_ID = None  # Use "common" for multi-tenant
    
    # Create authentication system
    office365_auth, auth_manager = create_office365_auth(
        app, CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, TENANT_ID
    )
    
    @app.route('/')
    def index():
        return '''
        <h1>Office 365 Authentication Demo</h1>
        <p><a href="/auth/office365/config">Configure Office 365</a></p>
        <p><a href="/auth/office365/login">Login with Office 365</a></p>
        <p><a href="/api/office365/status">Check Status</a></p>
        '''
    
    print("Office 365 Authentication System")
    print("================================")
    print("1. Go to http://localhost:5000/auth/office365/config to configure")
    print("2. Set up your Azure AD app registration")
    print("3. Test authentication at http://localhost:5000/auth/office365/login")
    print("")
    print("Required Azure AD App Permissions:")
    print("- Mail.ReadWrite")
    print("- Mail.Send") 
    print("- User.Read")
    print("- offline_access")
    
    app.run(debug=True, port=5000)