// Real-time collaboration JavaScript for templates/collaborate.html
// Save this as static/js/collaborate_realtime.js

class RealtimeCollaboration {
    constructor() {
        this.eventSource = null;
        this.currentSessionId = null;
        this.messageContainer = null;
        this.statusIndicator = null;
        this.isConnected = false;
        this.messageQueue = [];
        this.connectionAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.initializeUI();
    }
    
    initializeUI() {
        // Find the real-time collaboration panel
        this.findCollaborationPanel();
        
        if (!this.messageContainer) {
            console.warn('Real-time collaboration panel not found, creating one');
            this.createCollaborationPanel();
        }
        
        // Add status indicator
        this.addStatusIndicator();
        
        // Initial state
        this.showPlaceholder();
    }
    
    findCollaborationPanel() {
        // Try different selectors to find the collaboration panel
        const selectors = [
            '.Real-time\\ Collaboration',
            '[data-panel="realtime"]',
            '#realtime-collaboration',
            '.card:has(h5:contains("Real-time Collaboration"))',
            '.col-md-6:last-child .card'  // Assuming it's in the right column
        ];
        
        for (const selector of selectors) {
            try {
                const panel = document.querySelector(selector);
                if (panel) {
                    this.messageContainer = panel.querySelector('.message-container') || 
                                           this.createMessageContainer(panel);
                    return;
                }
            } catch (e) {
                // Continue to next selector
            }
        }
        
        // If not found, try to find by text content
        const cards = document.querySelectorAll('.card');
        for (const card of cards) {
            const header = card.querySelector('h5, h4, .card-header');
            if (header && header.textContent.includes('Real-time Collaboration')) {
                this.messageContainer = this.createMessageContainer(card);
                return;
            }
        }
    }
    
    createCollaborationPanel() {
        // Create the entire collaboration panel if it doesn't exist
        const rightColumn = document.querySelector('.col-md-6:last-child') || 
                           document.querySelector('.row .col-md-6:nth-child(2)');
        
        if (rightColumn) {
            const panelHtml = `
                <div class="card mt-4" id="realtime-collaboration">
                    <div class="card-header">
                        <h5 class="mb-0">
                            <i class="fas fa-comments me-2"></i>Real-time Collaboration
                        </h5>
                    </div>
                    <div class="card-body">
                        <div id="realtime-messages" class="message-container"></div>
                    </div>
                </div>
            `;
            
            rightColumn.insertAdjacentHTML('beforeend', panelHtml);
            this.messageContainer = document.getElementById('realtime-messages');
        }
    }
    
    createMessageContainer(parentElement) {
        // Create message container inside existing panel
        let container = parentElement.querySelector('.message-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'message-container';
            container.id = 'realtime-messages';
            
            // Find the card body or create one
            let cardBody = parentElement.querySelector('.card-body');
            if (!cardBody) {
                cardBody = document.createElement('div');
                cardBody.className = 'card-body';
                parentElement.appendChild(cardBody);
            }
            
            // Replace existing content or append
            cardBody.innerHTML = '';
            cardBody.appendChild(container);
        }
        
        // Style the container
        container.style.cssText = `
            height: 400px;
            overflow-y: auto;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
            margin-top: 10px;
        `;
        
        return container;
    }
    
    addStatusIndicator() {
        if (!this.messageContainer || !this.messageContainer.parentNode) return;
        
        // Add connection status indicator
        const statusDiv = document.createElement('div');
        statusDiv.id = 'realtime-status';
        statusDiv.style.cssText = `
            padding: 8px 12px;
            margin-bottom: 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            text-align: center;
        `;
        
        this.messageContainer.parentNode.insertBefore(statusDiv, this.messageContainer);
        this.statusIndicator = statusDiv;
        this.updateStatus('disconnected', 'Not connected');
    }
    
    updateStatus(state, message) {
        if (!this.statusIndicator) return;
        
        const states = {
            'connected': { color: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', icon: '🟢' },
            'connecting': { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', icon: '🟡' },
            'disconnected': { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', icon: '🔴' },
            'error': { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', icon: '❌' }
        };
        
        const style = states[state] || states['disconnected'];
        this.statusIndicator.style.color = style.color;
        this.statusIndicator.style.backgroundColor = style.bg;
        this.statusIndicator.innerHTML = `${style.icon} ${message}`;
        
        this.isConnected = (state === 'connected');
    }
    
    showPlaceholder() {
        if (!this.messageContainer) return;
        
        this.messageContainer.innerHTML = `
            <div style="text-align: center; color: #6b7280; padding: 40px 20px;">
                <div style="font-size: 24px; margin-bottom: 10px;">💬</div>
                <div style="font-size: 16px; margin-bottom: 8px;">Real-time Collaboration</div>
                <div style="font-size: 14px;">Start a collaboration to see live AI interactions...</div>
            </div>
        `;
    }
    
    connectToSession(sessionId) {
        if (this.currentSessionId === sessionId && this.isConnected) {
            console.log('Already connected to session:', sessionId);
            return;
        }
        
        this.disconnect();
        this.currentSessionId = sessionId;
        this.connectionAttempts = 0;
        this.connect();
    }
    
    connect() {
        if (!this.currentSessionId) {
            console.warn('No session ID provided for real-time connection');
            return;
        }
        
        this.updateStatus('connecting', 'Connecting to session...');
        
        try {
            // Close existing connection
            if (this.eventSource) {
                this.eventSource.close();
            }
            
            // Create new EventSource connection
            const url = `/api/shared-memory/sessions/${this.currentSessionId}/stream`;
            this.eventSource = new EventSource(url);
            
            this.eventSource.onopen = () => {
                console.log('Real-time connection opened');
                this.updateStatus('connected', `Connected to session ${this.currentSessionId.substring(0, 8)}...`);
                this.connectionAttempts = 0;
                this.clearMessages();
            };
            
            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Error parsing SSE message:', error);
                }
            };
            
            this.eventSource.onerror = (error) => {
                console.error('SSE connection error:', error);
                this.handleConnectionError();
            };
            
        } catch (error) {
            console.error('Error creating SSE connection:', error);
            this.updateStatus('error', 'Connection failed');
            this.attemptReconnect();
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'initial':
                this.handleInitialData(data.context);
                break;
            case 'message':
                this.displayMessage(data.message);
                break;
            case 'heartbeat':
                // Connection is alive
                break;
            case 'error':
                console.error('Server error:', data.error);
                this.updateStatus('error', `Server error: ${data.error}`);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    handleInitialData(context) {
        console.log('Received initial session context:', context);
        this.clearMessages();
        
        // Display session info
        this.addSessionHeader(context);
        
        // Display existing messages if any
        if (context.latest_thoughts && context.latest_thoughts.length > 0) {
            context.latest_thoughts.forEach(thought => {
                this.displayThought(thought);
            });
        }
    }
    
    addSessionHeader(context) {
        if (!this.messageContainer) return;
        
        const headerHtml = `
            <div class="session-header" style="
                padding: 12px;
                margin-bottom: 15px;
                background: rgba(59, 130, 246, 0.1);
                border-radius: 6px;
                border-left: 3px solid #3b82f6;
            ">
                <div style="font-weight: 600; color: #3b82f6; margin-bottom: 5px;">
                    Session: ${context.session_id ? context.session_id.substring(0, 8) + '...' : 'Unknown'}
                </div>
                <div style="font-size: 14px; color: #d1d5db;">
                    Participants: ${context.participants ? context.participants.join(', ') : 'None'}
                </div>
                <div style="font-size: 12px; color: #9ca3af; margin-top: 3px;">
                    Messages: ${context.message_count || 0}
                </div>
            </div>
        `;
        
        this.messageContainer.innerHTML = headerHtml;
    }
    
    displayMessage(message) {
        if (!this.messageContainer) return;
        
        const messageEl = this.createMessageElement(message);
        this.messageContainer.appendChild(messageEl);
        this.scrollToBottom();
    }
    
    displayThought(thought) {
        if (!this.messageContainer) return;
        
        const thoughtEl = this.createThoughtElement(thought);
        this.messageContainer.appendChild(thoughtEl);
        this.scrollToBottom();
    }
    
    createMessageElement(message) {
        const div = document.createElement('div');
        div.className = 'realtime-message';
        
        const typeColors = {
            'query': '#3b82f6',
            'response': '#10b981',
            'thought': '#f59e0b',
            'external_llm_response': '#8b5cf6',
            'collaboration': '#06b6d4',
            'fact': '#84cc16',
            'question': '#f97316',
            'conclusion': '#ef4444'
        };
        
        const color = typeColors[message.message_type] || '#6b7280';
        const timestamp = new Date(message.timestamp).toLocaleTimeString();
        
        div.style.cssText = `
            margin-bottom: 12px;
            padding: 10px 12px;
            background: rgba(255,255,255,0.03);
            border-radius: 6px;
            border-left: 3px solid ${color};
            animation: slideIn 0.3s ease-out;
        `;
        
        div.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-weight: 600; color: ${color}; font-size: 13px;">
                    ${message.agent_name} • ${message.message_type.replace('_', ' ').toUpperCase()}
                </div>
                <div style="font-size: 11px; color: #9ca3af;">${timestamp}</div>
            </div>
            <div style="color: #f3f4f6; font-size: 14px; line-height: 1.4;">
                ${this.formatContent(message.content)}
            </div>
        `;
        
        return div;
    }
    
    createThoughtElement(thought) {
        const div = document.createElement('div');
        div.className = 'realtime-thought';
        
        const timestamp = new Date(thought.timestamp).toLocaleTimeString();
        
        div.style.cssText = `
            margin-bottom: 10px;
            padding: 8px 10px;
            background: rgba(245, 158, 11, 0.1);
            border-radius: 4px;
            border-left: 2px solid #f59e0b;
        `;
        
        div.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <div style="font-weight: 600; color: #f59e0b; font-size: 12px;">
                    ${thought.agent} • THINKING
                </div>
                <div style="font-size: 10px; color: #9ca3af;">${timestamp}</div>
            </div>
            <div style="color: #fbbf24; font-size: 13px; font-style: italic;">
                ${this.formatContent(thought.content)}
            </div>
        `;
        
        return div;
    }
    
    formatContent(content) {
        // Basic content formatting
        if (content.length > 200) {
            return content.substring(0, 200) + '...';
        }
        return content.replace(/\n/g, '<br>');
    }
    
    clearMessages() {
        if (this.messageContainer) {
            this.messageContainer.innerHTML = '';
        }
    }
    
    scrollToBottom() {
        if (this.messageContainer) {
            this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
        }
    }
    
    handleConnectionError() {
        this.updateStatus('error', 'Connection lost');
        this.attemptReconnect();
    }
    
    attemptReconnect() {
        if (this.connectionAttempts >= this.maxReconnectAttempts) {
            this.updateStatus('error', 'Reconnection failed');
            return;
        }
        
        this.connectionAttempts++;
        const delay = Math.pow(2, this.connectionAttempts) * 1000; // Exponential backoff
        
        this.updateStatus('connecting', `Reconnecting... (${this.connectionAttempts}/${this.maxReconnectAttempts})`);
        
        setTimeout(() => {
            if (this.currentSessionId) {
                this.connect();
            }
        }, delay);
    }
    
    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        this.currentSessionId = null;
        this.isConnected = false;
        this.updateStatus('disconnected', 'Disconnected');
        this.showPlaceholder();
    }
}

// Global instance
let realtimeCollaboration = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    realtimeCollaboration = new RealtimeCollaboration();
    
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .message-container::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
        }
        
        .message-container::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
        }
        
        .message-container::-webkit-scrollbar-thumb:hover {
            background: rgba(255,255,255,0.3);
        }
    `;
    document.head.appendChild(style);
});

// Hook into the collaboration form submission
function hookIntoCollaborationForm() {
    const collaborateForm = document.getElementById('collaborate-form') || 
                           document.querySelector('form[data-form="collaborate"]') ||
                           document.querySelector('form');
    
    if (collaborateForm) {
        // Store original onsubmit
        const originalOnSubmit = collaborateForm.onsubmit;
        
        // Override the collaborate function
        if (window.collaborate && typeof window.collaborate === 'function') {
            const originalCollaborate = window.collaborate;
            
            window.collaborate = function() {
                // Generate session ID
                const sessionId = generateSessionId();
                
                // Start real-time connection immediately
                if (realtimeCollaboration) {
                    realtimeCollaboration.connectToSession(sessionId);
                }
                
                // Call original function (but we need to modify it to include session_id)
                return originalCollaborate.call(this, sessionId);
            };
        }
    }
}

function generateSessionId() {
    // Generate a UUID-like session ID
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Hook into form when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(hookIntoCollaborationForm, 500);
});

// Export for global access
window.realtimeCollaboration = realtimeCollaboration;
window.generateSessionId = generateSessionId;