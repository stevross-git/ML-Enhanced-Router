// Multi-Modal AI Chat JavaScript
let selectedModel = null;
let chatHistory = [];
let uploadedFiles = [];
let messageCount = 0;
let tokenCount = 0;

// Initialize the chat interface
document.addEventListener('DOMContentLoaded', function() {
    loadModels();
    setupDragAndDrop();
    setupVoiceRecognition();
    updateStats();
});

async function loadModels() {
    try {
        const response = await fetch('/api/models/detailed');
        const data = await response.json();
        
        const modelSelector = document.getElementById('modelSelector');
        modelSelector.innerHTML = '<option value="">Select a model...</option>';
        
        // Group models by provider
        const groupedModels = {};
        data.models.forEach(model => {
            if (!groupedModels[model.provider]) {
                groupedModels[model.provider] = [];
            }
            groupedModels[model.provider].push(model);
        });
        
        // Add grouped options
        Object.keys(groupedModels).forEach(provider => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = provider.toUpperCase();
            
            groupedModels[provider].forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                option.dataset.modelData = JSON.stringify(model);
                
                if (!model.api_key_available && model.deployment_type !== 'local') {
                    option.disabled = true;
                    option.textContent += ' (No API Key)';
                }
                
                optgroup.appendChild(option);
            });
            
            modelSelector.appendChild(optgroup);
        });
        
        // Select first available model
        const firstAvailable = data.models.find(m => m.api_key_available || m.deployment_type === 'local');
        if (firstAvailable) {
            modelSelector.value = firstAvailable.id;
            updateModel();
        }
        
    } catch (error) {
        console.error('Error loading models:', error);
        showNotification('Error loading models', 'error');
    }
}

function updateModel() {
    const modelSelector = document.getElementById('modelSelector');
    const selectedOption = modelSelector.options[modelSelector.selectedIndex];
    
    if (selectedOption && selectedOption.dataset.modelData) {
        selectedModel = JSON.parse(selectedOption.dataset.modelData);
        
        // Update model info
        const modelInfo = document.getElementById('modelInfo');
        const capabilities = selectedModel.capabilities || [];
        const capabilityText = capabilities.length > 0 ? capabilities.join(', ') : 'Text generation';
        
        modelInfo.innerHTML = `
            <strong>Provider:</strong> ${selectedModel.provider.toUpperCase()}<br>
            <strong>Type:</strong> ${selectedModel.model_type || 'text'}<br>
            <strong>Capabilities:</strong> ${capabilityText}<br>
            <strong>Cost:</strong> ${selectedModel.cost_per_1k_tokens === 0 ? 'Free' : `$${selectedModel.cost_per_1k_tokens}/1k tokens`}
        `;
        
        // Update settings
        document.getElementById('temperatureSlider').value = selectedModel.temperature || 0.7;
        document.getElementById('maxTokensInput').value = selectedModel.max_tokens || 1000;
        updateTemperature();
        
        addSystemMessage(`Switched to ${selectedModel.name}`);
    }
}

function updateTemperature() {
    const slider = document.getElementById('temperatureSlider');
    const value = document.getElementById('temperatureValue');
    value.textContent = slider.value;
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    if (!selectedModel) {
        showNotification('Please select a model first', 'error');
        return;
    }
    
    // Add user message to chat
    addMessage('user', message);
    messageInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                model_id: selectedModel.id,
                temperature: parseFloat(document.getElementById('temperatureSlider').value),
                max_tokens: parseInt(document.getElementById('maxTokensInput').value),
                enable_rag: document.getElementById('enableRAG').checked,
                enable_streaming: document.getElementById('enableStreaming').checked,
                chat_history: chatHistory
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to send message');
        }
        
        const data = await response.json();
        
        // Add assistant response
        addMessage('assistant', data.response);
        
        // Update token count
        tokenCount += data.tokens_used || 0;
        updateStats();
        
    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('system', 'Error: Failed to send message. Please try again.');
    } finally {
        hideTypingIndicator();
    }
}

function addMessage(sender, content) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    // Add sender icon
    let icon = '';
    if (sender === 'user') {
        icon = '<i class="fas fa-user me-2"></i>';
    } else if (sender === 'assistant') {
        icon = '<i class="fas fa-robot me-2"></i>';
    } else if (sender === 'system') {
        icon = '<i class="fas fa-info-circle me-2"></i>';
    }
    
    messageDiv.innerHTML = `${icon}${content}`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Update chat history
    if (sender !== 'system') {
        chatHistory.push({ role: sender, content: content });
        messageCount++;
        updateStats();
    }
}

function addSystemMessage(message) {
    addMessage('system', message);
}

function showTypingIndicator() {
    document.getElementById('typingIndicator').classList.add('active');
}

function hideTypingIndicator() {
    document.getElementById('typingIndicator').classList.remove('active');
}

function updateStats() {
    document.getElementById('messageCount').textContent = messageCount;
    document.getElementById('tokenCount').textContent = tokenCount.toLocaleString();
}

function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        document.getElementById('chatMessages').innerHTML = `
            <div class="message system">
                <i class="fas fa-robot me-2"></i>
                Welcome to the Multi-Modal AI Chat! You can send text, upload images, documents, and audio files for analysis.
            </div>
        `;
        chatHistory = [];
        messageCount = 0;
        tokenCount = 0;
        updateStats();
    }
}

// File upload handlers
function setupDragAndDrop() {
    const uploadAreas = document.querySelectorAll('.file-upload-area');
    
    uploadAreas.forEach(area => {
        area.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });
        
        area.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });
        
        area.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            handleFileUpload(files, this.id);
        });
    });
}

function handleImageUpload(event) {
    const files = event.target.files;
    displayUploadedFiles(files, 'uploadedImages', 'image');
}

function handleAudioUpload(event) {
    const files = event.target.files;
    displayUploadedFiles(files, 'uploadedAudio', 'audio');
}

function handleDocumentUpload(event) {
    const files = event.target.files;
    displayUploadedFiles(files, 'uploadedDocuments', 'document');
}

function displayUploadedFiles(files, containerId, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    Array.from(files).forEach((file, index) => {
        const fileDiv = document.createElement('div');
        fileDiv.className = 'uploaded-file';
        fileDiv.innerHTML = `
            <i class="fas fa-${getFileIcon(type)} me-2"></i>
            <span>${file.name}</span>
            <i class="fas fa-times remove-file" onclick="removeFile('${containerId}', ${index})"></i>
        `;
        container.appendChild(fileDiv);
    });
    
    // Store files for processing
    uploadedFiles = Array.from(files);
}

function getFileIcon(type) {
    switch(type) {
        case 'image': return 'image';
        case 'audio': return 'music';
        case 'document': return 'file-alt';
        default: return 'file';
    }
}

function removeFile(containerId, index) {
    uploadedFiles.splice(index, 1);
    const container = document.getElementById(containerId);
    container.children[index].remove();
}

// Multi-modal processing functions
async function analyzeImage() {
    const queryInput = document.getElementById('imageQueryInput');
    const query = queryInput.value.trim() || 'Analyze this image';
    
    if (uploadedFiles.length === 0) {
        showNotification('Please upload an image first', 'error');
        return;
    }
    
    addMessage('user', `Image analysis request: ${query}`);
    showTypingIndicator();
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedFiles[0]);
        formData.append('analysis_type', 'general');
        formData.append('query', query);
        
        const response = await fetch('/api/multimodal/analyze-image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.image_analysis && data.image_analysis.result) {
            addMessage('assistant', data.image_analysis.result.analysis);
        } else {
            addMessage('system', 'Image analysis failed');
        }
        
    } catch (error) {
        console.error('Error analyzing image:', error);
        addMessage('system', 'Error analyzing image');
    } finally {
        hideTypingIndicator();
    }
}

async function processAudio() {
    const analysisType = document.getElementById('audioAnalysisType').value;
    
    if (uploadedFiles.length === 0) {
        showNotification('Please upload an audio file first', 'error');
        return;
    }
    
    addMessage('user', `Audio ${analysisType} request`);
    showTypingIndicator();
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedFiles[0]);
        formData.append('analysis_type', analysisType);
        
        const response = await fetch('/api/multimodal/analyze-audio', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.audio_analysis && data.audio_analysis.result) {
            addMessage('assistant', data.audio_analysis.result.analysis);
        } else {
            addMessage('system', 'Audio processing failed');
        }
        
    } catch (error) {
        console.error('Error processing audio:', error);
        addMessage('system', 'Error processing audio');
    } finally {
        hideTypingIndicator();
    }
}

async function analyzeDocument() {
    const analysisType = document.getElementById('documentAnalysisType').value;
    
    if (uploadedFiles.length === 0) {
        showNotification('Please upload a document first', 'error');
        return;
    }
    
    addMessage('user', `Document ${analysisType} request`);
    showTypingIndicator();
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedFiles[0]);
        formData.append('analysis_type', analysisType);
        
        const response = await fetch('/api/multimodal/analyze-document', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.document_analysis && data.document_analysis.result) {
            addMessage('assistant', data.document_analysis.result.analysis);
        } else {
            addMessage('system', 'Document analysis failed');
        }
        
    } catch (error) {
        console.error('Error analyzing document:', error);
        addMessage('system', 'Error analyzing document');
    } finally {
        hideTypingIndicator();
    }
}

async function generateContent() {
    const generationType = document.getElementById('generationType').value;
    const quality = document.getElementById('generationQuality').value;
    const prompt = document.getElementById('generationPrompt').value.trim();
    
    if (!prompt) {
        showNotification('Please enter a description for generation', 'error');
        return;
    }
    
    addMessage('user', `Generate ${generationType}: ${prompt}`);
    showTypingIndicator();
    
    try {
        let endpoint = '';
        const requestData = {
            prompt: prompt,
            quality: quality
        };
        
        if (generationType === 'image') {
            endpoint = '/api/multimodal/generate-image';
        } else if (generationType === 'audio') {
            endpoint = '/api/multimodal/generate-speech';
            requestData.text = prompt;
            requestData.voice = quality;
        } else {
            // For text generation, use regular chat
            endpoint = '/api/chat';
            requestData.message = prompt;
            requestData.model_id = selectedModel.id;
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (generationType === 'image' && data.image_generation) {
            addMessage('assistant', `Image generated successfully: ${data.image_generation.result.status}`);
        } else if (generationType === 'audio' && data.speech_generation) {
            addMessage('assistant', `Speech generated successfully: ${data.speech_generation.result.status}`);
        } else if (generationType === 'text' && data.response) {
            addMessage('assistant', data.response);
        } else {
            addMessage('system', 'Content generation failed');
        }
        
    } catch (error) {
        console.error('Error generating content:', error);
        addMessage('system', 'Error generating content');
    } finally {
        hideTypingIndicator();
    }
}

// Voice recognition setup
function setupVoiceRecognition() {
    if ('webkitSpeechRecognition' in window) {
        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = function(event) {
            const result = event.results[0][0].transcript;
            document.getElementById('messageInput').value = result;
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            showNotification('Speech recognition error', 'error');
        };
        
        window.speechRecognition = recognition;
    }
}

function toggleVoiceInput() {
    const voiceBtn = document.getElementById('voiceBtn');
    
    if (window.speechRecognition) {
        if (voiceBtn.classList.contains('active')) {
            window.speechRecognition.stop();
            voiceBtn.classList.remove('active');
        } else {
            window.speechRecognition.start();
            voiceBtn.classList.add('active');
        }
    } else {
        showNotification('Speech recognition not supported', 'error');
    }
}

function updateGenerationOptions() {
    const generationType = document.getElementById('generationType').value;
    const qualitySelect = document.getElementById('generationQuality');
    
    qualitySelect.innerHTML = '';
    
    if (generationType === 'image') {
        qualitySelect.innerHTML = `
            <option value="standard">Standard</option>
            <option value="hd">HD Quality</option>
            <option value="artistic">Artistic</option>
            <option value="photorealistic">Photorealistic</option>
        `;
    } else if (generationType === 'audio') {
        qualitySelect.innerHTML = `
            <option value="alloy">Alloy</option>
            <option value="echo">Echo</option>
            <option value="fable">Fable</option>
            <option value="onyx">Onyx</option>
            <option value="nova">Nova</option>
            <option value="shimmer">Shimmer</option>
        `;
    } else {
        qualitySelect.innerHTML = `
            <option value="creative">Creative</option>
            <option value="balanced">Balanced</option>
            <option value="precise">Precise</option>
            <option value="technical">Technical</option>
        `;
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}