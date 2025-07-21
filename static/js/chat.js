// Multi-Modal AI Chat JavaScript - Updated to use stream and message endpoints
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

// Updated sendMessage function to use existing endpoints
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
    
    const enableStreaming = document.getElementById('enableStreaming').checked;
    
    try {
        if (enableStreaming) {
            // Use streaming endpoint
            await handleStreamingResponse(message);
        } else {
            // Use regular message endpoint
            await handleRegularResponse(message);
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('system', `Error: ${error.message}. Please try again.`);
    } finally {
        hideTypingIndicator();
    }
}

// Replace the handleRegularResponse function in chat.js with this final fixed version

async function handleRegularResponse(message) {
    const response = await fetch('/api/chat/message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: message,
            model_id: selectedModel.id,
            temperature: parseFloat(document.getElementById('temperatureSlider').value),
            max_tokens: parseInt(document.getElementById('maxTokensInput').value),
            system_message: null
        })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('Response data:', data); // Debug log
    
    // Handle the response structure
    if (data.status === 'success') {
        let messageContent = '';
        
        // Handle nested response structure (cached responses)
        if (data.response && typeof data.response === 'object' && data.response.response) {
            messageContent = data.response.response;
        } 
        // Handle direct response structure
        else if (data.response && typeof data.response === 'string') {
            messageContent = data.response;
        }
        else {
            throw new Error('No valid response content found');
        }
        
        // Add assistant response
        addMessage('assistant', messageContent);
        
        // Update token count - handle different usage formats
        let tokensUsed = 0;
        
        if (data.usage) {
            tokensUsed = data.usage.total_tokens || 
                        (data.usage.input_tokens + data.usage.output_tokens) || 
                        (data.usage.prompt_tokens + data.usage.completion_tokens) || 0;
        }
        
        // For cached responses, get metadata if available
        if (data.cached && data.response && data.response.metadata) {
            // Use estimated tokens for cached responses if no usage data
            tokensUsed = tokensUsed || 50; // Rough estimate for short responses
        }
        
        tokenCount += tokensUsed;
        updateStats();
        
        // Show cache indicator if response was cached
        if (data.cached) {
            console.log('✅ Response served from cache');
        }
        
    } else if (data.status === 'error') {
        throw new Error(data.error || 'Unknown error occurred');
    } else {
        // Handle legacy response format (if any)
        if (data.response) {
            const messageContent = typeof data.response === 'object' ? 
                                 data.response.response || JSON.stringify(data.response) : 
                                 data.response;
            addMessage('assistant', messageContent);
            const tokensUsed = data.tokens_used || 0;
            tokenCount += tokensUsed;
            updateStats();
        } else {
            throw new Error('Invalid response format from server');
        }
    }
}

async function handleStreamingResponse(message) {
    // Use POST version of streaming endpoint
    const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            model_id: selectedModel.id,
            system_message: null
        })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    // Handle Server-Sent Events
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let assistantMessageDiv = null;
    let fullResponse = '';
    
    // Create a message div for streaming response
    const chatMessages = document.getElementById('chatMessages');
    assistantMessageDiv = document.createElement('div');
    assistantMessageDiv.className = 'message assistant';
    assistantMessageDiv.innerHTML = '<i class="fas fa-robot me-2"></i><span class="message-content"></span>';
    chatMessages.appendChild(assistantMessageDiv);
    
    const messageContent = assistantMessageDiv.querySelector('.message-content');
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'start') {
                            console.log('Streaming started for model:', data.model);
                        } else if (data.type === 'token') {
                            fullResponse += data.content;
                            messageContent.textContent = fullResponse;
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        } else if (data.type === 'end') {
                            console.log('Streaming completed');
                            if (data.usage) {
                                const tokensUsed = data.usage.total_tokens || 
                                                 (data.usage.input_tokens + data.usage.output_tokens) || 0;
                                tokenCount += tokensUsed;
                                updateStats();
                            }
                        } else if (data.type === 'error') {
                            throw new Error(data.error);
                        }
                    } catch (parseError) {
                        console.error('Error parsing SSE data:', parseError);
                    }
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
    
    // Update chat history
    if (fullResponse) {
        chatHistory.push({ role: 'assistant', content: fullResponse });
        messageCount++;
        updateStats();
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

function handleFileUpload(files, uploadAreaId) {
    if (uploadAreaId.includes('image')) {
        displayUploadedFiles(files, 'uploadedImages', 'image');
    } else if (uploadAreaId.includes('audio')) {
        displayUploadedFiles(files, 'uploadedAudio', 'audio');
    } else if (uploadAreaId.includes('document')) {
        displayUploadedFiles(files, 'uploadedDocuments', 'document');
    }
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
        } else if (data.error) {
            addMessage('system', `Image analysis failed: ${data.error}`);
        } else {
            addMessage('system', 'Image analysis failed');
        }
        
    } catch (error) {
        console.error('Error analyzing image:', error);
        addMessage('system', `Error analyzing image: ${error.message}`);
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
        } else if (data.error) {
            addMessage('system', `Audio processing failed: ${data.error}`);
        } else {
            addMessage('system', 'Audio processing failed');
        }
        
    } catch (error) {
        console.error('Error processing audio:', error);
        addMessage('system', `Error processing audio: ${error.message}`);
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
        } else if (data.error) {
            addMessage('system', `Document analysis failed: ${data.error}`);
        } else {
            addMessage('system', 'Document analysis failed');
        }
        
    } catch (error) {
        console.error('Error analyzing document:', error);
        addMessage('system', `Error analyzing document: ${error.message}`);
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
            // For text generation, use regular chat endpoint
            const enableStreaming = document.getElementById('enableStreaming').checked;
            
            if (enableStreaming) {
                await handleStreamingResponse(prompt);
                return;
            } else {
                await handleRegularResponse(prompt);
                return;
            }
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
        } else if (data.error) {
            addMessage('system', `Content generation failed: ${data.error}`);
        } else {
            addMessage('system', 'Content generation failed');
        }
        
    } catch (error) {
        console.error('Error generating content:', error);
        addMessage('system', `Error generating content: ${error.message}`);
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
        
        recognition.onend = function() {
            const voiceBtn = document.getElementById('voiceBtn');
            if (voiceBtn) {
                voiceBtn.classList.remove('active');
            }
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
        if (document.body.contains(notification)) {
            notification.remove();
        }
    }, 5000);
}

// Alternative streaming method using EventSource (if POST streaming doesn't work)
async function handleStreamingResponseAlternative(message) {
    const params = new URLSearchParams({
        query: message,
        model_id: selectedModel.id,
        system_message: ''
    });
    
    const eventSource = new EventSource(`/api/chat/stream?${params}`);
    let fullResponse = '';
    let assistantMessageDiv = null;
    
    // Create a message div for streaming response
    const chatMessages = document.getElementById('chatMessages');
    assistantMessageDiv = document.createElement('div');
    assistantMessageDiv.className = 'message assistant';
    assistantMessageDiv.innerHTML = '<i class="fas fa-robot me-2"></i><span class="message-content"></span>';
    chatMessages.appendChild(assistantMessageDiv);
    
    const messageContent = assistantMessageDiv.querySelector('.message-content');
    
    return new Promise((resolve, reject) => {
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'start') {
                    console.log('Streaming started for model:', data.model);
                } else if (data.type === 'token') {
                    fullResponse += data.content;
                    messageContent.textContent = fullResponse;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                } else if (data.type === 'end') {
                    console.log('Streaming completed');
                    if (data.usage) {
                        const tokensUsed = data.usage.total_tokens || 0;
                        tokenCount += tokensUsed;
                        updateStats();
                    }
                    
                    // Update chat history
                    if (fullResponse) {
                        chatHistory.push({ role: 'assistant', content: fullResponse });
                        messageCount++;
                        updateStats();
                    }
                    
                    eventSource.close();
                    resolve();
                } else if (data.type === 'error') {
                    eventSource.close();
                    reject(new Error(data.error));
                }
            } catch (parseError) {
                console.error('Error parsing SSE data:', parseError);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('EventSource error:', error);
            eventSource.close();
            reject(new Error('Streaming connection failed'));
        };
    });
}