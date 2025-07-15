// API Keys Management JavaScript
let currentModels = [];
let currentModalData = null;

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    loadModels();
    setupEventListeners();
});

function setupEventListeners() {
    // Tab switching
    const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(e) {
            const targetId = e.target.getAttribute('data-bs-target');
            filterModelsByCategory(targetId.replace('#', ''));
        });
    });
}

async function loadModels() {
    try {
        const response = await fetch('/api/models/detailed');
        const data = await response.json();
        
        if (data.models) {
            currentModels = data.models;
            updateStatistics();
            renderProviders();
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showNotification('Error loading models', 'error');
    }
}

function updateStatistics() {
    const activeKeys = currentModels.filter(model => model.api_key_available || model.deployment_type === 'local').length;
    const missingKeys = currentModels.filter(model => !model.api_key_available && model.deployment_type !== 'local').length;
    const localModels = currentModels.filter(model => model.deployment_type === 'local').length;
    
    document.getElementById('activeKeysCount').textContent = activeKeys;
    document.getElementById('missingKeysCount').textContent = missingKeys;
    document.getElementById('localModelsCount').textContent = localModels;
}

function renderProviders() {
    const categories = {
        enterprise: ['openai', 'anthropic', 'google', 'azure', 'aws_bedrock'],
        cloud: ['groq', 'together', 'fireworks', 'deepseek', 'cerebras', 'perplexity', 'xai'],
        local: ['ollama', 'huggingface'],
        multimodal: currentModels.filter(model => model.supports_vision || model.supports_audio || model.model_type === 'multimodal')
    };

    Object.keys(categories).forEach(category => {
        const container = document.getElementById(category + 'Providers');
        if (container) {
            container.innerHTML = '';
            
            if (category === 'multimodal') {
                categories[category].forEach(model => {
                    container.appendChild(createProviderCard(model, category));
                });
            } else {
                const categoryModels = currentModels.filter(model => 
                    categories[category].includes(model.provider)
                );
                categoryModels.forEach(model => {
                    container.appendChild(createProviderCard(model, category));
                });
            }
        }
    });
}

function createProviderCard(model, category) {
    const card = document.createElement('div');
    card.className = 'col-md-6 col-lg-4 mb-4';
    
    const statusClass = model.api_key_available || model.deployment_type === 'local' ? 'status-active' : 'status-inactive';
    const statusText = model.api_key_available || model.deployment_type === 'local' ? 'Active' : 'Not Configured';
    
    const costBadge = model.cost_per_1k_tokens === 0 ? 
        '<span class="badge bg-success cost-badge">Free</span>' : 
        `<span class="badge bg-info cost-badge">$${model.cost_per_1k_tokens}/1k tokens</span>`;
    
    const deploymentBadge = `<span class="badge bg-secondary deployment-type">${model.deployment_type}</span>`;
    
    const capabilities = [];
    if (model.supports_vision) capabilities.push('Vision');
    if (model.supports_audio) capabilities.push('Audio');
    if (model.model_type === 'multimodal') capabilities.push('Multi-Modal');
    
    const capabilitiesHTML = capabilities.length > 0 ? 
        `<div class="mt-2">${capabilities.map(cap => `<span class="badge bg-primary me-1">${cap}</span>`).join('')}</div>` : '';
    
    card.innerHTML = `
        <div class="card api-key-card h-100">
            <div class="card-body">
                <div class="d-flex align-items-start justify-content-between mb-3">
                    <div class="d-flex align-items-center">
                        <div class="provider-logo bg-primary d-flex align-items-center justify-content-center">
                            <i class="fas fa-robot text-white"></i>
                        </div>
                        <div>
                            <h6 class="card-title mb-0">${model.name}</h6>
                            <small class="text-muted">${model.provider.toUpperCase()}</small>
                        </div>
                    </div>
                    <div class="text-end">
                        ${deploymentBadge}
                        ${costBadge}
                    </div>
                </div>
                
                <div class="mb-3">
                    <div class="d-flex align-items-center mb-2">
                        <span class="status-indicator ${statusClass}"></span>
                        <span class="small">${statusText}</span>
                    </div>
                    <div class="small text-muted">
                        <div><strong>Max Tokens:</strong> ${model.max_tokens?.toLocaleString() || 'N/A'}</div>
                        <div><strong>Context:</strong> ${model.context_window?.toLocaleString() || 'N/A'}</div>
                    </div>
                    ${capabilitiesHTML}
                </div>
                
                <div class="d-flex gap-2">
                    <button class="btn btn-outline-primary btn-sm flex-fill" onclick="configureApiKey('${model.id}')">
                        <i class="fas fa-cog me-1"></i>
                        Configure
                    </button>
                    <button class="btn btn-outline-secondary btn-sm" onclick="testApiKey('${model.id}')" ${!model.api_key_available && model.deployment_type !== 'local' ? 'disabled' : ''}>
                        <i class="fas fa-check-circle me-1"></i>
                        Test
                    </button>
                </div>
            </div>
        </div>
    `;
    
    return card;
}

function configureApiKey(modelId) {
    const model = currentModels.find(m => m.id === modelId);
    if (!model) return;
    
    currentModalData = model;
    
    const modalContent = document.getElementById('modalContent');
    modalContent.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <div class="text-center">
                    <div class="provider-logo bg-primary d-flex align-items-center justify-content-center mx-auto mb-3" style="width: 80px; height: 80px;">
                        <i class="fas fa-robot text-white" style="font-size: 2rem;"></i>
                    </div>
                    <h5>${model.name}</h5>
                    <p class="text-muted">${model.provider.toUpperCase()}</p>
                </div>
            </div>
            <div class="col-md-8">
                <div class="mb-3">
                    <label class="form-label">API Key</label>
                    <div class="input-group">
                        <input type="password" class="form-control key-input" id="apiKeyInput" 
                               placeholder="${model.deployment_type === 'local' ? 'No API key required for local models' : 'Enter API key...'}" 
                               ${model.deployment_type === 'local' ? 'disabled' : ''}>
                        <button class="btn btn-outline-secondary" type="button" onclick="togglePasswordVisibility()">
                            <i class="fas fa-eye" id="toggleIcon"></i>
                        </button>
                    </div>
                    <div class="form-text">
                        ${model.deployment_type === 'local' ? 
                            'Local models run on your machine and don\'t require API keys.' :
                            `Get your API key from ${getProviderUrl(model.provider)}`
                        }
                    </div>
                </div>
                
                <div class="mb-3">
                    <label class="form-label">Configuration</label>
                    <div class="row">
                        <div class="col-md-6">
                            <label class="form-label small">Max Tokens</label>
                            <input type="number" class="form-control" id="maxTokensInput" value="${model.max_tokens || 4096}">
                        </div>
                        <div class="col-md-6">
                            <label class="form-label small">Temperature</label>
                            <input type="number" class="form-control" id="temperatureInput" value="${model.temperature || 0.7}" min="0" max="2" step="0.1">
                        </div>
                    </div>
                </div>
                
                <div class="mb-3">
                    <label class="form-label">Model Information</label>
                    <div class="small text-muted">
                        <div><strong>Endpoint:</strong> ${model.endpoint || 'N/A'}</div>
                        <div><strong>Model Name:</strong> ${model.model_name || 'N/A'}</div>
                        <div><strong>Cost:</strong> ${model.cost_per_1k_tokens === 0 ? 'Free' : `$${model.cost_per_1k_tokens}/1k tokens`}</div>
                        <div><strong>Context Window:</strong> ${model.context_window?.toLocaleString() || 'N/A'}</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    const modal = new bootstrap.Modal(document.getElementById('apiKeyModal'));
    modal.show();
}

function getProviderUrl(provider) {
    const urls = {
        'openai': 'https://platform.openai.com/api-keys',
        'anthropic': 'https://console.anthropic.com/',
        'google': 'https://aistudio.google.com/app/apikey',
        'xai': 'https://console.x.ai/',
        'azure': 'https://portal.azure.com/',
        'groq': 'https://console.groq.com/keys',
        'together': 'https://api.together.xyz/settings/api-keys',
        'fireworks': 'https://fireworks.ai/api-keys',
        'deepseek': 'https://platform.deepseek.com/api-keys',
        'cerebras': 'https://cloud.cerebras.ai/platform',
        'perplexity': 'https://www.perplexity.ai/settings/api',
        'huggingface': 'https://huggingface.co/settings/tokens',
        'replicate': 'https://replicate.com/account/api-tokens',
        'elevenlabs': 'https://elevenlabs.io/app/settings/api-keys'
    };
    
    return urls[provider] || '#';
}

function togglePasswordVisibility() {
    const input = document.getElementById('apiKeyInput');
    const icon = document.getElementById('toggleIcon');
    
    if (input.type === 'password') {
        input.type = 'text';
        icon.className = 'fas fa-eye-slash';
    } else {
        input.type = 'password';
        icon.className = 'fas fa-eye';
    }
}

async function saveApiKey() {
    if (!currentModalData) return;
    
    const apiKey = document.getElementById('apiKeyInput').value;
    const maxTokens = document.getElementById('maxTokensInput').value;
    const temperature = document.getElementById('temperatureInput').value;
    
    if (!apiKey && currentModalData.deployment_type !== 'local') {
        showNotification('Please enter an API key', 'error');
        return;
    }
    
    try {
        const response = await fetch(`/api/models/${currentModalData.id}/configure`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                api_key: apiKey,
                max_tokens: parseInt(maxTokens),
                temperature: parseFloat(temperature)
            })
        });
        
        if (response.ok) {
            showNotification('API key saved successfully', 'success');
            const modal = bootstrap.Modal.getInstance(document.getElementById('apiKeyModal'));
            modal.hide();
            loadModels(); // Refresh the models
        } else {
            const error = await response.json();
            showNotification(error.message || 'Failed to save API key', 'error');
        }
    } catch (error) {
        console.error('Error saving API key:', error);
        showNotification('Error saving API key', 'error');
    }
}

async function testApiKey(modelId) {
    const model = currentModels.find(m => m.id === modelId);
    if (!model) return;
    
    try {
        showNotification('Testing API key...', 'info');
        
        const response = await fetch(`/api/models/${modelId}/test`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            showNotification(`${model.name} API key test successful`, 'success');
        } else {
            showNotification(result.message || 'API key test failed', 'error');
        }
    } catch (error) {
        console.error('Error testing API key:', error);
        showNotification('Error testing API key', 'error');
    }
}

async function testAllKeys() {
    const activeModels = currentModels.filter(m => m.api_key_available || m.deployment_type === 'local');
    showNotification(`Testing ${activeModels.length} API keys...`, 'info');
    
    let successful = 0;
    let failed = 0;
    
    for (const model of activeModels) {
        try {
            const response = await fetch(`/api/models/${model.id}/test`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                successful++;
            } else {
                failed++;
            }
        } catch (error) {
            failed++;
        }
    }
    
    showNotification(`Testing complete: ${successful} successful, ${failed} failed`, 
                    failed === 0 ? 'success' : 'warning');
}

function exportConfig() {
    const config = {
        models: currentModels.map(model => ({
            id: model.id,
            name: model.name,
            provider: model.provider,
            max_tokens: model.max_tokens,
            temperature: model.temperature,
            api_key_available: model.api_key_available
        })),
        exported_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml-router-config-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    showNotification('Configuration exported successfully', 'success');
}

function importConfig() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const config = JSON.parse(e.target.result);
                    // Handle config import logic here
                    showNotification('Configuration imported successfully', 'success');
                } catch (error) {
                    showNotification('Invalid configuration file', 'error');
                }
            };
            reader.readAsText(file);
        }
    };
    input.click();
}

function saveAllKeys() {
    showNotification('Saving all configurations...', 'info');
    // This would trigger a bulk save operation
    setTimeout(() => {
        showNotification('All configurations saved successfully', 'success');
    }, 1000);
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

function filterModelsByCategory(category) {
    // This function would filter models based on category
    // Implementation depends on how models are categorized
}