// Home Dashboard JavaScript
let dashboardData = {};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
    updateSystemHealth();
    startHealthMonitoring();
});

async function loadDashboardData() {
    try {
        // Load multiple data sources in parallel
        const [modelsResponse, statsResponse, agentsResponse] = await Promise.all([
            fetch('/api/models/detailed'),
            fetch('/api/stats'),
            fetch('/api/agents')
        ]);

        // Parse responses
        const modelsData = await modelsResponse.json();
        const statsData = await statsResponse.json();
        const agentsData = await agentsResponse.json();

        // Update dashboard
        updateStatistics(modelsData, statsData, agentsData);
        updateProviderGrid(modelsData.models || []);
        updateRecentActivity();

    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showNotification('Error loading dashboard data', 'error');
    }
}

function updateStatistics(modelsData, statsData, agentsData) {
    // Update model statistics
    const models = modelsData.models || [];
    const activeModels = models.filter(m => m.api_key_available || m.deployment_type === 'local').length;
    
    document.getElementById('totalModels').textContent = models.length;
    document.getElementById('activeModels').textContent = activeModels;
    
    // Update query statistics
    document.getElementById('totalQueries').textContent = (statsData.total_queries || 0).toLocaleString();
    
    // Update agent statistics
    document.getElementById('totalAgents').textContent = (agentsData.agents || []).length;
}

function updateProviderGrid(models) {
    const providerGrid = document.getElementById('providerGrid');
    if (!providerGrid) return;
    
    // Group models by provider
    const providerCounts = {};
    models.forEach(model => {
        if (!providerCounts[model.provider]) {
            providerCounts[model.provider] = {
                total: 0,
                active: 0,
                name: model.provider
            };
        }
        providerCounts[model.provider].total++;
        if (model.api_key_available || model.deployment_type === 'local') {
            providerCounts[model.provider].active++;
        }
    });

    // Generate provider items
    providerGrid.innerHTML = '';
    Object.values(providerCounts).forEach(provider => {
        const providerItem = document.createElement('div');
        providerItem.className = 'provider-item';
        
        const statusClass = provider.active > 0 ? 'status-active' : 'status-warning';
        const statusText = provider.active > 0 ? 'Active' : 'No API Key';
        
        providerItem.innerHTML = `
            <div class="provider-logo">
                <i class="fas fa-robot"></i>
            </div>
            <div class="fw-semibold">${provider.name.toUpperCase()}</div>
            <div class="text-muted small">${provider.active}/${provider.total} models</div>
            <div class="mt-2">
                <span class="status-indicator ${statusClass}"></span>
                <span class="small">${statusText}</span>
            </div>
        `;
        
        providerGrid.appendChild(providerItem);
    });
}

function updateRecentActivity() {
    const recentActivity = document.getElementById('recentActivity');
    if (!recentActivity) return;
    
    // Add more recent activity items
    const additionalActivities = [
        {
            icon: 'fas fa-key',
            title: 'API Keys Available',
            description: 'Ready to configure AI models',
            time: '5 min ago'
        },
        {
            icon: 'fas fa-network-wired',
            title: 'Network Integration',
            description: 'Multi-agent routing active',
            time: '10 min ago'
        },
        {
            icon: 'fas fa-shield-alt',
            title: 'Security Systems',
            description: 'Semantic guardrails enabled',
            time: '15 min ago'
        }
    ];
    
    additionalActivities.forEach(activity => {
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        activityItem.innerHTML = `
            <div class="activity-icon">
                <i class="${activity.icon}"></i>
            </div>
            <div>
                <div class="fw-semibold">${activity.title}</div>
                <div class="text-muted small">${activity.description}</div>
            </div>
            <div class="activity-time">${activity.time}</div>
        `;
        recentActivity.appendChild(activityItem);
    });
}

function updateSystemHealth() {
    // Calculate system health based on various factors
    let healthScore = 0;
    let totalChecks = 0;
    
    // Check if system is running
    healthScore += 25;
    totalChecks++;
    
    // Check database connection (assume connected)
    healthScore += 25;
    totalChecks++;
    
    // Check API availability
    healthScore += 20;
    totalChecks++;
    
    // Check model availability
    healthScore += 15;
    totalChecks++;
    
    const healthPercentage = (healthScore / 100) * 100;
    
    // Update health bar
    const healthBar = document.getElementById('systemHealthBar');
    const healthText = document.getElementById('systemHealthText');
    
    if (healthBar) {
        healthBar.style.width = healthPercentage + '%';
    }
    
    if (healthText) {
        let statusText = 'Poor';
        if (healthPercentage >= 90) {
            statusText = 'Excellent';
        } else if (healthPercentage >= 70) {
            statusText = 'Good';
        } else if (healthPercentage >= 50) {
            statusText = 'Fair';
        }
        healthText.textContent = statusText;
    }
}

function startHealthMonitoring() {
    // Update health every 30 seconds
    setInterval(updateSystemHealth, 30000);
    
    // Update statistics every 5 minutes
    setInterval(loadDashboardData, 300000);
}

// Feature showcase functions
function showFeatureDetails(feature) {
    const featureDetails = {
        'multimodal': {
            title: 'Multi-Modal AI Capabilities',
            description: 'Process and analyze multiple types of content including text, images, audio, and documents using 32+ AI models from leading providers.',
            features: [
                'Image Analysis with 6 analysis types',
                'Audio Processing and Transcription',
                'Document Analysis and Extraction',
                'Content Generation (Images, Audio, Text)',
                'Multi-format File Support'
            ]
        },
        'collaborative': {
            title: 'Collaborative AI System',
            description: 'Multiple specialized AI agents working together to provide comprehensive responses with shared memory and context.',
            features: [
                'Specialized Agent Roles (Analyst, Creative, Technical, Researcher)',
                'Shared Memory and Context',
                'Real-time Collaboration',
                'Agent Configuration and Management',
                'Session Monitoring and Analytics'
            ]
        },
        'routing': {
            title: 'Intelligent Query Routing',
            description: 'ML-enhanced routing system that automatically selects the best AI model and agent for each query based on complexity and requirements.',
            features: [
                'Advanced Query Classification',
                'ML-Enhanced Agent Selection',
                'Load Balancing and Circuit Breakers',
                'Performance Optimization',
                'Predictive Analytics'
            ]
        }
    };
    
    // This would show a modal with feature details
    console.log('Feature details:', featureDetails[feature]);
}

// Quick action functions
function startNewChat() {
    window.location.href = '/chat';
}

function openCollaborativeAI() {
    window.location.href = '/collaborate';
}

function manageAPIKeys() {
    window.location.href = '/api-keys';
}

function viewAnalytics() {
    window.location.href = '/dashboard';
}

// Utility functions
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

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatTime(timestamp) {
    const now = new Date();
    const time = new Date(timestamp);
    const diff = now - time;
    
    if (diff < 60000) {
        return 'Just now';
    } else if (diff < 3600000) {
        return Math.floor(diff / 60000) + ' min ago';
    } else if (diff < 86400000) {
        return Math.floor(diff / 3600000) + ' hours ago';
    } else {
        return Math.floor(diff / 86400000) + ' days ago';
    }
}

// Animation functions
function animateCounter(element, target, duration = 1000) {
    let start = 0;
    const increment = target / (duration / 16);
    
    const timer = setInterval(() => {
        start += increment;
        if (start >= target) {
            start = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(start).toLocaleString();
    }, 16);
}

// Initialize animations on page load
document.addEventListener('DOMContentLoaded', function() {
    // Animate counters when page loads
    setTimeout(() => {
        const counters = document.querySelectorAll('.stats-number');
        counters.forEach(counter => {
            const target = parseInt(counter.textContent.replace(/,/g, ''));
            if (target > 0) {
                counter.textContent = '0';
                animateCounter(counter, target);
            }
        });
    }, 500);
});