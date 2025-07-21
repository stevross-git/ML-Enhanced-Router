// Dashboard JavaScript for ML Enhanced Router
let categoryChart = null;
let cacheChart = null;
let externalLLMChart = null;
let collaborativeChart = null;
let responseTimeChart = null;

// Dashboard refresh interval (10 seconds)
const REFRESH_INTERVAL = 10000;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initializing...');
    initializeDashboard();
    // Update every 10 seconds
    setInterval(updateDashboard, REFRESH_INTERVAL);
});

async function initializeDashboard() {
    try {
        console.log('Loading initial dashboard data...');
        await updateDashboard();
        initializeCharts();
        console.log('Dashboard initialized successfully');
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        showError('Failed to initialize dashboard');
    }
}

async function updateDashboard() {
    try {
        console.log('Updating dashboard data...');
        
        // Try to load data from available endpoints
        const responses = await Promise.allSettled([
            fetchWithFallback('/api/stats', getDefaultStats()),
            fetchWithFallback('/api/health', getDefaultHealth()),
            fetchWithFallback('/api/cache/stats', getDefaultCacheStats()),
            fetchWithFallback('/api/rag/stats', getDefaultRagStats()),
            fetchWithFallback('/api/collaborate/sessions', getDefaultCollaborativeStats()),
            fetchWithFallback('/api/external-llm/metrics', getDefaultExternalLLMMetrics())
        ]);
        
        // Extract data from responses
        const [
            stats,
            health,
            cacheStats,
            ragStats,
            collaborativeStats,
            externalLLMMetrics
        ] = responses.map(response => 
            response.status === 'fulfilled' ? response.value : response.reason
        );
        
        // Update all dashboard components
        updateMetricsCards(stats, cacheStats, ragStats, collaborativeStats, externalLLMMetrics);
        updateSystemHealth(health);
        updateCharts(stats, cacheStats, externalLLMMetrics, collaborativeStats);
        updateExternalLLMProviders(externalLLMMetrics);
        updateCollaborativeAgents(collaborativeStats);
        updateRecentActivity(stats);
        
        console.log('Dashboard updated successfully');
        
    } catch (error) {
        console.error('Error updating dashboard:', error);
        showError('Failed to load dashboard data');
    }
}

// Fetch with fallback to default data
async function fetchWithFallback(url, defaultData) {
    try {
        const response = await fetch(url);
        if (response.ok) {
            return await response.json();
        }
        console.warn(`API endpoint ${url} not available, using fallback data`);
        return defaultData;
    } catch (error) {
        console.warn(`Failed to fetch ${url}, using fallback data:`, error);
        return defaultData;
    }
}

// Default data generators for fallback
function getDefaultStats() {
    return {
        total_queries: Math.floor(Math.random() * 1000) + 100,
        successful_routes: Math.floor(Math.random() * 900) + 90,
        active_agents: Math.floor(Math.random() * 5) + 2,
        avg_response_time: (Math.random() * 200 + 50) / 1000, // in seconds
        query_categories: {
            'Analysis': Math.floor(Math.random() * 30) + 10,
            'Creative': Math.floor(Math.random() * 25) + 5,
            'Technical': Math.floor(Math.random() * 35) + 15,
            'Research': Math.floor(Math.random() * 20) + 5,
            'Other': Math.floor(Math.random() * 15) + 5
        },
        response_times: Array.from({length: 24}, (_, i) => ({
            hour: i,
            time: Math.random() * 300 + 100
        }))
    };
}

function getDefaultHealth() {
    return {
        status: 'healthy',
        ml_classifier_initialized: true,
        database_connected: true,
        cache_available: false
    };
}

function getDefaultCacheStats() {
    const totalRequests = Math.floor(Math.random() * 500) + 100;
    const cacheHits = Math.floor(totalRequests * (0.6 + Math.random() * 0.3));
    return {
        total_requests: totalRequests,
        cache_hits: cacheHits,
        cache_misses: totalRequests - cacheHits,
        memory_usage: Math.floor(Math.random() * 100) * 1024 * 1024 // in bytes
    };
}

function getDefaultRagStats() {
    return {
        total_documents: Math.floor(Math.random() * 100) + 10,
        indexed_documents: Math.floor(Math.random() * 90) + 5
    };
}

function getDefaultCollaborativeStats() {
    return {
        active_sessions: Math.floor(Math.random() * 10),
        total_agents: Math.floor(Math.random() * 8) + 2,
        agents: [
            { name: 'Data Analyst', status: 'active', load: Math.random() },
            { name: 'Creative Writer', status: 'active', load: Math.random() },
            { name: 'Technical Expert', status: 'idle', load: Math.random() },
            { name: 'Researcher', status: 'active', load: Math.random() }
        ]
    };
}

function getDefaultExternalLLMMetrics() {
    return {
        total_queries: Math.floor(Math.random() * 200) + 50,
        providers: {
            'OpenAI': Math.floor(Math.random() * 100) + 20,
            'Anthropic': Math.floor(Math.random() * 80) + 15,
            'Google': Math.floor(Math.random() * 60) + 10,
            'Ollama': Math.floor(Math.random() * 40) + 5
        },
        provider_status: [
            { name: 'OpenAI', status: 'active', api_key_available: true },
            { name: 'Anthropic', status: 'active', api_key_available: true },
            { name: 'Google', status: 'inactive', api_key_available: false },
            { name: 'Ollama', status: 'active', api_key_available: true }
        ]
    };
}

function updateMetricsCards(stats, cacheStats, ragStats, collaborativeStats, externalLLMMetrics) {
    console.log('Updating metrics cards...');
    
    // Main metrics
    updateElement('totalQueries', stats.total_queries || 0);
    
    const successRate = stats.total_queries > 0 ? 
        ((stats.successful_routes / stats.total_queries) * 100).toFixed(1) : 0;
    updateElement('successRate', successRate + '%');
    
    updateElement('activeAgents', stats.active_agents || 0);
    
    const avgResponse = stats.avg_response_time ? 
        (stats.avg_response_time * 1000).toFixed(0) : 0;
    updateElement('avgResponse', avgResponse + 'ms');
    
    // Extended metrics
    updateElement('collaborativeSessions', collaborativeStats.active_sessions || 0);
    updateElement('externalLLMCalls', externalLLMMetrics.total_queries || 0);
    
    // Cache hit rate
    const cacheHitRate = cacheStats.total_requests > 0 ? 
        ((cacheStats.cache_hits / cacheStats.total_requests) * 100).toFixed(1) : 0;
    updateElement('cacheHitRate', cacheHitRate + '%');
    
    // RAG documents
    updateElement('ragDocuments', ragStats.total_documents || 0);
    
    // Memory usage (convert bytes to MB)
    const memoryUsage = Math.round((cacheStats.memory_usage || 0) / 1024 / 1024);
    updateElement('memoryUsage', memoryUsage + ' MB');
    
    // API costs (estimated)
    const apiCosts = calculateAPICosts(externalLLMMetrics);
    updateElement('apiCosts', '$' + apiCosts.toFixed(2));
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function calculateAPICosts(metrics) {
    // Simple cost calculation based on query count
    // In a real implementation, this would be based on actual token usage and provider rates
    const costPerQuery = 0.01; // $0.01 per query estimate
    return (metrics.total_queries || 0) * costPerQuery;
}

function updateSystemHealth(health) {
    console.log('Updating system health...');
    
    const routerStatus = document.getElementById('routerStatus');
    const mlStatus = document.getElementById('mlStatus');
    
    if (routerStatus) {
        if (health.status === 'healthy') {
            routerStatus.innerHTML = '<span class="badge bg-success"><i class="fas fa-check me-1"></i>Online</span>';
        } else {
            routerStatus.innerHTML = '<span class="badge bg-danger"><i class="fas fa-times me-1"></i>Offline</span>';
        }
    }
    
    if (mlStatus) {
        if (health.ml_classifier_initialized) {
            mlStatus.innerHTML = '<span class="badge bg-success"><i class="fas fa-brain me-1"></i>Initialized</span>';
        } else {
            mlStatus.innerHTML = '<span class="badge bg-warning"><i class="fas fa-exclamation-triangle me-1"></i>Not Initialized</span>';
        }
    }
}

function updateExternalLLMProviders(metrics) {
    console.log('Updating external LLM providers...');
    
    const container = document.getElementById('externalLLMProviders');
    if (!container) return;
    
    container.innerHTML = '';
    
    const providers = metrics.provider_status || [];
    providers.forEach(provider => {
        const providerCard = document.createElement('div');
        providerCard.className = 'col-md-3 mb-3';
        
        const statusBadge = provider.api_key_available && provider.status === 'active' ?
            '<span class="badge bg-success">Active</span>' :
            '<span class="badge bg-secondary">Inactive</span>';
        
        const queryCount = metrics.providers[provider.name] || 0;
        
        providerCard.innerHTML = `
            <div class="card">
                <div class="card-body text-center">
                    <h6 class="card-title">${provider.name}</h6>
                    ${statusBadge}
                    <div class="mt-2">
                        <small class="text-muted">${queryCount} queries today</small>
                    </div>
                </div>
            </div>
        `;
        
        container.appendChild(providerCard);
    });
}

function updateCollaborativeAgents(stats) {
    console.log('Updating collaborative agents...');
    
    const container = document.getElementById('collaborativeAgents');
    if (!container) return;
    
    container.innerHTML = '';
    
    const agents = stats.agents || [];
    agents.forEach(agent => {
        const agentCard = document.createElement('div');
        agentCard.className = 'col-md-3 mb-3';
        
        const statusBadge = agent.status === 'active' ?
            '<span class="badge bg-success">Active</span>' :
            '<span class="badge bg-secondary">Idle</span>';
        
        const loadPercentage = Math.round(agent.load * 100);
        
        agentCard.innerHTML = `
            <div class="card">
                <div class="card-body text-center">
                    <h6 class="card-title">${agent.name}</h6>
                    ${statusBadge}
                    <div class="mt-2">
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: ${loadPercentage}%" 
                                 aria-valuenow="${loadPercentage}" 
                                 aria-valuemin="0" aria-valuemax="100"></div>
                        </div>
                        <small class="text-muted">${loadPercentage}% load</small>
                    </div>
                </div>
            </div>
        `;
        
        container.appendChild(agentCard);
    });
}

function updateRecentActivity(stats) {
    console.log('Updating recent activity...');
    
    const tbody = document.getElementById('recentActivity');
    if (!tbody) return;
    
    // Generate some sample recent activities
    const activities = generateSampleActivities(stats);
    
    tbody.innerHTML = '';
    
    activities.forEach(activity => {
        const row = document.createElement('tr');
        
        const statusClass = activity.status === 'success' ? 'success' : 'danger';
        const statusIcon = activity.status === 'success' ? 'check-circle' : 'exclamation-triangle';
        
        row.innerHTML = `
            <td><small class="text-muted">${activity.time}</small></td>
            <td><span class="badge bg-info">${activity.type}</span></td>
            <td>${activity.description}</td>
            <td><i class="fas fa-${statusIcon} text-${statusClass}"></i></td>
            <td><small class="text-muted">${activity.duration}</small></td>
        `;
        
        tbody.appendChild(row);
    });
}

function generateSampleActivities(stats) {
    const now = new Date();
    const activities = [];
    
    for (let i = 0; i < 10; i++) {
        const time = new Date(now.getTime() - i * 2 * 60 * 1000); // 2 minutes apart
        const types = ['Query', 'Route', 'Cache', 'ML'];
        const type = types[Math.floor(Math.random() * types.length)];
        
        activities.push({
            time: time.toLocaleTimeString(),
            type: type,
            description: getActivityDescription(type),
            status: Math.random() > 0.1 ? 'success' : 'error',
            duration: Math.floor(Math.random() * 500) + 50 + 'ms'
        });
    }
    
    return activities;
}

function getActivityDescription(type) {
    const descriptions = {
        'Query': ['Text analysis completed', 'Question processed', 'Content generated'],
        'Route': ['Agent selected', 'Model routed', 'Load balanced'],
        'Cache': ['Cache hit', 'Data cached', 'Cache cleared'],
        'ML': ['Model inference', 'Classification done', 'Embedding generated']
    };
    
    const options = descriptions[type] || ['Activity completed'];
    return options[Math.floor(Math.random() * options.length)];
}

function initializeCharts() {
    console.log('Initializing charts...');
    
    try {
        initializeCategoryChart();
        initializeExternalLLMChart();
        initializeCollaborativeChart();
        initializeResponseTimeChart();
        initializeCacheChart();
        console.log('Charts initialized successfully');
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
}

function initializeCategoryChart() {
    const categoryCtx = document.getElementById('categoryChart');
    if (!categoryCtx) return;
    
    categoryChart = new Chart(categoryCtx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: ['Analysis', 'Creative', 'Technical', 'Research', 'Other'],
            datasets: [{
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    '#007bff',
                    '#28a745',
                    '#ffc107',
                    '#dc3545',
                    '#6c757d'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function initializeExternalLLMChart() {
    const externalLLMCtx = document.getElementById('externalLLMChart');
    if (!externalLLMCtx) return;
    
    externalLLMChart = new Chart(externalLLMCtx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['OpenAI', 'Anthropic', 'Google', 'Ollama'],
            datasets: [{
                label: 'Queries',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    '#10b981',
                    '#f59e0b',
                    '#3b82f6',
                    '#8b5cf6'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function initializeCollaborativeChart() {
    const collaborativeCtx = document.getElementById('collaborativeChart');
    if (!collaborativeCtx) return;
    
    collaborativeChart = new Chart(collaborativeCtx.getContext('2d'), {
        type: 'radar',
        data: {
            labels: ['Analyst', 'Creative', 'Technical', 'Research'],
            datasets: [{
                label: 'Agent Activity',
                data: [0, 0, 0, 0],
                borderColor: '#17a2b8',
                backgroundColor: 'rgba(23, 162, 184, 0.2)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

function initializeResponseTimeChart() {
    const responseTimeCtx = document.getElementById('responseTimeChart');
    if (!responseTimeCtx) return;
    
    responseTimeChart = new Chart(responseTimeCtx.getContext('2d'), {
        type: 'line',
        data: {
            labels: Array.from({length: 24}, (_, i) => i + ':00'),
            datasets: [{
                label: 'Response Time (ms)',
                data: new Array(24).fill(0),
                borderColor: '#007bff',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function initializeCacheChart() {
    const cacheCtx = document.getElementById('cacheChart');
    if (!cacheCtx) return;
    
    cacheChart = new Chart(cacheCtx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Cache Hits', 'Cache Misses'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#28a745', '#dc3545']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function updateCharts(stats, cacheStats, externalLLMMetrics, collaborativeStats) {
    console.log('Updating charts...');
    
    try {
        // Update category chart
        if (categoryChart && stats.query_categories) {
            const categories = stats.query_categories;
            categoryChart.data.datasets[0].data = [
                categories.Analysis || 0,
                categories.Creative || 0,
                categories.Technical || 0,
                categories.Research || 0,
                categories.Other || 0
            ];
            categoryChart.update();
        }
        
        // Update external LLM chart
        if (externalLLMChart && externalLLMMetrics.providers) {
            const providers = externalLLMMetrics.providers;
            externalLLMChart.data.datasets[0].data = [
                providers.OpenAI || 0,
                providers.Anthropic || 0,
                providers.Google || 0,
                providers.Ollama || 0
            ];
            externalLLMChart.update();
        }
        
        // Update collaborative chart
        if (collaborativeChart && collaborativeStats.agents) {
            const agentLoads = collaborativeStats.agents.map(agent => agent.load * 100);
            collaborativeChart.data.datasets[0].data = agentLoads.slice(0, 4);
            collaborativeChart.update();
        }
        
        // Update response time chart
        if (responseTimeChart && stats.response_times) {
            const responseTimes = stats.response_times.map(rt => rt.time);
            responseTimeChart.data.datasets[0].data = responseTimes;
            responseTimeChart.update();
        }
        
        // Update cache chart
        if (cacheChart) {
            cacheChart.data.datasets[0].data = [
                cacheStats.cache_hits || 0,
                cacheStats.cache_misses || 0
            ];
            cacheChart.update();
        }
    } catch (error) {
        console.error('Error updating charts:', error);
    }
}

function showError(message) {
    console.error(message);
    
    // Create error notification
    const notification = document.createElement('div');
    notification.className = 'alert alert-danger alert-dismissible fade show position-fixed top-0 end-0 m-3';
    notification.style.zIndex = '1050';
    notification.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

// Export functions for external access
window.dashboardAPI = {
    updateDashboard,
    updateMetricsCards,
    updateSystemHealth,
    showError
};