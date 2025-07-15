# Enterprise Features Usage Guide

This guide provides step-by-step instructions for using the advanced enterprise features in the ML Query Router.

## Getting Started

### Prerequisites
- ML Query Router running on port 5000
- Enterprise features enabled in configuration
- User authentication configured (if required)

### Accessing Enterprise Features
The enterprise features are accessible through:
- **REST API**: Direct API calls to enterprise endpoints
- **Web Interface**: Integrated with Personal AI router interface
- **Swagger UI**: Interactive API documentation at `/api/docs`

## Cross-Persona Memory Inference

### Step 1: Analyze Persona Compatibility
```bash
curl -X POST http://localhost:5000/api/cross-persona/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "persona_1": "work_persona",
    "persona_2": "learning_persona",
    "persona_1_data": {
      "skills": ["analysis", "communication", "problem_solving"],
      "interests": ["technology", "productivity", "efficiency"],
      "preferences": {"formality": 0.8, "structure": 0.9}
    },
    "persona_2_data": {
      "skills": ["research", "critical_thinking", "problem_solving"],
      "interests": ["science", "technology", "learning"],
      "preferences": {"formality": 0.7, "structure": 0.8}
    }
  }'
```

### Step 2: Generate Linkage Graph
```bash
curl -X GET "http://localhost:5000/api/cross-persona/linkage-graph?user_id=demo_user"
```

### Step 3: Get Cross-Persona Insights
```bash
curl -X GET "http://localhost:5000/api/cross-persona/insights?user_id=demo_user"
```

### Expected Response
```json
{
  "linkages": [
    {
      "persona_1": "work_persona",
      "persona_2": "learning_persona",
      "linkage_type": "skill_transfer",
      "confidence": 0.85,
      "description": "Shared problem-solving skills enable knowledge transfer",
      "created_at": "2025-07-15T08:00:00Z"
    }
  ],
  "insights": [
    {
      "insight_type": "skill_complementarity",
      "description": "Work and learning personas have complementary skills",
      "confidence": 0.75,
      "recommendation": "Consider combining personas for complex analytical tasks"
    }
  ]
}
```

## Cognitive Loop Debugging

### Step 1: Start Debugging Session
```bash
curl -X POST http://localhost:5000/api/cognitive/session \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "session_metadata": {
      "purpose": "routing_optimization",
      "context": "debugging_complex_queries"
    }
  }'
```

### Step 2: Submit Query for Debugging
Submit a query through the normal routing process while debugging is active:
```bash
curl -X POST http://localhost:5000/api/submit \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can I optimize my work-life balance?",
    "user_id": "demo_user"
  }'
```

### Step 3: Get Logged Decisions
```bash
curl -X GET "http://localhost:5000/api/cognitive/decisions?user_id=demo_user&limit=5"
```

### Step 4: Explain Specific Decision
```bash
curl -X POST http://localhost:5000/api/cognitive/explain \
  -H "Content-Type: application/json" \
  -d '{
    "decision_id": "decision_demo_user_20250715_080000"
  }'
```

### Expected Response
```json
{
  "decisions": [
    {
      "decision_id": "decision_demo_user_20250715_080000",
      "decision_type": "agent_selection",
      "decision_made": "Selected wellness_agent for work-life balance query",
      "reasoning": "Query contains work-life balance keywords and personal optimization themes",
      "confidence": 0.92,
      "context": {
        "query_category": "personal_optimization",
        "persona_active": "work_persona",
        "time_context": "evening"
      },
      "alternatives": ["productivity_agent", "general_assistant"],
      "timestamp": "2025-07-15T08:00:00Z"
    }
  ],
  "explanation": {
    "decision_path": [
      "Query received: 'How can I optimize my work-life balance?'",
      "Classified as personal_optimization (confidence: 0.89)",
      "Active persona: work_persona",
      "Available agents: wellness_agent, productivity_agent, general_assistant",
      "Wellness_agent selected (confidence: 0.92)"
    ],
    "reasoning_factors": [
      "Keyword match: 'work-life balance' strongly associated with wellness",
      "Persona context: work_persona often deals with productivity topics",
      "Time context: Evening queries often relate to personal well-being"
    ]
  }
}
```

## Temporal Memory Weighting

### Step 1: Get Current Memory State
```bash
curl -X GET "http://localhost:5000/api/temporal/memories?user_id=demo_user&limit=10"
```

### Step 2: Get Memory Insights
```bash
curl -X GET "http://localhost:5000/api/temporal/insights?user_id=demo_user"
```

### Step 3: Optimize Memory Storage
```bash
curl -X POST http://localhost:5000/api/temporal/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "optimization_type": "decay_optimization"
  }'
```

### Expected Response
```json
{
  "memories": [
    {
      "memory_id": "important_meeting_20250715",
      "content": "Meeting with client about project requirements",
      "confidence": 0.87,
      "priority": "high",
      "access_count": 5,
      "last_accessed": "2025-07-15T07:30:00Z",
      "created_at": "2025-07-15T06:00:00Z",
      "temporal_weight": 0.92,
      "relevance_score": 0.89
    }
  ],
  "insights": [
    {
      "insight_type": "memory_pattern_analysis",
      "description": "High-priority memories maintain relevance longer",
      "confidence": 0.85,
      "recommendation": "Consider increasing priority for frequently accessed memories"
    }
  ],
  "optimization_results": {
    "memories_optimized": 15,
    "performance_improvement": 0.23,
    "storage_efficiency": 0.18
  }
}
```

## Web Interface Usage

### Personal AI Router Integration
1. Navigate to the Personal AI interface
2. Enterprise features are automatically integrated
3. View cognitive debugging information in real-time
4. Access cross-persona insights through the persona management panel
5. Monitor temporal memory optimization in the memory dashboard

### Dashboard Integration
- **Cross-Persona Metrics**: View linkage statistics and compatibility scores
- **Cognitive Debugging**: Monitor decision patterns and confidence levels
- **Temporal Memory**: Track memory optimization and relevance trends

## Common Use Cases

### 1. Debugging Query Routing Issues
```bash
# Start debugging session
curl -X POST http://localhost:5000/api/cognitive/session \
  -d '{"user_id": "troubleshoot_user"}'

# Submit problematic query
curl -X POST http://localhost:5000/api/submit \
  -d '{"query": "Complex technical question", "user_id": "troubleshoot_user"}'

# Analyze decision
curl -X GET "http://localhost:5000/api/cognitive/decisions?user_id=troubleshoot_user"
```

### 2. Optimizing Persona Relationships
```bash
# Analyze all persona combinations
for persona1 in work learning creative; do
  for persona2 in work learning creative; do
    if [ "$persona1" != "$persona2" ]; then
      curl -X POST http://localhost:5000/api/cross-persona/analyze \
        -d "{\"persona_1\": \"$persona1\", \"persona_2\": \"$persona2\"}"
    fi
  done
done
```

### 3. Memory Performance Optimization
```bash
# Get memory insights
curl -X GET "http://localhost:5000/api/temporal/insights?user_id=optimize_user"

# Run optimization
curl -X POST http://localhost:5000/api/temporal/optimize \
  -d '{"user_id": "optimize_user", "optimization_type": "full_optimization"}'
```

## Best Practices

### Cross-Persona Memory Inference
- **Regular Analysis**: Run compatibility analysis after persona updates
- **Privacy Consideration**: Review bridge suggestions before implementation
- **Performance**: Monitor linkage graph complexity for large persona sets

### Cognitive Loop Debugging
- **Session Management**: Use focused debugging sessions for specific issues
- **Pattern Recognition**: Review decision patterns regularly for optimization
- **Confidence Monitoring**: Track confidence scores to identify improvement areas

### Temporal Memory Weighting
- **Regular Optimization**: Run memory optimization daily or weekly
- **Priority Management**: Adjust memory priorities based on usage patterns
- **Performance Monitoring**: Track memory retrieval efficiency metrics

## Troubleshooting

### Common Issues

#### Cross-Persona Analysis Fails
```bash
# Check system status
curl -X GET "http://localhost:5000/api/health"

# Verify persona data format
curl -X POST http://localhost:5000/api/cross-persona/analyze \
  -d '{"persona_1": "test", "persona_2": "test2", "persona_1_data": {}, "persona_2_data": {}}'
```

#### Cognitive Debugging Not Tracking
```bash
# Check if debugging is enabled
curl -X GET "http://localhost:5000/api/cognitive/decisions?user_id=test_user"

# Verify session is active
curl -X POST http://localhost:5000/api/cognitive/session \
  -d '{"user_id": "test_user"}'
```

#### Memory Optimization Slow
```bash
# Check memory statistics
curl -X GET "http://localhost:5000/api/temporal/insights?user_id=test_user"

# Run targeted optimization
curl -X POST http://localhost:5000/api/temporal/optimize \
  -d '{"user_id": "test_user", "optimization_type": "quick_optimization"}'
```

### Performance Tips
- Use pagination for large result sets
- Monitor API response times
- Implement caching for frequently accessed data
- Regular maintenance of enterprise feature databases

## Advanced Configuration

### Environment Variables
```bash
# Performance tuning
export CROSS_PERSONA_CACHE_SIZE=1000
export COGNITIVE_DEBUG_BUFFER_SIZE=500
export TEMPORAL_MEMORY_BATCH_SIZE=100

# Feature toggles
export ENABLE_REAL_TIME_INSIGHTS=true
export ENABLE_ADVANCED_ANALYTICS=true
export ENABLE_PREDICTIVE_OPTIMIZATION=true
```

### Database Optimization
```sql
-- Optimize cross-persona queries
CREATE INDEX idx_persona_linkages_user ON persona_linkages(user_id);
CREATE INDEX idx_memory_bridges_confidence ON memory_bridges(confidence);

-- Optimize cognitive debugging
CREATE INDEX idx_decisions_user_time ON decisions(user_id, timestamp);
CREATE INDEX idx_decisions_type ON decisions(decision_type);

-- Optimize temporal memory
CREATE INDEX idx_temporal_memories_user ON temporal_memories(user_id);
CREATE INDEX idx_temporal_memories_weight ON temporal_memories(temporal_weight);
```

This comprehensive usage guide provides everything needed to effectively use the enterprise features in the ML Query Router system.