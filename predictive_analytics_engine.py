#!/usr/bin/env python3
"""
Predictive Analytics Engine with Machine Learning
Provides advanced predictive capabilities for query routing and system optimization
"""

import asyncio
import json
import logging
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Types of predictions"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    AGENT_LOAD = "agent_load"
    QUERY_COMPLEXITY = "query_complexity"
    USER_SATISFACTION = "user_satisfaction"
    SYSTEM_HEALTH = "system_health"
    COST_PREDICTION = "cost_prediction"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class PredictionResult:
    """Prediction result with confidence intervals"""
    prediction_type: PredictionType
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    model_used: str
    features_used: List[str]
    prediction_horizon: int  # minutes
    accuracy_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingData:
    """Training data for ML models"""
    features: List[Dict[str, Any]]
    targets: List[float]
    feature_names: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class FeatureExtractor:
    """Extract features from system data for ML models"""
    
    def __init__(self):
        self.feature_extractors = {
            PredictionType.RESPONSE_TIME: self._extract_response_time_features,
            PredictionType.THROUGHPUT: self._extract_throughput_features,
            PredictionType.ERROR_RATE: self._extract_error_rate_features,
            PredictionType.RESOURCE_USAGE: self._extract_resource_features,
            PredictionType.AGENT_LOAD: self._extract_agent_load_features
        }
        
    def extract_features(self, data: Dict[str, Any], prediction_type: PredictionType) -> Dict[str, Any]:
        """Extract features for specific prediction type"""
        if prediction_type in self.feature_extractors:
            return self.feature_extractors[prediction_type](data)
        else:
            return self._extract_generic_features(data)
            
    def _extract_response_time_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for response time prediction"""
        return {
            "query_length": len(data.get("query", "")),
            "query_complexity": data.get("complexity", 0.5),
            "agent_load": data.get("agent_load", 0.5),
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "concurrent_queries": data.get("concurrent_queries", 1),
            "agent_performance": data.get("agent_performance", 0.8),
            "cache_hit_rate": data.get("cache_hit_rate", 0.5),
            "query_category": hash(data.get("query_category", "unknown")) % 100,
            "historical_avg_response": data.get("historical_avg_response", 500),
            "system_cpu_usage": data.get("system_cpu_usage", 0.5),
            "system_memory_usage": data.get("system_memory_usage", 0.5)
        }
        
    def _extract_throughput_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for throughput prediction"""
        return {
            "active_agents": data.get("active_agents", 5),
            "average_query_complexity": data.get("average_query_complexity", 0.5),
            "system_load": data.get("system_load", 0.5),
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "historical_throughput": data.get("historical_throughput", 100),
            "cache_efficiency": data.get("cache_efficiency", 0.8),
            "network_latency": data.get("network_latency", 50),
            "concurrent_users": data.get("concurrent_users", 10),
            "query_types_diversity": data.get("query_types_diversity", 0.5),
            "agent_availability": data.get("agent_availability", 0.9)
        }
        
    def _extract_error_rate_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for error rate prediction"""
        return {
            "system_stress_level": data.get("system_stress_level", 0.3),
            "agent_failure_rate": data.get("agent_failure_rate", 0.05),
            "query_complexity_variance": data.get("query_complexity_variance", 0.2),
            "network_instability": data.get("network_instability", 0.1),
            "resource_contention": data.get("resource_contention", 0.2),
            "time_since_last_deployment": data.get("time_since_last_deployment", 24),
            "configuration_changes": data.get("configuration_changes", 0),
            "external_dependencies_health": data.get("external_dependencies_health", 0.95),
            "traffic_anomalies": data.get("traffic_anomalies", 0.0),
            "historical_error_rate": data.get("historical_error_rate", 0.02)
        }
        
    def _extract_resource_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for resource usage prediction"""
        return {
            "current_cpu_usage": data.get("current_cpu_usage", 0.5),
            "current_memory_usage": data.get("current_memory_usage", 0.5),
            "query_processing_rate": data.get("query_processing_rate", 50),
            "data_volume": data.get("data_volume", 1000),
            "concurrent_operations": data.get("concurrent_operations", 10),
            "time_of_day": datetime.now().hour,
            "historical_peak_usage": data.get("historical_peak_usage", 0.8),
            "scheduled_jobs": data.get("scheduled_jobs", 0),
            "background_tasks": data.get("background_tasks", 5),
            "cache_size": data.get("cache_size", 1000)
        }
        
    def _extract_agent_load_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for agent load prediction"""
        return {
            "agent_capacity": data.get("agent_capacity", 10),
            "current_assignments": data.get("current_assignments", 3),
            "agent_response_time": data.get("agent_response_time", 200),
            "agent_success_rate": data.get("agent_success_rate", 0.95),
            "query_queue_length": data.get("query_queue_length", 2),
            "agent_specialization_match": data.get("agent_specialization_match", 0.8),
            "historical_load_pattern": data.get("historical_load_pattern", 0.5),
            "agent_health_score": data.get("agent_health_score", 0.9),
            "time_since_last_task": data.get("time_since_last_task", 30),
            "priority_queue_length": data.get("priority_queue_length", 1)
        }
        
    def _extract_generic_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract generic features"""
        return {
            "timestamp": time.time(),
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "system_load": data.get("system_load", 0.5),
            "active_users": data.get("active_users", 10),
            "query_rate": data.get("query_rate", 5.0)
        }


class MLModelManager:
    """Manage ML models for predictions"""
    
    def __init__(self, model_dir: str = "./ml_models"):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Model configurations
        if ML_AVAILABLE:
            self.model_configs = {
                "random_forest": {
                    "class": RandomForestRegressor,
                    "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
                },
                "gradient_boosting": {
                    "class": GradientBoostingRegressor,
                    "params": {"n_estimators": 100, "random_state": 42}
                },
                "linear_regression": {
                    "class": LinearRegression,
                    "params": {}
                },
                "ridge_regression": {
                    "class": Ridge,
                    "params": {"alpha": 1.0, "random_state": 42}
                }
            }
        else:
            self.model_configs = {}
        
    def train_model(self, prediction_type: PredictionType, training_data: TrainingData, 
                   model_type: str = "random_forest") -> Dict[str, Any]:
        """Train a model for specific prediction type"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, cannot train models")
            return {"error": "ML libraries not available"}
            
        try:
            # Prepare data
            if ML_AVAILABLE:
                X = pd.DataFrame(training_data.features)
                y = np.array(training_data.targets)
            else:
                # Use numpy for basic operations
                X = np.array([list(feat.values()) for feat in training_data.features])
                y = np.array(training_data.targets)
            
            # Handle missing values
            if ML_AVAILABLE:
                X = X.fillna(X.mean())
            else:
                # Simple NaN handling for numpy arrays
                X = np.nan_to_num(X)
            
            # Split data
            if ML_AVAILABLE:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                # Simple train/test split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            if ML_AVAILABLE:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                scaler = None
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            if model_type not in self.model_configs:
                raise ValueError(f"Model type {model_type} not available")
                
            model_config = self.model_configs[model_type]
            model = model_config["class"](**model_config["params"])
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            if ML_AVAILABLE:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            else:
                # Simple MSE calculation
                mse = np.mean((y_test - y_pred) ** 2)
                r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            
            # Store model and scaler
            model_key = f"{prediction_type.value}_{model_type}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_names[model_key] = list(X.columns) if ML_AVAILABLE else training_data.feature_names
            
            # Store performance metrics
            self.model_performance[model_key] = {
                "mse": mse,
                "r2_score": r2,
                "rmse": np.sqrt(mse),
                "training_samples": len(X_train),
                "features_count": len(X.columns),
                "trained_at": datetime.now().isoformat()
            }
            
            # Save model to disk
            self._save_model(model_key, model, scaler)
            
            logger.info(f"Model {model_key} trained successfully. R² score: {r2:.3f}")
            
            return {
                "model_key": model_key,
                "performance": self.model_performance[model_key],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {"error": str(e), "success": False}
            
    def predict(self, prediction_type: PredictionType, features: Dict[str, Any], 
               model_type: str = "random_forest") -> Optional[PredictionResult]:
        """Make prediction using trained model"""
        if not ML_AVAILABLE:
            return None
            
        model_key = f"{prediction_type.value}_{model_type}"
        
        if model_key not in self.models:
            logger.warning(f"Model {model_key} not found")
            return None
            
        try:
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            feature_names = self.feature_names[model_key]
            
            # Prepare features
            feature_vector = []
            for feature_name in feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
                
            # Scale features
            X = scaler.transform([feature_vector])
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Calculate confidence interval (approximation)
            if hasattr(model, 'predict') and hasattr(model, 'estimators_'):
                # For ensemble methods, use prediction variance
                predictions = [tree.predict(X)[0] for tree in model.estimators_[:min(50, len(model.estimators_))]]
                std_dev = np.std(predictions)
                confidence_interval = (prediction - 1.96 * std_dev, prediction + 1.96 * std_dev)
                confidence_score = 1.0 - min(std_dev / abs(prediction), 1.0) if prediction != 0 else 0.5
            else:
                # Default confidence interval
                confidence_interval = (prediction * 0.8, prediction * 1.2)
                confidence_score = 0.7
                
            # Get model performance
            performance = self.model_performance.get(model_key, {})
            accuracy_score = performance.get("r2_score", 0.5)
            
            return PredictionResult(
                prediction_type=prediction_type,
                predicted_value=prediction,
                confidence_interval=confidence_interval,
                confidence_score=confidence_score,
                model_used=model_key,
                features_used=feature_names,
                prediction_horizon=30,  # 30 minutes default
                accuracy_score=accuracy_score,
                metadata={
                    "model_performance": performance,
                    "input_features": features
                }
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
            
    def _save_model(self, model_key: str, model: Any, scaler: Any):
        """Save model and scaler to disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_key}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_key}_scaler.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
                
            logger.info(f"Model {model_key} saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
    def load_model(self, model_key: str) -> bool:
        """Load model from disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_key}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_key}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    self.models[model_key] = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scalers[model_key] = pickle.load(f)
                    
                logger.info(f"Model {model_key} loaded from disk")
                return True
            else:
                logger.warning(f"Model files not found for {model_key}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        return {
            "available_models": list(self.models.keys()),
            "model_performance": self.model_performance,
            "feature_names": self.feature_names,
            "model_configs": self.model_configs
        }


class PredictiveAnalyticsEngine:
    """Main predictive analytics engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_extractor = FeatureExtractor()
        self.model_manager = MLModelManager()
        self.training_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.prediction_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, Dict] = defaultdict(dict)
        
        # Auto-training configuration
        self.auto_train_threshold = 100  # Minimum samples for auto-training
        self.auto_train_interval = 3600  # Auto-train every hour
        self.last_auto_train = {}
        
        # Initialize with pre-existing models
        self._load_existing_models()
        
    def _load_existing_models(self):
        """Load existing models from disk"""
        if not ML_AVAILABLE:
            return
            
        for prediction_type in PredictionType:
            for model_type in self.model_manager.model_configs:
                model_key = f"{prediction_type.value}_{model_type}"
                self.model_manager.load_model(model_key)
                
    def add_training_data(self, prediction_type: PredictionType, features: Dict[str, Any], 
                         target: float):
        """Add training data point"""
        self.training_data[prediction_type.value].append({
            "features": features,
            "target": target,
            "timestamp": datetime.now()
        })
        
        # Auto-train if threshold reached
        if len(self.training_data[prediction_type.value]) >= self.auto_train_threshold:
            self._check_auto_train(prediction_type)
            
    def _check_auto_train(self, prediction_type: PredictionType):
        """Check if auto-training should be triggered"""
        last_train = self.last_auto_train.get(prediction_type.value, 0)
        if time.time() - last_train > self.auto_train_interval:
            asyncio.create_task(self._auto_train_model(prediction_type))
            
    async def _auto_train_model(self, prediction_type: PredictionType):
        """Auto-train model with accumulated data"""
        try:
            data_points = list(self.training_data[prediction_type.value])
            if len(data_points) < self.auto_train_threshold:
                return
                
            # Prepare training data
            features = [point["features"] for point in data_points]
            targets = [point["target"] for point in data_points]
            feature_names = list(features[0].keys()) if features else []
            
            training_data = TrainingData(
                features=features,
                targets=targets,
                feature_names=feature_names
            )
            
            # Train model
            result = self.model_manager.train_model(prediction_type, training_data)
            
            if result.get("success"):
                self.last_auto_train[prediction_type.value] = time.time()
                logger.info(f"Auto-trained model for {prediction_type.value}")
            else:
                logger.error(f"Auto-training failed for {prediction_type.value}: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error in auto-training: {e}")
            
    async def predict(self, prediction_type: PredictionType, context: Dict[str, Any], 
                     horizon_minutes: int = 30) -> Optional[PredictionResult]:
        """Make prediction for given context"""
        # Extract features
        features = self.feature_extractor.extract_features(context, prediction_type)
        
        # Make prediction
        result = self.model_manager.predict(prediction_type, features)
        
        if result:
            result.prediction_horizon = horizon_minutes
            self.prediction_history.append({
                "prediction_type": prediction_type.value,
                "predicted_value": result.predicted_value,
                "confidence_score": result.confidence_score,
                "timestamp": datetime.now(),
                "context": context
            })
            
        return result
        
    async def batch_predict(self, predictions: List[Tuple[PredictionType, Dict[str, Any]]]) -> List[Optional[PredictionResult]]:
        """Make batch predictions"""
        tasks = []
        for prediction_type, context in predictions:
            task = asyncio.create_task(self.predict(prediction_type, context))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return results
        
    def get_prediction_insights(self, prediction_type: PredictionType, 
                              hours_back: int = 24) -> Dict[str, Any]:
        """Get insights about predictions"""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        
        relevant_predictions = [
            p for p in self.prediction_history 
            if p["timestamp"] > cutoff and p["prediction_type"] == prediction_type.value
        ]
        
        if not relevant_predictions:
            return {"message": "No recent predictions"}
            
        values = [p["predicted_value"] for p in relevant_predictions]
        confidences = [p["confidence_score"] for p in relevant_predictions]
        
        return {
            "prediction_count": len(relevant_predictions),
            "value_statistics": {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "std": np.std(values) if len(values) > 1 else 0
            },
            "confidence_statistics": {
                "min": min(confidences),
                "max": max(confidences),
                "mean": sum(confidences) / len(confidences),
                "std": np.std(confidences) if len(confidences) > 1 else 0
            },
            "trend_analysis": self._analyze_trend(values),
            "accuracy_assessment": self._assess_accuracy(prediction_type, relevant_predictions)
        }
        
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in predicted values"""
        if len(values) < 2:
            return {"trend": "insufficient_data"}
            
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            trend = "increasing"
        elif slope < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"
            
        return {
            "trend": trend,
            "slope": slope,
            "volatility": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        }
        
    def _assess_accuracy(self, prediction_type: PredictionType, 
                        predictions: List[Dict]) -> Dict[str, Any]:
        """Assess prediction accuracy (placeholder - would need actual values)"""
        # This would compare predictions with actual observed values
        # For now, return placeholder accuracy metrics
        return {
            "estimated_accuracy": 0.85,
            "confidence_correlation": 0.75,
            "prediction_reliability": "good",
            "note": "Accuracy assessment requires actual observed values"
        }
        
    def generate_prediction_report(self, prediction_types: List[PredictionType] = None, 
                                 hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive prediction report"""
        if prediction_types is None:
            prediction_types = list(PredictionType)
            
        report = {
            "generated_at": datetime.now().isoformat(),
            "time_window_hours": hours_back,
            "prediction_types_analyzed": len(prediction_types),
            "insights": {},
            "model_performance": self.model_manager.get_model_info(),
            "recommendations": []
        }
        
        for prediction_type in prediction_types:
            insights = self.get_prediction_insights(prediction_type, hours_back)
            report["insights"][prediction_type.value] = insights
            
            # Generate recommendations based on insights
            recommendations = self._generate_recommendations(prediction_type, insights)
            report["recommendations"].extend(recommendations)
            
        return report
        
    def _generate_recommendations(self, prediction_type: PredictionType, 
                                insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on prediction insights"""
        recommendations = []
        
        if "trend_analysis" in insights:
            trend = insights["trend_analysis"].get("trend")
            
            if trend == "increasing":
                if prediction_type in [PredictionType.RESPONSE_TIME, PredictionType.ERROR_RATE]:
                    recommendations.append({
                        "type": "performance_warning",
                        "prediction_type": prediction_type.value,
                        "title": f"Increasing {prediction_type.value} trend detected",
                        "description": "Consider investigating system performance",
                        "priority": "high"
                    })
                    
            elif trend == "decreasing":
                if prediction_type in [PredictionType.THROUGHPUT, PredictionType.USER_SATISFACTION]:
                    recommendations.append({
                        "type": "performance_degradation",
                        "prediction_type": prediction_type.value,
                        "title": f"Decreasing {prediction_type.value} trend detected",
                        "description": "System performance may be degrading",
                        "priority": "medium"
                    })
                    
        # Check confidence levels
        if "confidence_statistics" in insights:
            avg_confidence = insights["confidence_statistics"].get("mean", 0)
            if avg_confidence < 0.6:
                recommendations.append({
                    "type": "prediction_reliability",
                    "prediction_type": prediction_type.value,
                    "title": f"Low prediction confidence for {prediction_type.value}",
                    "description": "Consider collecting more training data",
                    "priority": "medium"
                })
                
        return recommendations
        
    def get_system_health_prediction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive system health prediction"""
        health_predictions = {}
        
        # Key system metrics to predict
        key_metrics = [
            PredictionType.RESPONSE_TIME,
            PredictionType.THROUGHPUT,
            PredictionType.ERROR_RATE,
            PredictionType.RESOURCE_USAGE
        ]
        
        overall_health_score = 1.0
        
        for metric in key_metrics:
            prediction = self.model_manager.predict(metric, 
                self.feature_extractor.extract_features(context, metric))
                
            if prediction:
                health_predictions[metric.value] = {
                    "predicted_value": prediction.predicted_value,
                    "confidence": prediction.confidence_score,
                    "trend": "stable"  # Would analyze trend from recent predictions
                }
                
                # Adjust overall health score based on predictions
                if metric == PredictionType.RESPONSE_TIME and prediction.predicted_value > 1000:
                    overall_health_score *= 0.8
                elif metric == PredictionType.ERROR_RATE and prediction.predicted_value > 0.05:
                    overall_health_score *= 0.7
                elif metric == PredictionType.RESOURCE_USAGE and prediction.predicted_value > 0.9:
                    overall_health_score *= 0.6
                    
        # Determine overall health status
        if overall_health_score >= 0.9:
            health_status = "excellent"
        elif overall_health_score >= 0.7:
            health_status = "good"
        elif overall_health_score >= 0.5:
            health_status = "fair"
        else:
            health_status = "poor"
            
        return {
            "overall_health_score": overall_health_score,
            "health_status": health_status,
            "predictions": health_predictions,
            "timestamp": datetime.now().isoformat(),
            "prediction_horizon": "30 minutes"
        }
        
    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics engine statistics"""
        return {
            "ml_available": ML_AVAILABLE,
            "trained_models": len(self.model_manager.models),
            "training_data_points": {
                pred_type: len(data) for pred_type, data in self.training_data.items()
            },
            "prediction_history_size": len(self.prediction_history),
            "auto_train_threshold": self.auto_train_threshold,
            "last_auto_train": self.last_auto_train,
            "available_prediction_types": [pt.value for pt in PredictionType],
            "model_performance": self.model_manager.model_performance
        }