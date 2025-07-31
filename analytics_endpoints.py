"""
Analytics API Endpoints for HackRX 6.0 Project
Comprehensive analytics and monitoring endpoints
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Optional
import logging

# Try to import enhanced database service
try:
    from database.enhanced_service import enhanced_db_service
    DB_AVAILABLE = enhanced_db_service is not None
except ImportError as e:
    print(f"⚠️ Enhanced database service not available: {e}")
    enhanced_db_service = None
    DB_AVAILABLE = False

# Try to import upload and interaction logging service
try:
    from database.upload_interaction_service import upload_interaction_logger
    UPLOAD_LOGGING_AVAILABLE = upload_interaction_logger is not None
except ImportError as e:
    print(f"⚠️ Upload and interaction logging not available: {e}")
    upload_interaction_logger = None
    UPLOAD_LOGGING_AVAILABLE = False

# Create analytics router
analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])

logger = logging.getLogger(__name__)

@analytics_router.get("/health")
async def get_system_health():
    """
    Get current system health status
    
    Returns comprehensive health information about all system components
    """
    try:
        if not DB_AVAILABLE:
            return {
                "overall_status": "degraded",
                "database_status": "unavailable",
                "api_status": "healthy",
                "message": "PostgreSQL analytics not available - API running in fallback mode",
                "fallback_mode": True
            }
        
        health_data = enhanced_db_service.get_system_health()
        return health_data
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@analytics_router.get("/system")
async def get_system_analytics(
    hours: int = Query(24, description="Number of hours to analyze", ge=1, le=720)
):
    """
    Get comprehensive system analytics for specified time period
    
    Provides detailed metrics about system performance, usage patterns,
    cost analysis, and error distribution.
    """
    try:
        if not DB_AVAILABLE:
            return {
                "message": "PostgreSQL analytics not available",
                "fallback_data": {
                    "time_period_hours": hours,
                    "status": "running_in_fallback_mode",
                    "note": "Install PostgreSQL dependencies for detailed analytics"
                }
            }
        
        analytics_data = enhanced_db_service.get_system_analytics(hours)
        return analytics_data
        
    except Exception as e:
        logger.error(f"❌ System analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@analytics_router.get("/document/{document_url:path}")
async def get_document_insights(document_url: str):
    """
    Get comprehensive insights about a specific document's processing history
    
    Provides detailed analysis of document processing patterns, performance trends,
    and historical statistics.
    """
    try:
        if not DB_AVAILABLE:
            return {
                "message": "PostgreSQL analytics not available",
                "document_url": document_url,
                "note": "Install PostgreSQL dependencies for document insights"
            }
        
        insights = enhanced_db_service.get_document_insights(document_url)
        return insights
        
    except Exception as e:
        logger.error(f"❌ Document insights failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document insights failed: {str(e)}")

@analytics_router.get("/questions")
async def get_question_analytics():
    """
    Get analytics about question patterns and answer quality
    
    Provides insights into question types, categories, processing patterns,
    and answer quality metrics.
    """
    try:
        if not DB_AVAILABLE:
            return {
                "message": "PostgreSQL analytics not available",
                "note": "Install PostgreSQL dependencies for question analytics"
            }
        
        question_data = enhanced_db_service.get_question_analytics()
        return question_data
        
    except Exception as e:
        logger.error(f"❌ Question analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question analytics failed: {str(e)}")

@analytics_router.get("/performance")
async def get_performance_metrics(
    hours: int = Query(6, description="Number of hours to analyze", ge=1, le=168)
):
    """
    Get focused performance metrics for system monitoring
    
    Provides key performance indicators for monitoring system health
    and identifying performance bottlenecks.
    """
    try:
        if not DB_AVAILABLE:
            return {
                "message": "PostgreSQL analytics not available",
                "performance_status": "monitoring_disabled",
                "note": "Install PostgreSQL dependencies for performance monitoring"
            }
        
        # Get system analytics and extract performance data
        analytics_data = enhanced_db_service.get_system_analytics(hours)
        
        # Extract key performance metrics
        performance_data = {
            "time_period_hours": hours,
            "request_performance": analytics_data.get("request_summary", {}),
            "processing_performance": analytics_data.get("performance_metrics", {}),
            "cost_performance": analytics_data.get("cost_analysis", {}),
            "error_performance": analytics_data.get("error_analysis", {}),
            "timestamp": analytics_data.get("analysis_period", {}).get("end")
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"❌ Performance metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")

@analytics_router.get("/costs")
async def get_cost_analysis(
    days: int = Query(7, description="Number of days to analyze", ge=1, le=90)
):
    """
    Get comprehensive cost analysis and usage tracking
    
    Provides detailed breakdown of API costs, token usage,
    and cost optimization recommendations.
    """
    try:
        if not DB_AVAILABLE:
            return {
                "message": "PostgreSQL analytics not available",
                "cost_tracking": "disabled",
                "note": "Install PostgreSQL dependencies for cost analysis"
            }
        
        hours = days * 24
        analytics_data = enhanced_db_service.get_system_analytics(hours)
        
        # Extract and enhance cost data
        cost_data = analytics_data.get("cost_analysis", {})
        request_data = analytics_data.get("request_summary", {})
        
        enhanced_cost_data = {
            "analysis_period_days": days,
            "cost_summary": cost_data,
            "usage_efficiency": {
                "cost_per_successful_request": cost_data.get("avg_cost_per_session", 0),
                "tokens_per_request": cost_data.get("total_tokens_used", 0) / max(request_data.get("total_requests", 1), 1),
                "success_rate_impact": request_data.get("success_rate", 0)
            },
            "optimization_recommendations": []
        }
        
        # Add optimization recommendations
        if cost_data.get("avg_cost_per_session", 0) > 0.10:
            enhanced_cost_data["optimization_recommendations"].append({
                "type": "high_cost_per_session",
                "message": "Consider optimizing prompt length or using more efficient models",
                "current_avg_cost": cost_data.get("avg_cost_per_session", 0)
            })
        
        if enhanced_cost_data["usage_efficiency"]["tokens_per_request"] > 3000:
            enhanced_cost_data["optimization_recommendations"].append({
                "type": "high_token_usage",
                "message": "High token usage detected - consider chunking optimization",
                "current_avg_tokens": enhanced_cost_data["usage_efficiency"]["tokens_per_request"]
            })
        
        return enhanced_cost_data
        
    except Exception as e:
        logger.error(f"❌ Cost analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost analysis failed: {str(e)}")

@analytics_router.get("/usage-patterns")
async def get_usage_patterns(
    days: int = Query(7, description="Number of days to analyze", ge=1, le=30)
):
    """
    Get usage patterns and document type analysis
    
    Provides insights into usage patterns, popular document types,
    and processing trends.
    """
    try:
        if not DB_AVAILABLE:
            return {
                "message": "PostgreSQL analytics not available",
                "usage_tracking": "disabled",
                "note": "Install PostgreSQL dependencies for usage pattern analysis"
            }
        
        hours = days * 24
        analytics_data = enhanced_db_service.get_system_analytics(hours)
        
        # Extract usage patterns
        usage_patterns = {
            "analysis_period_days": days,
            "document_patterns": analytics_data.get("document_distribution", {}),
            "processing_patterns": analytics_data.get("processing_modes", {}),
            "temporal_patterns": analytics_data.get("hourly_distribution", []),
            "success_patterns": {
                "overall_success_rate": analytics_data.get("request_summary", {}).get("success_rate", 0),
                "error_breakdown": analytics_data.get("error_analysis", {}).get("error_breakdown", {})
            }
        }
        
        # Add insights
        doc_types = usage_patterns["document_patterns"].get("types", {})
        if doc_types:
            most_popular = max(doc_types.items(), key=lambda x: x[1])
            usage_patterns["insights"] = {
                "most_popular_document_type": {
                    "type": most_popular[0],
                    "count": most_popular[1],
                    "percentage": (most_popular[1] / sum(doc_types.values())) * 100
                }
            }
        
        return usage_patterns
        
    except Exception as e:
        logger.error(f"❌ Usage patterns analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Usage patterns analysis failed: {str(e)}")

@analytics_router.get("/uploads")
async def get_upload_analytics(
    days: int = Query(7, description="Number of days to analyze", ge=1, le=90)
):
    """
    Get comprehensive document upload analytics
    
    Provides detailed analysis of document uploads, processing success rates,
    file type distribution, and storage statistics.
    """
    try:
        if not UPLOAD_LOGGING_AVAILABLE:
            return {
                "message": "Upload logging not available",
                "upload_analytics": "disabled",
                "note": "Install PostgreSQL dependencies for upload analytics"
            }
        
        upload_stats = upload_interaction_logger.get_upload_statistics(days)
        
        # Add insights and recommendations
        if "success_rate" in upload_stats:
            success_rate = upload_stats["success_rate"]
            upload_stats["insights"] = {
                "performance_status": "excellent" if success_rate > 95 else "good" if success_rate > 85 else "needs_improvement",
                "recommendations": []
            }
            
            if success_rate < 90:
                upload_stats["insights"]["recommendations"].append({
                    "type": "success_rate_improvement",
                    "message": f"Upload success rate is {success_rate:.1f}%. Consider investigating common failure patterns.",
                    "priority": "high" if success_rate < 80 else "medium"
                })
        
        return upload_stats
        
    except Exception as e:
        logger.error(f"❌ Upload analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload analytics failed: {str(e)}")

@analytics_router.get("/interactions")
async def get_interaction_analytics(
    days: int = Query(7, description="Number of days to analyze", ge=1, le=90)
):
    """
    Get comprehensive user interaction analytics
    
    Provides detailed analysis of user interactions, performance metrics,
    success rates, and user behavior patterns.
    """
    try:
        if not UPLOAD_LOGGING_AVAILABLE:
            return {
                "message": "Interaction logging not available",
                "interaction_analytics": "disabled",
                "note": "Install PostgreSQL dependencies for interaction analytics"
            }
        
        interaction_stats = upload_interaction_logger.get_interaction_statistics(days)
        
        # Add user engagement insights
        if "unique_users" in interaction_stats and "total_interactions" in interaction_stats:
            total_interactions = interaction_stats["total_interactions"]
            unique_users = interaction_stats["unique_users"]
            
            interaction_stats["user_engagement"] = {
                "avg_interactions_per_user": total_interactions / max(unique_users, 1),
                "user_retention": "high" if unique_users > 0 else "none",
                "engagement_level": "high" if total_interactions / max(unique_users, 1) > 5 else "moderate"
            }
        
        # Add performance insights
        if "performance" in interaction_stats:
            perf = interaction_stats["performance"]
            avg_time = perf.get("avg_processing_time_ms", 0)
            
            interaction_stats["performance_insights"] = {
                "response_speed": "fast" if avg_time < 2000 else "moderate" if avg_time < 5000 else "slow",
                "optimization_needed": avg_time > 3000,
                "user_experience": "excellent" if avg_time < 1500 else "good" if avg_time < 3000 else "needs_improvement"
            }
        
        return interaction_stats
        
    except Exception as e:
        logger.error(f"❌ Interaction analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interaction analytics failed: {str(e)}")

@analytics_router.get("/status")
async def get_analytics_status():
    """
    Get analytics system status and capabilities
    
    Returns information about available analytics features and system status.
    """
    return {
        "analytics_enabled": DB_AVAILABLE,
        "database_status": "available" if DB_AVAILABLE else "unavailable",
        "enhanced_features": {
            "system_health_monitoring": DB_AVAILABLE,
            "performance_analytics": DB_AVAILABLE,
            "cost_tracking": DB_AVAILABLE,
            "document_insights": DB_AVAILABLE,
            "question_analytics": DB_AVAILABLE,
            "usage_patterns": DB_AVAILABLE,
            "upload_logging": UPLOAD_LOGGING_AVAILABLE,
            "interaction_logging": UPLOAD_LOGGING_AVAILABLE,
            "upload_analytics": UPLOAD_LOGGING_AVAILABLE,
            "user_interaction_analytics": UPLOAD_LOGGING_AVAILABLE
        },
        "available_endpoints": [
            "/analytics/health",
            "/analytics/system",
            "/analytics/document/{document_url}",
            "/analytics/questions",
            "/analytics/performance",
            "/analytics/costs",
            "/analytics/usage-patterns",
            "/analytics/uploads",
            "/analytics/interactions",
            "/analytics/status"
        ],
        "fallback_mode": not DB_AVAILABLE,
        "note": "Install PostgreSQL dependencies (psycopg2-binary, sqlalchemy) for full analytics"
    }

# Export router for inclusion in main app
__all__ = ["analytics_router"]
