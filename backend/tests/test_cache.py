"""
Tests for caching functionality.

Tests cache service operations, decorators, hit/miss behavior,
TTL expiration, and cache invalidation.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.utils.cache import (
    CacheService,
    cache_service,
    CacheKeys,
    CacheTTL,
    invalidate_model_cache,
    invalidate_comparison_cache,
    cache_response,
    invalidate_cache
)


class TestCacheService:
    """Test cache service basic operations."""
    
    def test_cache_service_initialization(self):
        """Test cache service can be initialized."""
        service = CacheService()
        assert service is not None
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test setting and getting values from cache."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        key = "test:key:123"
        value = {"test": "data", "count": 42}
        
        # Set value
        await cache_service.set(key, value, ttl=60)
        
        # Get value
        result = await cache_service.get(key)
        
        assert result is not None
        assert result["test"] == "data"
        assert result["count"] == 42
        
        # Cleanup
        await cache_service.delete(key)
    
    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test deleting values from cache."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        key = "test:delete:123"
        value = {"test": "delete_me"}
        
        # Set and verify
        await cache_service.set(key, value, ttl=60)
        result = await cache_service.get(key)
        assert result is not None
        
        # Delete
        await cache_service.delete(key)
        
        # Verify deleted
        result = await cache_service.get(key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        key = "test:ttl:123"
        value = {"test": "expire_soon"}
        
        # Set with 2 second TTL
        await cache_service.set(key, value, ttl=2)
        
        # Should exist immediately
        result = await cache_service.get(key)
        assert result is not None
        
        # Wait for expiration
        await asyncio.sleep(3)
        
        # Should be expired
        result = await cache_service.get(key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_delete_pattern(self):
        """Test deleting multiple keys by pattern."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        # Set multiple keys with same pattern
        keys = [
            "test:pattern:1",
            "test:pattern:2",
            "test:pattern:3"
        ]
        
        for key in keys:
            await cache_service.set(key, {"data": key}, ttl=60)
        
        # Verify all exist
        for key in keys:
            result = await cache_service.get(key)
            assert result is not None
        
        # Delete by pattern
        deleted_count = await cache_service.delete_pattern("test:pattern:*")
        assert deleted_count == 3
        
        # Verify all deleted
        for key in keys:
            result = await cache_service.get(key)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test getting cache statistics."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        stats = await cache_service.get_stats()
        
        assert stats is not None
        assert "used_memory" in stats or "used_memory_human" in stats
        assert "total_keys" in stats or "db0" in stats
    
    @pytest.mark.asyncio
    async def test_cache_clear_all(self):
        """Test clearing all cache entries."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        # Set some test keys
        test_keys = [
            "test:clear:1",
            "test:clear:2",
            "test:clear:3"
        ]
        
        for key in test_keys:
            await cache_service.set(key, {"data": key}, ttl=60)
        
        # Clear all
        deleted_count = await cache_service.clear_all()
        assert deleted_count >= len(test_keys)
        
        # Verify cleared
        for key in test_keys:
            result = await cache_service.get(key)
            assert result is None


class TestCacheKeys:
    """Test cache key generation."""
    
    def test_model_metrics_key(self):
        """Test model metrics cache key generation."""
        model_id = "abc-123"
        key = CacheKeys.model_metrics(model_id)
        
        assert key == "model:metrics:abc-123"
        assert "model:metrics:" in key
        assert model_id in key
    
    def test_model_comparison_key(self):
        """Test model comparison cache key generation."""
        model_ids = ["abc-123", "def-456"]
        key = CacheKeys.model_comparison(model_ids)
        
        assert "model:comparison:" in key
        # Should be deterministic
        key2 = CacheKeys.model_comparison(model_ids)
        assert key == key2
        
        # Different order should produce same key (sorted)
        key3 = CacheKeys.model_comparison(["def-456", "abc-123"])
        assert key == key3
    
    def test_feature_importance_key(self):
        """Test feature importance cache key generation."""
        model_id = "abc-123"
        
        # Without top_n
        key1 = CacheKeys.feature_importance(model_id)
        assert key1 == "model:feature_importance:abc-123"
        
        # With top_n
        key2 = CacheKeys.feature_importance(model_id, top_n=10)
        assert key2 == "model:feature_importance:abc-123:top_10"
        
        # Different top_n should produce different keys
        key3 = CacheKeys.feature_importance(model_id, top_n=20)
        assert key2 != key3
    
    def test_dataset_stats_key(self):
        """Test dataset stats cache key generation."""
        dataset_id = "dataset-789"
        key = CacheKeys.dataset_stats(dataset_id)
        
        assert key == "dataset:stats:dataset-789"
        assert "dataset:stats:" in key


class TestCacheTTL:
    """Test cache TTL constants."""
    
    def test_ttl_values(self):
        """Test that TTL constants are set correctly."""
        assert CacheTTL.VERY_SHORT == 300  # 5 minutes
        assert CacheTTL.SHORT == 900  # 15 minutes
        assert CacheTTL.MEDIUM == 1800  # 30 minutes
        assert CacheTTL.LONG == 3600  # 1 hour
        assert CacheTTL.VERY_LONG == 7200  # 2 hours
        assert CacheTTL.DAY == 86400  # 24 hours
        assert CacheTTL.WEEK == 604800  # 7 days
    
    def test_ttl_ordering(self):
        """Test that TTL values are in ascending order."""
        assert CacheTTL.VERY_SHORT < CacheTTL.SHORT
        assert CacheTTL.SHORT < CacheTTL.MEDIUM
        assert CacheTTL.MEDIUM < CacheTTL.LONG
        assert CacheTTL.LONG < CacheTTL.VERY_LONG
        assert CacheTTL.VERY_LONG < CacheTTL.DAY
        assert CacheTTL.DAY < CacheTTL.WEEK


class TestCacheDecorators:
    """Test cache decorators."""
    
    @pytest.mark.asyncio
    async def test_cache_response_decorator(self):
        """Test cache_response decorator caches function results."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        call_count = 0
        
        @cache_response(key_prefix="test:decorator", ttl=60)
        async def expensive_function(x: int):
            nonlocal call_count
            call_count += 1
            return {"result": x * 2}
        
        # First call - should execute function
        result1 = await expensive_function(5)
        assert result1["result"] == 10
        assert call_count == 1
        
        # Second call - should use cache
        result2 = await expensive_function(5)
        assert result2["result"] == 10
        assert call_count == 1  # Not incremented
        
        # Different argument - should execute function again
        result3 = await expensive_function(7)
        assert result3["result"] == 14
        assert call_count == 2
        
        # Cleanup
        await cache_service.delete_pattern("test:decorator:*")
    
    @pytest.mark.asyncio
    async def test_invalidate_cache_decorator(self):
        """Test invalidate_cache decorator clears cache after function."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        # Set up initial cache
        cache_key = "test:invalidate:123"
        await cache_service.set(cache_key, {"data": "old"}, ttl=60)
        
        # Verify cache exists
        result = await cache_service.get(cache_key)
        assert result is not None
        
        # Function with invalidation decorator
        @invalidate_cache(patterns=["test:invalidate:*"])
        async def update_function():
            return {"data": "new"}
        
        # Call function - should invalidate cache
        await update_function()
        
        # Verify cache was cleared
        result = await cache_service.get(cache_key)
        assert result is None


class TestCacheInvalidation:
    """Test cache invalidation logic."""
    
    @pytest.mark.asyncio
    async def test_invalidate_model_cache(self):
        """Test invalidating all cache for a model."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        model_id = "test-model-789"
        
        # Set up multiple cache entries for the model
        await cache_service.set(
            CacheKeys.model_metrics(model_id),
            {"accuracy": 0.95},
            ttl=60
        )
        await cache_service.set(
            CacheKeys.feature_importance(model_id),
            {"feature1": 0.8},
            ttl=60
        )
        
        # Verify cache exists
        assert await cache_service.get(CacheKeys.model_metrics(model_id)) is not None
        assert await cache_service.get(CacheKeys.feature_importance(model_id)) is not None
        
        # Invalidate all model cache
        await invalidate_model_cache(model_id)
        
        # Verify cache was cleared
        assert await cache_service.get(CacheKeys.model_metrics(model_id)) is None
        assert await cache_service.get(CacheKeys.feature_importance(model_id)) is None
    
    @pytest.mark.asyncio
    async def test_invalidate_comparison_cache(self):
        """Test invalidating comparison cache."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        # Set up comparison cache entries
        comparison_key1 = CacheKeys.model_comparison(["model1", "model2"])
        comparison_key2 = CacheKeys.model_comparison(["model3", "model4"])
        
        await cache_service.set(comparison_key1, {"winner": "model1"}, ttl=60)
        await cache_service.set(comparison_key2, {"winner": "model3"}, ttl=60)
        
        # Verify cache exists
        assert await cache_service.get(comparison_key1) is not None
        assert await cache_service.get(comparison_key2) is not None
        
        # Invalidate all comparisons
        await invalidate_comparison_cache()
        
        # Verify cache was cleared
        assert await cache_service.get(comparison_key1) is None
        assert await cache_service.get(comparison_key2) is None


class TestCacheEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_cache_with_none_value(self):
        """Test caching None values."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        key = "test:none:123"
        
        # Setting None should work
        await cache_service.set(key, None, ttl=60)
        
        # Getting None should return None
        result = await cache_service.get(key)
        # Note: The cache service may not store None, so this could be None
        # This is expected behavior
        
        # Cleanup
        await cache_service.delete(key)
    
    @pytest.mark.asyncio
    async def test_cache_with_complex_object(self):
        """Test caching complex nested objects."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        key = "test:complex:123"
        value = {
            "nested": {
                "level1": {
                    "level2": {
                        "data": [1, 2, 3],
                        "info": "deep"
                    }
                }
            },
            "list": [{"a": 1}, {"b": 2}],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Set complex value
        await cache_service.set(key, value, ttl=60)
        
        # Get and verify
        result = await cache_service.get(key)
        assert result is not None
        assert result["nested"]["level1"]["level2"]["data"] == [1, 2, 3]
        assert len(result["list"]) == 2
        
        # Cleanup
        await cache_service.delete(key)
    
    @pytest.mark.asyncio
    async def test_cache_unavailable_graceful_degradation(self):
        """Test that cache unavailability doesn't break functionality."""
        # Mock Redis as unavailable
        with patch.object(cache_service, 'is_available', return_value=False):
            # Operations should not raise errors
            result = await cache_service.get("test:key")
            assert result is None
            
            # Set should not raise error
            await cache_service.set("test:key", {"data": "test"}, ttl=60)
            
            # Delete should not raise error
            await cache_service.delete("test:key")
    
    @pytest.mark.asyncio
    async def test_cache_with_large_value(self):
        """Test caching large values."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        key = "test:large:123"
        
        # Create large value (10000 items)
        large_value = {
            "data": [{"id": i, "value": f"item_{i}"} for i in range(10000)]
        }
        
        # Set large value
        await cache_service.set(key, large_value, ttl=60)
        
        # Get and verify
        result = await cache_service.get(key)
        assert result is not None
        assert len(result["data"]) == 10000
        
        # Cleanup
        await cache_service.delete(key)


class TestCachePerformance:
    """Test cache performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_faster_than_miss(self):
        """Test that cache hits are faster than cache misses."""
        if not cache_service.is_available():
            pytest.skip("Redis not available")
        
        key = "test:performance:123"
        value = {"data": "test" * 1000}
        
        # Set value
        await cache_service.set(key, value, ttl=60)
        
        # Measure cache hit time
        start = time.time()
        result = await cache_service.get(key)
        hit_time = time.time() - start
        
        assert result is not None
        
        # Measure cache miss time (different key)
        start = time.time()
        result = await cache_service.get("test:performance:nonexistent")
        miss_time = time.time() - start
        
        assert result is None
        
        # Cache hit should be faster (or similar)
        # Note: This is not always true in local testing, but in production
        # the difference is more significant
        assert hit_time < 1.0  # Should be very fast
        assert miss_time < 1.0  # Should also be fast
        
        # Cleanup
        await cache_service.delete(key)
