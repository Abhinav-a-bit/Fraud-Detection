import os
import json
import redis

REDIS_URL = os.getenv("REDIS_URL","redis://redis:6379/0")

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None

def get_prediction(transaction_id: str):
    """Fetch a cached prediction if it exists."""
    if not redis_client: 
        return None
        
    data = redis_client.get(f"txn:{transaction_id}")
    if data:
        return json.loads(data)
    return None

def set_prediction(transaction_id: str, result: dict, ttl_seconds: int = 60):
    """Save a prediction to the cache for 60 seconds."""
    if not redis_client: 
        return

    # Ensure datetimes are converted to strings before saving to JSON
    if "processed_at" in result and not isinstance(result["processed_at"], str):
        result["processed_at"] = result["processed_at"].isoformat()
        
    # setex = Set with Expiration
    redis_client.setex(f"txn:{transaction_id}", ttl_seconds, json.dumps(result))