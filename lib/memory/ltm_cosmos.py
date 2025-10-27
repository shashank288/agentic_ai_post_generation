"""
Long-Term Memory (LTM) storage using Azure Cosmos DB.

This module manages user preferences and platform defaults in Cosmos DB.
Stores information like tone, platform preferences, and recent successful hashtags.

Falls back to in-memory storage if Cosmos credentials are not configured.
"""

import os
from typing import Dict, Optional
import warnings

try:
    from azure.cosmos import CosmosClient, PartitionKey
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False
    warnings.warn("azure-cosmos not installed. Using in-memory LTM storage.")


class LongTermMemory:
    """
    Long-term memory for user preferences and platform defaults.
    
    Stores user-specific information in Cosmos DB:
    - Preferred tone (insightful, casual, professional, etc.)
    - Platform defaults (LinkedIn/Twitter preferences)
    - Recent successful hashtags
    - Writing style preferences
    
    Partition Key: /user_id (for efficient single-user queries)
    
    Attributes:
        client: Cosmos DB client
        database: Cosmos database instance
        container: Cosmos container for LTM
        use_cosmos: Whether Cosmos DB is configured
        memory_store: In-memory fallback dictionary
    """
    
    def __init__(self):
        """
        Initialize LTM storage.
        
        Attempts to connect to Cosmos DB using environment variables.
        Falls back to in-memory storage if Cosmos is not configured.
        """
        self.use_cosmos = False
        self.memory_store: Dict[str, dict] = {}  # In-memory fallback
        
        # Try to initialize Cosmos DB
        if COSMOS_AVAILABLE:
            endpoint = os.getenv("COSMOS_ENDPOINT")
            key = os.getenv("COSMOS_KEY")
            database_name = os.getenv("COSMOS_DATABASE_NAME", "agentic_post_gen")
            container_name = os.getenv("COSMOS_LTM_CONTAINER", "ltm")
            
            if endpoint and key:
                try:
                    self.client = CosmosClient(endpoint, key)
                    self.database = self.client.get_database_client(database_name)
                    self.container = self.database.get_container_client(container_name)
                    self.use_cosmos = True
                    print(f"LTM connected to Cosmos DB: {database_name}/{container_name}")
                except Exception as e:
                    print(f"Failed to connect to Cosmos DB: {e}")
                    print("   Using in-memory LTM storage instead.")
            else:
                print("Cosmos DB credentials not found. Using in-memory LTM storage.")
        else:
            print("azure-cosmos package not available. Using in-memory LTM storage.")
    
    def get_user_preferences(self, user_id: str) -> Optional[dict]:
        """
        Retrieve user preferences from LTM.
        
        Args:
            user_id (str): User identifier
            
        Returns:
            Optional[dict]: User preferences or None if not found
            
        Example return value:
        {
            "user_id": "user-123",
            "preferred_tone": "insightful",
            "platform_defaults": {
                "linkedin": {"length": "~150w"},
                "twitter": {"length": "~280c"}
            },
            "last_hashtags": ["#AI", "#NLP", "#MachineLearning"]
        }
        """
        if self.use_cosmos:
            try:
                # Query with partition key for efficiency
                item = self.container.read_item(
                    item=user_id,
                    partition_key=user_id
                )
                return item
            except CosmosResourceNotFoundError:
                return None
            except Exception as e:
                print(f"Error reading from Cosmos DB: {e}")
                return None
        else:
            # In-memory fallback
            return self.memory_store.get(user_id)
    
    def upsert_user_preferences(self, user_id: str, preferences: dict) -> bool:
        """
        Create or update user preferences in LTM.
        
        Args:
            user_id (str): User identifier
            preferences (dict): User preferences to store
            
        Returns:
            bool: True if successful, False otherwise
            
        Example usage:
            ltm.upsert_user_preferences("user-123", {
                "user_id": "user-123",
                "preferred_tone": "insightful",
                "platform_defaults": {"linkedin": {"length": "~150w"}},
                "last_hashtags": ["#AI", "#NLP"]
            })
        """
        # Ensure user_id is in the preferences
        preferences["user_id"] = user_id
        preferences["id"] = user_id  # Cosmos DB requires 'id' field
        
        if self.use_cosmos:
            try:
                self.container.upsert_item(preferences)
                return True
            except Exception as e:
                print(f"Error writing to Cosmos DB: {e}")
                return False
        else:
            # In-memory fallback
            self.memory_store[user_id] = preferences
            return True
    
    def update_tone(self, user_id: str, tone: str) -> bool:
        """
        Update user's preferred tone.
        
        Args:
            user_id (str): User identifier
            tone (str): New tone (e.g., "insightful", "casual", "professional")
            
        Returns:
            bool: True if successful, False otherwise
        """
        prefs = self.get_user_preferences(user_id) or {"user_id": user_id}
        prefs["preferred_tone"] = tone
        return self.upsert_user_preferences(user_id, prefs)
    
    def add_hashtags(self, user_id: str, hashtags: list) -> bool:
        """
        Add successful hashtags to user's history.
        
        Args:
            user_id (str): User identifier
            hashtags (list): List of hashtag strings
            
        Returns:
            bool: True if successful, False otherwise
        """
        prefs = self.get_user_preferences(user_id) or {"user_id": user_id}
        
        # Get existing hashtags or initialize empty list
        existing = prefs.get("last_hashtags", [])
        
        # Add new hashtags (avoid duplicates)
        for tag in hashtags:
            if tag not in existing:
                existing.append(tag)
        
        # Keep only last 20 hashtags
        prefs["last_hashtags"] = existing[-20:]
        
        return self.upsert_user_preferences(user_id, prefs)
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete all user data (for GDPR compliance).
        
        Args:
            user_id (str): User identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.use_cosmos:
            try:
                self.container.delete_item(
                    item=user_id,
                    partition_key=user_id
                )
                return True
            except CosmosResourceNotFoundError:
                return True  # Already deleted
            except Exception as e:
                print(f"Error deleting from Cosmos DB: {e}")
                return False
        else:
            # In-memory fallback
            if user_id in self.memory_store:
                del self.memory_store[user_id]
            return True


def create_ltm_container():
    """
    Utility function to create the LTM container in Cosmos DB.
    
    Run this once during setup to create the required container.
    
    Container configuration:
    - Partition Key: /user_id
    - No TTL (data persists until manually deleted)
    """
    if not COSMOS_AVAILABLE:
        print("azure-cosmos package not installed")
        return
    
    endpoint = os.getenv("COSMOS_ENDPOINT")
    key = os.getenv("COSMOS_KEY")
    database_name = os.getenv("COSMOS_DATABASE_NAME", "agentic_post_gen")
    container_name = os.getenv("COSMOS_LTM_CONTAINER", "ltm")
    
    if not endpoint or not key:
        print("Cosmos DB credentials not configured")
        return
    
    try:
        client = CosmosClient(endpoint, key)
        database = client.create_database_if_not_exists(database_name)
        
        # Create container with /user_id partition key
        container = database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/user_id"),
        )
        
        print(f"LTM container created: {database_name}/{container_name}")
        print(f"   Partition Key: /user_id")
    except Exception as e:
        print(f"Error creating LTM container: {e}")


if __name__ == "__main__":
    # Test/setup script
    print("Setting up LTM container...")
    create_ltm_container()
