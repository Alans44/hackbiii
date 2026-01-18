"""
MongoDB Atlas Integration for EcoShelf
=======================================
Cloud-native database for storing detection history and user preferences.

MLH Prize: Best Use of MongoDB Atlas

Features:
- Real-time detection logging
- User preference storage
- Analytics aggregation
- Time-series freshness data
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

# MongoDB Configuration
MONGODB_URI = os.environ.get(
    'MONGODB_URI', 
    'mongodb+srv://ecoshelf:password@cluster0.xxxxx.mongodb.net/ecoshelf?retryWrites=true&w=majority'
)
DATABASE_NAME = os.environ.get('MONGODB_DATABASE', 'ecoshelf')

# Try to import pymongo, provide fallback if not installed
try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("pymongo not installed. Run: pip install pymongo")


class MongoDBAtlas:
    """
    MongoDB Atlas client for EcoShelf.
    Stores detection history, user data, and analytics.
    """
    
    def __init__(self, uri: str = None):
        self.uri = uri or MONGODB_URI
        self.client = None
        self.db = None
        
        if MONGODB_AVAILABLE:
            try:
                self.client = MongoClient(self.uri)
                self.db = self.client[DATABASE_NAME]
                # Test connection
                self.client.admin.command('ping')
                print("✅ Connected to MongoDB Atlas")
            except Exception as e:
                print(f"⚠️ MongoDB connection failed: {e}")
                self.db = None
    
    # ============== COLLECTIONS ==============
    
    @property
    def detections(self) -> Optional[Collection]:
        """Collection for storing food detections"""
        return self.db['detections'] if self.db else None
    
    @property
    def users(self) -> Optional[Collection]:
        """Collection for user preferences"""
        return self.db['users'] if self.db else None
    
    @property
    def analytics(self) -> Optional[Collection]:
        """Collection for aggregated analytics"""
        return self.db['analytics'] if self.db else None
    
    @property
    def waste_log(self) -> Optional[Collection]:
        """Collection for tracking prevented waste"""
        return self.db['waste_log'] if self.db else None
    
    # ============== DETECTION OPERATIONS ==============
    
    def log_detection(self, detection: Dict[str, Any]) -> Optional[str]:
        """
        Log a food detection event.
        
        Args:
            detection: {
                'item': str,
                'freshness': int,
                'confidence': float,
                'category': str,
                'timestamp': datetime (optional)
            }
        
        Returns:
            Inserted document ID
        """
        if not self.detections:
            return None
        
        doc = {
            **detection,
            'timestamp': detection.get('timestamp', datetime.utcnow()),
            'created_at': datetime.utcnow()
        }
        
        result = self.detections.insert_one(doc)
        return str(result.inserted_id)
    
    def get_recent_detections(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        """Get detections from the last N hours"""
        if not self.detections:
            return []
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        cursor = self.detections.find(
            {'timestamp': {'$gte': cutoff}},
            {'_id': 0}
        ).sort('timestamp', -1).limit(limit)
        
        return list(cursor)
    
    def get_item_history(self, item: str, days: int = 30) -> List[Dict]:
        """Get freshness history for a specific item"""
        if not self.detections:
            return []
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        cursor = self.detections.find(
            {'item': item, 'timestamp': {'$gte': cutoff}},
            {'_id': 0, 'freshness': 1, 'timestamp': 1}
        ).sort('timestamp', 1)
        
        return list(cursor)
    
    # ============== USER OPERATIONS ==============
    
    def create_user(self, user_id: str, preferences: Dict = None) -> bool:
        """Create or update user preferences"""
        if not self.users:
            return False
        
        doc = {
            'user_id': user_id,
            'preferences': preferences or {},
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        self.users.update_one(
            {'user_id': user_id},
            {'$set': doc},
            upsert=True
        )
        return True
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        if not self.users:
            return None
        
        return self.users.find_one({'user_id': user_id}, {'_id': 0})
    
    def update_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user preferences"""
        if not self.users:
            return False
        
        self.users.update_one(
            {'user_id': user_id},
            {
                '$set': {
                    'preferences': preferences,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        return True
    
    # ============== ANALYTICS ==============
    
    def get_freshness_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get aggregated freshness statistics"""
        if not self.detections:
            return {}
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {'$match': {'timestamp': {'$gte': cutoff}}},
            {'$group': {
                '_id': '$item',
                'avg_freshness': {'$avg': '$freshness'},
                'min_freshness': {'$min': '$freshness'},
                'max_freshness': {'$max': '$freshness'},
                'count': {'$sum': 1}
            }},
            {'$sort': {'avg_freshness': 1}}  # Sort by lowest freshness first
        ]
        
        results = list(self.detections.aggregate(pipeline))
        return {
            'period_days': days,
            'items': results,
            'total_detections': sum(r['count'] for r in results)
        }
    
    def log_waste_prevented(self, item: str, weight_kg: float, action: str) -> Optional[str]:
        """
        Log when food waste was prevented.
        
        Args:
            item: Food item name
            weight_kg: Estimated weight
            action: What was done (e.g., "eaten", "composted", "donated")
        """
        if not self.waste_log:
            return None
        
        doc = {
            'item': item,
            'weight_kg': weight_kg,
            'action': action,
            'co2_saved_kg': weight_kg * 2.5,  # Rough estimate
            'timestamp': datetime.utcnow()
        }
        
        result = self.waste_log.insert_one(doc)
        return str(result.inserted_id)
    
    def get_impact_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get environmental impact metrics"""
        if not self.waste_log:
            return {}
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {'$match': {'timestamp': {'$gte': cutoff}}},
            {'$group': {
                '_id': None,
                'total_weight_kg': {'$sum': '$weight_kg'},
                'total_co2_saved_kg': {'$sum': '$co2_saved_kg'},
                'items_saved': {'$sum': 1}
            }}
        ]
        
        results = list(self.waste_log.aggregate(pipeline))
        if results:
            return {
                'period_days': days,
                **results[0],
                '_id': None
            }
        return {'period_days': days, 'total_weight_kg': 0, 'total_co2_saved_kg': 0, 'items_saved': 0}


# MongoDB Schema Definitions (for reference)
SCHEMAS = {
    "detections": {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["item", "freshness", "timestamp"],
                "properties": {
                    "item": {"bsonType": "string", "description": "Food item name"},
                    "freshness": {"bsonType": "int", "minimum": 0, "maximum": 100},
                    "confidence": {"bsonType": "double", "minimum": 0, "maximum": 1},
                    "category": {"enum": ["produce", "bottle", "snack", "protein", "prepared"]},
                    "timestamp": {"bsonType": "date"},
                    "user_id": {"bsonType": "string"}
                }
            }
        }
    },
    "users": {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["user_id"],
                "properties": {
                    "user_id": {"bsonType": "string"},
                    "email": {"bsonType": "string"},
                    "preferences": {
                        "bsonType": "object",
                        "properties": {
                            "voice_alerts": {"bsonType": "bool"},
                            "notification_threshold": {"bsonType": "int"},
                            "dietary_restrictions": {"bsonType": "array"}
                        }
                    }
                }
            }
        }
    }
}

# Index definitions for performance
INDEXES = {
    "detections": [
        {"keys": [("timestamp", -1)], "name": "timestamp_desc"},
        {"keys": [("item", 1), ("timestamp", -1)], "name": "item_timestamp"},
        {"keys": [("user_id", 1)], "name": "user_id", "sparse": True}
    ],
    "users": [
        {"keys": [("user_id", 1)], "name": "user_id_unique", "unique": True}
    ],
    "waste_log": [
        {"keys": [("timestamp", -1)], "name": "timestamp_desc"}
    ]
}


def setup_database():
    """Initialize database with schemas and indexes"""
    if not MONGODB_AVAILABLE:
        print("MongoDB not available. Install with: pip install pymongo")
        return
    
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    
    # Create collections with validation
    for collection_name, schema in SCHEMAS.items():
        try:
            db.create_collection(collection_name, **schema)
            print(f"✅ Created collection: {collection_name}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"ℹ️ Collection exists: {collection_name}")
            else:
                print(f"⚠️ Error creating {collection_name}: {e}")
    
    # Create indexes
    for collection_name, indexes in INDEXES.items():
        collection = db[collection_name]
        for index in indexes:
            try:
                collection.create_index(**index)
                print(f"✅ Created index: {index['name']} on {collection_name}")
            except Exception as e:
                print(f"⚠️ Error creating index: {e}")
    
    print("\n✅ Database setup complete!")


if __name__ == "__main__":
    print("=" * 60)
    print("MongoDB Atlas Configuration for EcoShelf")
    print("=" * 60)
    print("\nSetup Instructions:")
    print("1. Create free account at https://www.mongodb.com/atlas")
    print("2. Create a free M0 cluster (no credit card required)")
    print("3. Create database user and get connection string")
    print("4. Set MONGODB_URI environment variable")
    print("5. Run: pip install pymongo")
    print("\nTo initialize database schema:")
    print("  python mongodb_atlas.py --setup")
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--setup':
        setup_database()
    else:
        # Demo connection test
        mongo = MongoDBAtlas()
        if mongo.db:
            print("\n✅ Connection successful!")
            print(f"Database: {DATABASE_NAME}")
            print(f"Collections: {mongo.db.list_collection_names()}")
