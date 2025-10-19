"""
Vector store client for embeddings and semantic search
Using Qdrant for vector storage and similarity search
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant vector store client"""
    
    def __init__(self):
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION", "tariff_articles")
        
        self.client = QdrantClient(host=self.host, port=self.port)
        self.dimension = 1024  # For bge-m3 model
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure collection exists"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def add_article_embedding(
        self,
        article_id: int,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bool:
        """Add article embedding to vector store"""
        try:
            point = PointStruct(
                id=article_id,
                vector=embedding.tolist(),
                payload={
                    "article_id": article_id,
                    "title": metadata.get("title", ""),
                    "source": metadata.get("source", ""),
                    "language": metadata.get("language", "zh"),
                    "published_at": metadata.get("published_at"),
                    "tags": metadata.get("tags", []),
                    "score": metadata.get("score", 0.0)
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embedding for article {article_id}: {e}")
            return False
    
    def search_similar_articles(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search for similar articles"""
        try:
            # Build filter if provided
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter
            )
            
            return [
                (result.id, result.score, result.payload)
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to search similar articles: {e}")
            return []
    
    def check_similarity(
        self,
        article_embedding: np.ndarray,
        threshold: float = 0.95
    ) -> Optional[Tuple[int, float]]:
        """Check if article is similar to existing ones (for deduplication)"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=article_embedding.tolist(),
                limit=1,
                score_threshold=threshold
            )
            
            if results:
                result = results[0]
                return (result.id, result.score)
            return None
            
        except Exception as e:
            logger.error(f"Failed to check similarity: {e}")
            return None
    
    def update_article_metadata(
        self,
        article_id: int,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update article metadata without changing vector"""
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                points=[article_id],
                payload=metadata
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata for article {article_id}: {e}")
            return False
    
    def delete_article(self, article_id: int) -> bool:
        """Delete article from vector store"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[article_id]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete article {article_id}: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False


# Lazy initialization - don't create instance at module level
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get vector store instance (lazy initialization)"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance