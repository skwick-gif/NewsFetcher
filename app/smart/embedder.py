"""
Embeddings generator and semantic similarity calculator
Uses multilingual models for Chinese-English text analysis
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import os
try:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError:
    np = None
    torch = None
    SentenceTransformer = None
    ML_AVAILABLE = False
from storage.vector import get_vector_store

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for semantic similarity analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("embeddings", {})
        self.model_name = self.config.get("model", "BAAI/bge-m3")
        self.device = self.config.get("device", "cpu")
        self.batch_size = self.config.get("batch_size", 32)
        
        # Similarity threshold from config
        self.similarity_threshold = config.get("thresholds", {}).get("semantic_min", 0.78)
        
        # Initialize model
        self.model = None
        self._load_model()
        
        # Reference embeddings for topic matching
        self.topic_embeddings = self._create_topic_embeddings()
        
        # Vector store for similarity search (lazy initialization)
        self._vector_store = None
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a smaller model
            try:
                logger.info("Falling back to multilingual-e5-small")
                self.model = SentenceTransformer("intfloat/multilingual-e5-small", device=self.device)
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def _create_topic_embeddings(self) -> Dict[str, np.ndarray]:
        """Create reference embeddings for key topics"""
        if not self.model:
            return {}
        
        # Topic descriptions in multiple languages
        topics = {
            "us_china_tariffs": [
                "US China trade war tariffs customs duties 301 investigation",
                "中美贸易战关税加征清单301调查",
                "美国对华关税措施反制贸易摩擦"
            ],
            "export_controls": [
                "export controls entity list technology transfer restrictions",
                "出口管制实体清单技术转让限制", 
                "科技出口管制不可靠实体名单"
            ],
            "trade_negotiations": [
                "trade negotiations agreement talks diplomatic economic",
                "贸易谈判协议磋商外交经济",
                "中美经贸谈判协商对话"
            ],
            "retaliation_measures": [
                "retaliation countermeasures response sanctions reciprocal",
                "反制措施对等回应制裁报复",
                "反击手段应对措施贸易反制"
            ]
        }
        
        topic_embeddings = {}
        
        try:
            for topic, descriptions in topics.items():
                # Combine descriptions and generate embedding
                combined_text = " ".join(descriptions)
                embedding = self.model.encode([combined_text])[0]
                topic_embeddings[topic] = embedding
                logger.debug(f"Created embedding for topic: {topic}")
            
            return topic_embeddings
            
        except Exception as e:
            logger.error(f"Failed to create topic embeddings: {e}")
            return {}
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text"""
        if not self.model or not text.strip():
            return None
        
        try:
            # Clean text
            cleaned_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode([cleaned_text])[0]
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple texts"""
        if not self.model or not texts:
            return [None] * len(texts)
        
        try:
            # Clean texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Filter out empty texts but keep track of positions
            text_mapping = {}
            valid_texts = []
            for i, text in enumerate(cleaned_texts):
                if text.strip():
                    text_mapping[len(valid_texts)] = i
                    valid_texts.append(text)
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(batch)
                embeddings.extend(batch_embeddings)
            
            # Map back to original positions
            result = [None] * len(texts)
            for emb_idx, text_idx in text_mapping.items():
                result[text_idx] = embeddings[emb_idx]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding"""
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Limit length to avoid memory issues (models typically have max length)
        max_length = 512  # Most models handle up to 512 tokens well
        if len(text) > max_length * 4:  # Rough estimate: 4 chars per token
            text = text[:max_length * 4]
        
        return text
    
    def calculate_semantic_score(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate semantic similarity scores against topic embeddings"""
        title = article.get("title", "")
        content = article.get("content", "")
        
        # Combine title and content (title weighted more heavily)
        text_for_embedding = f"{title} {title} {content}"[:2000]  # Limit length
        
        # Generate article embedding
        article_embedding = self.generate_embedding(text_for_embedding)
        
        if article_embedding is None:
            return {
                "semantic_score": 0.0,
                "topic_scores": {},
                "best_topic": None,
                "passes_semantic_filter": False
            }
        
        # Calculate similarity with each topic
        topic_scores = {}
        best_score = 0.0
        best_topic = None
        
        for topic, topic_embedding in self.topic_embeddings.items():
            try:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(article_embedding, topic_embedding)
                topic_scores[topic] = similarity
                
                if similarity > best_score:
                    best_score = similarity
                    best_topic = topic
                    
            except Exception as e:
                logger.error(f"Failed to calculate similarity for topic {topic}: {e}")
                topic_scores[topic] = 0.0
        
        # Store embedding in vector store for similarity search
        try:
            metadata = {
                "title": title[:200],
                "source": article.get("source_name", ""),
                "language": article.get("language", "unknown"),
                "published_at": str(article.get("published_at", "")),
                "score": best_score
            }
            
            # Use a temporary ID if article doesn't have one yet
            article_id = article.get("id", hash(title + str(article.get("url", ""))))
            self._get_vector_store().add_article_embedding(article_id, article_embedding, metadata)
            
        except Exception as e:
            logger.debug(f"Failed to store embedding in vector store: {e}")
        
        return {
            "semantic_score": best_score,
            "topic_scores": topic_scores,
            "best_topic": best_topic,
            "passes_semantic_filter": best_score >= self.similarity_threshold,
            "embedding": article_embedding.tolist()  # Store for later use
        }
    
    def _get_vector_store(self):
        """Get vector store instance (lazy initialization)"""
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Ensure result is in valid range
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def find_similar_articles(
        self, 
        article: Dict[str, Any], 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Find similar articles using vector search"""
        try:
            # Get article embedding
            embedding = article.get("embedding")
            if embedding is None:
                # Generate embedding if not available
                title = article.get("title", "")
                content = article.get("content", "")
                text = f"{title} {content}"[:2000]
                embedding = self.generate_embedding(text)
            
            if embedding is None:
                return []
            
            # Convert to numpy if it's a list
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # Search for similar articles
            results = self._get_vector_store().search_similar_articles(
                embedding, 
                limit=limit,
                score_threshold=score_threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar articles: {e}")
            return []
    
    def check_semantic_duplicate(self, article: Dict[str, Any], threshold: float = 0.95) -> Optional[Tuple[int, float]]:
        """Check if article is semantically similar to existing ones"""
        try:
            embedding = article.get("embedding")
            if embedding is None:
                title = article.get("title", "")
                content = article.get("content", "")
                text = f"{title} {content}"[:2000]
                embedding = self.generate_embedding(text)
            
            if embedding is None:
                return None
            
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            result = self._get_vector_store().check_similarity(embedding, threshold)
            return result
            
        except Exception as e:
            logger.error(f"Failed to check semantic duplicate: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.model:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "embedding_dimension": len(list(self.topic_embeddings.values())[0]) if self.topic_embeddings else 'unknown',
            "available_topics": list(self.topic_embeddings.keys())
        }


def analyze_semantic_similarity(articles: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze semantic similarity for a list of articles"""
    embedder = EmbeddingGenerator(config)
    
    for article in articles:
        try:
            semantic_analysis = embedder.calculate_semantic_score(article)
            article.update(semantic_analysis)
            
        except Exception as e:
            logger.error(f"Failed to analyze article semantically: {e}")
            # Set default values
            article.update({
                "semantic_score": 0.0,
                "topic_scores": {},
                "best_topic": None,
                "passes_semantic_filter": False
            })
    
    return articles


def main():
    """Test the embedding generator"""
    config = {
        "embeddings": {
            "model": "BAAI/bge-m3",
            "device": "cpu",
            "batch_size": 4
        },
        "thresholds": {
            "semantic_min": 0.7
        }
    }
    
    test_articles = [
        {
            "title": "国务院关税税则委员会发布对美加征关税商品清单",
            "content": "根据《中华人民共和国对外贸易法》等法律法规...",
            "language": "zh"
        },
        {
            "title": "US announces new tariff measures on Chinese semiconductors",
            "content": "The United States Trade Representative announced new tariff measures...",
            "language": "en"
        },
        {
            "title": "今日股市行情分析",
            "content": "上证指数今日开盘3200点...",
            "language": "zh"
        }
    ]
    
    embedder = EmbeddingGenerator(config)
    
    print("Model info:")
    print(embedder.get_model_info())
    print()
    
    for i, article in enumerate(test_articles):
        print(f"Article {i+1}: {article['title']}")
        analysis = embedder.calculate_semantic_score(article)
        print(f"  Semantic score: {analysis['semantic_score']:.3f}")
        print(f"  Best topic: {analysis['best_topic']}")
        print(f"  Passes filter: {analysis['passes_semantic_filter']}")
        print(f"  Topic scores: {analysis['topic_scores']}")
        print()


if __name__ == "__main__":
    main()