"""
Machine learning classifier for article relevance scoring
Lightweight model using scikit-learn for fast inference
"""
import logging
import pickle
from typing import Dict, Any, List, Optional, Tuple
try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    ML_AVAILABLE = True
except ImportError:
    np = None
    LogisticRegression = None
    RandomForestClassifier = None
    TfidfVectorizer = None
    Pipeline = None
    train_test_split = None
    classification_report = None
    confusion_matrix = None
    joblib = None
    ML_AVAILABLE = False
import os

logger = logging.getLogger(__name__)


class RelevanceClassifier:
    """ML classifier for determining article relevance to US-China trade"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.classifier_threshold = config.get("thresholds", {}).get("classifier_min", 0.6)
        
        # Model configuration
        self.model_type = config.get("classifier", {}).get("type", "logistic")
        self.model_path = config.get("classifier", {}).get("model_path", "models/relevance_classifier.pkl")
        
        # Feature configuration
        self.max_features = config.get("classifier", {}).get("max_features", 10000)
        self.ngram_range = config.get("classifier", {}).get("ngram_range", (1, 2))
        
        # Model and vectorizer
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        
        # Load or create model
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"Loaded classifier from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        
        # Create new model
        logger.info("Creating new classifier")
        self._create_model()
    
    def _create_model(self):
        """Create and train initial model with bootstrap data"""
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=None,  # Keep all words for multilingual
            lowercase=True,
            sublinear_tf=True
        )
        
        # Create classifier
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight="balanced"
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using logistic regression")
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        # Train with bootstrap data
        self._train_with_bootstrap_data()
    
    def _train_with_bootstrap_data(self):
        """Train model with initial bootstrap training data"""
        # Bootstrap training data (positive and negative examples)
        bootstrap_data = self._get_bootstrap_data()
        
        if len(bootstrap_data) < 10:
            logger.warning("Insufficient bootstrap data for training")
            return
        
        texts = [item["text"] for item in bootstrap_data]
        labels = [item["label"] for item in bootstrap_data]
        
        try:
            self.pipeline.fit(texts, labels)
            logger.info(f"Trained classifier with {len(bootstrap_data)} bootstrap samples")
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.pipeline, self.model_path)
            
        except Exception as e:
            logger.error(f"Failed to train classifier: {e}")
    
    def _get_bootstrap_data(self) -> List[Dict[str, Any]]:
        """Get bootstrap training data"""
        positive_examples = [
            # Chinese examples (relevant)
            {
                "text": "国务院关税税则委员会发布对美加征关税商品清单 根据相关法律法规，决定对美国进口商品加征关税",
                "label": 1
            },
            {
                "text": "商务部回应美方301关税措施 中方将采取必要反制措施维护自身合法权益",
                "label": 1
            },
            {
                "text": "海关总署发布关于暂停美国某企业产品进口的公告 即日起暂停进口",
                "label": 1
            },
            {
                "text": "中美经贸磋商取得积极进展 双方就贸易平衡等问题交换意见",
                "label": 1
            },
            {
                "text": "工信部发布不可靠实体清单 将对违法违规外国企业采取必要措施",
                "label": 1
            },
            
            # English examples (relevant)  
            {
                "text": "US announces new tariffs on Chinese imports USTR office released updated tariff list covering semiconductors steel",
                "label": 1
            },
            {
                "text": "China retaliates with countermeasures on US products Ministry of Commerce announced reciprocal tariff measures",
                "label": 1
            },
            {
                "text": "Trade war escalation affects global supply chains Businesses report disruptions due to ongoing tariff disputes",
                "label": 1
            },
            {
                "text": "WTO ruling on US China trade dispute Panel finds violations of international trade rules",
                "label": 1
            },
            {
                "text": "Export controls tighten on technology transfers New regulations restrict semiconductor equipment sales to China",
                "label": 1
            },
            
            # Negative examples (not relevant)
            {
                "text": "今日天气预报 明天多云转晴 气温15-25度 适合户外活动",
                "label": 0
            },
            {
                "text": "股市行情分析 上证指数今日收涨1.2% 科技股表现强劲",
                "label": 0
            },
            {
                "text": "城市交通管制通知 因道路施工 部分路段将实施交通管制",
                "label": 0
            },
            {
                "text": "Entertainment news celebrity wedding announcement Popular actor announces engagement to longtime partner",
                "label": 0
            },
            {
                "text": "Sports results basketball game recap Local team wins championship with outstanding performance",
                "label": 0
            },
            {
                "text": "Health tips for winter season Doctors recommend proper diet and exercise during cold weather",
                "label": 0
            },
            {
                "text": "Tourism promotion new attractions opened Theme park launches exciting rides and entertainment shows",
                "label": 0
            }
        ]
        
        return positive_examples
    
    def extract_features(self, article: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from article"""
        features = {}
        
        try:
            # Basic text features
            title = article.get("title", "")
            content = article.get("content", "")
            
            features["title_length"] = len(title)
            features["content_length"] = len(content)
            features["total_length"] = len(title + content)
            
            # Keyword features (if available)
            keyword_score = article.get("keyword_score", 0)
            features["keyword_score"] = keyword_score
            features["has_keywords"] = 1.0 if keyword_score > 0 else 0.0
            
            # Semantic features (if available)
            semantic_score = article.get("semantic_score", 0)
            features["semantic_score"] = semantic_score
            features["has_semantic_match"] = 1.0 if semantic_score > 0.5 else 0.0
            
            # Source features
            source_priority = article.get("source_priority", "medium")
            priority_mapping = {"high": 3.0, "medium": 2.0, "low": 1.0}
            features["source_priority"] = priority_mapping.get(source_priority, 2.0)
            
            # Language features
            language = article.get("language", "unknown")
            features["is_chinese"] = 1.0 if language == "zh" else 0.0
            features["is_english"] = 1.0 if language == "en" else 0.0
            
            # Metadata features
            metadata = article.get("metadata", {})
            features["is_government_doc"] = 1.0 if metadata.get("is_government_document", False) else 0.0
            features["is_urgent"] = 1.0 if metadata.get("is_urgent", False) else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return {}
    
    def classify_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Classify article relevance"""
        if not self.pipeline:
            return {
                "classifier_score": 0.0,
                "classifier_confidence": 0.0,
                "passes_classifier_filter": False,
                "features": {}
            }
        
        try:
            # Prepare text for classification
            title = article.get("title", "")
            content = article.get("content", "")
            text = f"{title} {content}"
            
            # Get prediction and probability
            prediction = self.pipeline.predict([text])[0]
            probabilities = self.pipeline.predict_proba([text])[0]
            
            # Get confidence (probability of positive class)
            confidence = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            
            # Extract additional features
            features = self.extract_features(article)
            
            return {
                "classifier_score": float(confidence),
                "classifier_prediction": int(prediction),
                "classifier_confidence": float(max(probabilities)),
                "passes_classifier_filter": confidence >= self.classifier_threshold,
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Failed to classify article: {e}")
            return {
                "classifier_score": 0.0,
                "classifier_confidence": 0.0,
                "passes_classifier_filter": False,
                "features": {}
            }
    
    def update_model(self, training_data: List[Tuple[Dict[str, Any], int]]):
        """Update model with new training data"""
        if not training_data or len(training_data) < 5:
            logger.warning("Insufficient training data for model update")
            return False
        
        try:
            # Prepare training data
            texts = []
            labels = []
            
            for article, label in training_data:
                title = article.get("title", "")
                content = article.get("content", "")
                text = f"{title} {content}"
                texts.append(text)
                labels.append(label)
            
            # Retrain model (incremental learning if supported)
            if hasattr(self.pipeline.named_steps['classifier'], 'partial_fit'):
                # Incremental learning
                X = self.pipeline.named_steps['vectorizer'].transform(texts)
                self.pipeline.named_steps['classifier'].partial_fit(X, labels)
            else:
                # Full retraining
                self.pipeline.fit(texts, labels)
            
            # Save updated model
            joblib.dump(self.pipeline, self.model_path)
            logger.info(f"Updated classifier with {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return False
    
    def evaluate_model(self, test_data: List[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
        """Evaluate model performance"""
        if not test_data or not self.pipeline:
            return {}
        
        try:
            # Prepare test data
            texts = []
            labels = []
            
            for article, label in test_data:
                title = article.get("title", "")
                content = article.get("content", "")
                text = f"{title} {content}"
                texts.append(text)
                labels.append(label)
            
            # Make predictions
            predictions = self.pipeline.predict(texts)
            probabilities = self.pipeline.predict_proba(texts)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted')
            recall = recall_score(labels, predictions, average='weighted')
            f1 = f1_score(labels, predictions, average='weighted')
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "num_samples": len(test_data),
                "classification_report": classification_report(labels, predictions, output_dict=True)
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the classifier"""
        if not self.pipeline:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_type": self.model_type,
            "model_path": self.model_path,
            "threshold": self.classifier_threshold
        }
        
        # Try to get model-specific info
        try:
            if hasattr(self.pipeline.named_steps['vectorizer'], 'vocabulary_'):
                vocab_size = len(self.pipeline.named_steps['vectorizer'].vocabulary_)
                info["vocabulary_size"] = vocab_size
            
            if hasattr(self.pipeline.named_steps['classifier'], 'n_features_in_'):
                info["n_features"] = self.pipeline.named_steps['classifier'].n_features_in_
                
        except Exception as e:
            logger.debug(f"Could not get detailed model info: {e}")
        
        return info


def classify_articles(articles: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Classify a list of articles"""
    classifier = RelevanceClassifier(config)
    
    for article in articles:
        try:
            classification = classifier.classify_article(article)
            article.update(classification)
        except Exception as e:
            logger.error(f"Failed to classify article: {e}")
            # Set default values
            article.update({
                "classifier_score": 0.0,
                "classifier_confidence": 0.0,
                "passes_classifier_filter": False,
                "features": {}
            })
    
    return articles


def main():
    """Test the classifier"""
    config = {
        "thresholds": {
            "classifier_min": 0.6
        },
        "classifier": {
            "type": "logistic",
            "max_features": 5000,
            "ngram_range": (1, 2)
        }
    }
    
    test_articles = [
        {
            "title": "国务院关税税则委员会发布对美加征关税商品清单",
            "content": "根据相关法律法规，决定对美国进口商品加征关税...",
            "language": "zh",
            "source_priority": "high",
            "keyword_score": 8.5,
            "semantic_score": 0.92
        },
        {
            "title": "今日天气预报",
            "content": "明天多云转晴，气温适宜...",
            "language": "zh",
            "source_priority": "low",
            "keyword_score": 0.0,
            "semantic_score": 0.1
        }
    ]
    
    classifier = RelevanceClassifier(config)
    
    print("Model info:")
    print(classifier.get_model_info())
    print()
    
    for i, article in enumerate(test_articles):
        print(f"Article {i+1}: {article['title'][:50]}...")
        result = classifier.classify_article(article)
        print(f"  Score: {result['classifier_score']:.3f}")
        print(f"  Prediction: {result['classifier_prediction']}")
        print(f"  Passes filter: {result['passes_classifier_filter']}")
        print()


if __name__ == "__main__":
    main()