"""
Simple script to run stock_predictions migration
"""
import sys
sys.path.insert(0, 'app')

from storage.db import get_db_engine

# Get database engine
engine = get_db_engine()

# Create stock_predictions table using raw SQL
print("🔧 Creating stock_predictions table...")

with engine.connect() as conn:
    # Check if table already exists
    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_predictions'"
    )
    
    if result.fetchone():
        print("✅ stock_predictions table already exists")
    else:
        # Create the table
        conn.execute("""
            CREATE TABLE stock_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                -- Stock identification
                symbol VARCHAR(20) NOT NULL,
                company_name VARCHAR(255),
                
                -- Prediction details
                prediction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                prediction_source VARCHAR(50) NOT NULL,
                price_at_prediction FLOAT NOT NULL,
                
                -- Prediction parameters
                predicted_direction VARCHAR(10),
                confidence_score FLOAT,
                timeframe VARCHAR(20),
                target_price FLOAT,
                expected_return FLOAT,
                
                -- Reasoning
                reason TEXT,
                sector VARCHAR(100),
                keywords_matched JSON,
                news_sentiment FLOAT,
                
                -- ML scores
                ml_score FLOAT,
                technical_score FLOAT,
                fundamental_score FLOAT,
                
                -- Related article
                article_id INTEGER,
                
                -- Outcome tracking
                actual_price_1d FLOAT,
                actual_price_1w FLOAT,
                actual_price_1m FLOAT,
                actual_price_3m FLOAT,
                
                actual_return_1d FLOAT,
                actual_return_1w FLOAT,
                actual_return_1m FLOAT,
                actual_return_3m FLOAT,
                
                -- Performance evaluation
                prediction_accuracy FLOAT,
                was_correct BOOLEAN,
                max_gain_achieved FLOAT,
                max_loss_suffered FLOAT,
                
                -- Status tracking
                status VARCHAR(20) DEFAULT 'active',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                outcome_recorded_at TIMESTAMP,
                
                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                
                FOREIGN KEY (article_id) REFERENCES articles(id)
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX idx_prediction_symbol ON stock_predictions(symbol)")
        conn.execute("CREATE INDEX idx_prediction_date ON stock_predictions(prediction_date)")
        conn.execute("CREATE INDEX idx_prediction_status ON stock_predictions(status)")
        conn.execute("CREATE INDEX idx_prediction_symbol_date ON stock_predictions(symbol, prediction_date)")
        conn.execute("CREATE INDEX idx_prediction_performance ON stock_predictions(was_correct, confidence_score)")
        conn.execute("CREATE INDEX idx_prediction_sector ON stock_predictions(sector, prediction_date)")
        
        conn.commit()
        
        print("✅ stock_predictions table created with 6 indexes")

print("🎉 Migration completed successfully!")
