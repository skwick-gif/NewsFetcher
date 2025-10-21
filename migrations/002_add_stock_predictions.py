"""
Migration: Add stock_predictions table for tracking ML predictions
Date: 2025-10-20
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey, Index, create_engine
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


def upgrade(engine):
    """Add stock_predictions table"""
    print("🔧 Adding stock_predictions table...")
    
    # Create the table using raw SQL to ensure it works
    with engine.connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_predictions (
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
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_symbol 
            ON stock_predictions(symbol)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_date 
            ON stock_predictions(prediction_date)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_status 
            ON stock_predictions(status)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_symbol_date 
            ON stock_predictions(symbol, prediction_date)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_performance 
            ON stock_predictions(was_correct, confidence_score)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_sector 
            ON stock_predictions(sector, prediction_date)
        """)
        
        conn.commit()
        print("✅ stock_predictions table created successfully")


def downgrade(engine):
    """Remove stock_predictions table"""
    print("🔧 Removing stock_predictions table...")
    
    with engine.connect() as conn:
        conn.execute("DROP TABLE IF EXISTS stock_predictions")
        conn.commit()
        print("✅ stock_predictions table removed")


if __name__ == "__main__":
    from sqlalchemy import create_engine
    import sys
    
    # Default database path
    db_path = "D:/Projects/NewsFetcher/app/tariff_radar.db"
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    print(f"📂 Using database: {db_path}")
    engine = create_engine(f'sqlite:///{db_path}')
    
    # Run upgrade
    upgrade(engine)
    print("🎉 Migration completed!")
