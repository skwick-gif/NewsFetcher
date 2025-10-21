"""
Smart Sector Scanner - Dynamic Stock Discovery System
Scans news and market data to find high-potential stocks by sector
"""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class SectorScanner:
    """
    Intelligent sector-based stock scanner
    Finds potential stocks based on sector-specific keywords and patterns
    """
    
    def __init__(self):
        # Define sector-specific keywords and patterns
        self.sector_keywords = {
            'healthcare_fda': {
                'name': 'Healthcare - FDA Approvals',
                'keywords': [
                    'FDA approval', 'FDA clearance', 'drug approval',
                    'clinical trial', 'breakthrough therapy', 'accelerated approval',
                    'orphan drug', 'priority review', 'fast track',
                    'phase 3', 'phase III', 'NDA submitted', 'BLA approval',
                    'biotech', 'pharmaceutical', 'medical device approval',
                    'EUA granted', 'emergency use authorization'
                ],
                'positive_patterns': [
                    'approved', 'granted', 'clearance', 'positive results',
                    'successful trial', 'breakthrough', 'accelerated'
                ],
                'negative_patterns': [
                    'rejected', 'delayed', 'failed trial', 'adverse events',
                    'warning letter', 'recall'
                ],
                'tickers_to_watch': [
                    'PFE', 'JNJ', 'MRNA', 'BNTX', 'GILD', 'BMY', 'ABBV',
                    'REGN', 'VRTX', 'BIIB', 'AMGN', 'ISRG'
                ]
            },
            
            'quantum_computing': {
                'name': 'Quantum Computing',
                'keywords': [
                    'quantum computing', 'quantum processor', 'qubit',
                    'quantum advantage', 'quantum supremacy',
                    'quantum algorithm', 'quantum breakthrough',
                    'quantum chip', 'quantum cloud', 'quantum network',
                    'superconducting qubit', 'quantum error correction'
                ],
                'positive_patterns': [
                    'breakthrough', 'milestone', 'first', 'revolutionary',
                    'partnership', 'contract', 'deal', 'investment'
                ],
                'negative_patterns': [
                    'setback', 'delay', 'failed', 'loss'
                ],
                'tickers_to_watch': [
                    'IONQ', 'RGTI', 'QUBT', 'IBM', 'GOOGL', 'MSFT',
                    'NVDA', 'INTC'
                ]
            },
            
            'ai_chips': {
                'name': 'AI & Semiconductors',
                'keywords': [
                    'AI chip', 'GPU', 'neural processor', 'TPU',
                    'AI accelerator', 'inference chip', 'training chip',
                    'semiconductor', 'chip shortage', 'foundry',
                    'TSMC', 'fab capacity', 'node', 'lithography',
                    'artificial intelligence', 'machine learning chip'
                ],
                'positive_patterns': [
                    'record revenue', 'demand surge', 'capacity expansion',
                    'new product', 'breakthrough', 'partnership'
                ],
                'negative_patterns': [
                    'shortage', 'delay', 'competition', 'decline'
                ],
                'tickers_to_watch': [
                    'NVDA', 'AMD', 'INTC', 'TSM', 'AVGO', 'QCOM',
                    'MU', 'ASML', 'LRCX', 'KLAC'
                ]
            },
            
            'ev_battery': {
                'name': 'Electric Vehicles & Battery Tech',
                'keywords': [
                    'electric vehicle', 'EV', 'battery technology',
                    'solid state battery', 'lithium', 'charging station',
                    'range', 'autonomous', 'self-driving',
                    'battery capacity', 'fast charging', 'gigafactory',
                    'EV sales', 'delivery numbers'
                ],
                'positive_patterns': [
                    'record deliveries', 'production ramp', 'new model',
                    'battery breakthrough', 'range increase', 'partnership'
                ],
                'negative_patterns': [
                    'recall', 'production halt', 'delay', 'fire'
                ],
                'tickers_to_watch': [
                    'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
                    'GM', 'F', 'ALB', 'LAC', 'SQM'
                ]
            },
            
            'space_tech': {
                'name': 'Space Technology',
                'keywords': [
                    'satellite', 'rocket launch', 'space mission',
                    'orbital', 'LEO', 'constellation', 'starlink',
                    'space station', 'reusable rocket', 'payload',
                    'launch contract', 'NASA', 'commercial space'
                ],
                'positive_patterns': [
                    'successful launch', 'contract awarded', 'mission success',
                    'new contract', 'partnership'
                ],
                'negative_patterns': [
                    'launch failure', 'explosion', 'delay', 'scrubbed'
                ],
                'tickers_to_watch': [
                    'RKLB', 'SPCE', 'BA', 'LMT', 'NOC', 'RTX'
                ]
            },
            
            'renewable_energy': {
                'name': 'Renewable Energy',
                'keywords': [
                    'solar', 'wind energy', 'renewable', 'green energy',
                    'carbon neutral', 'clean energy', 'hydrogen',
                    'energy storage', 'grid battery', 'offshore wind',
                    'solar panel', 'wind turbine', 'IRA tax credit'
                ],
                'positive_patterns': [
                    'new project', 'capacity increase', 'contract',
                    'government support', 'subsidy', 'record production'
                ],
                'negative_patterns': [
                    'tariff', 'subsidy cut', 'project cancelled'
                ],
                'tickers_to_watch': [
                    'ENPH', 'SEDG', 'NEE', 'BEP', 'FSLR', 'RUN',
                    'PLUG', 'BE', 'ICLN'
                ]
            },
            
            'cybersecurity': {
                'name': 'Cybersecurity',
                'keywords': [
                    'cybersecurity', 'data breach', 'ransomware',
                    'zero trust', 'endpoint security', 'cloud security',
                    'threat detection', 'vulnerability', 'hack',
                    'encryption', 'firewall', 'security software'
                ],
                'positive_patterns': [
                    'new product', 'contract win', 'partnership',
                    'revenue growth', 'acquisition'
                ],
                'negative_patterns': [
                    'breach', 'vulnerability', 'competition'
                ],
                'tickers_to_watch': [
                    'CRWD', 'PANW', 'ZS', 'OKTA', 'S', 'FTNT',
                    'NET', 'CYBR'
                ]
            },
            
            'fintech': {
                'name': 'Financial Technology',
                'keywords': [
                    'fintech', 'digital payment', 'crypto', 'blockchain',
                    'buy now pay later', 'BNPL', 'neobank',
                    'payment processing', 'digital wallet', 'defi',
                    'merchant services', 'peer-to-peer'
                ],
                'positive_patterns': [
                    'user growth', 'transaction volume', 'partnership',
                    'new feature', 'expansion', 'profitability'
                ],
                'negative_patterns': [
                    'regulation', 'fraud', 'decline', 'competition'
                ],
                'tickers_to_watch': [
                    'SQ', 'PYPL', 'AFRM', 'COIN', 'SOFI', 'UPST',
                    'NU', 'V', 'MA'
                ]
            },
            
            'genomics': {
                'name': 'Genomics & Gene Therapy',
                'keywords': [
                    'gene therapy', 'CRISPR', 'genomics', 'DNA sequencing',
                    'personalized medicine', 'gene editing',
                    'CAR-T', 'immunotherapy', 'precision medicine',
                    'genetic testing', 'RNA therapy'
                ],
                'positive_patterns': [
                    'breakthrough', 'approval', 'clinical success',
                    'partnership', 'acquisition'
                ],
                'negative_patterns': [
                    'trial failure', 'safety concern', 'delay'
                ],
                'tickers_to_watch': [
                    'CRSP', 'EDIT', 'NTLA', 'ILMN', 'BEAM', 'PACB',
                    'IONS', 'ARKG'
                ]
            },
            
            'cloud_saas': {
                'name': 'Cloud & SaaS',
                'keywords': [
                    'cloud computing', 'SaaS', 'software as a service',
                    'cloud infrastructure', 'data center', 'serverless',
                    'kubernetes', 'container', 'microservices',
                    'API', 'platform', 'subscription'
                ],
                'positive_patterns': [
                    'ARR growth', 'new customers', 'expansion',
                    'product launch', 'partnership', 'upsell'
                ],
                'negative_patterns': [
                    'churn', 'competition', 'price cut', 'customer loss'
                ],
                'tickers_to_watch': [
                    'CRM', 'NOW', 'SNOW', 'DDOG', 'MDB', 'NET',
                    'PLTR', 'WDAY', 'ZM', 'DOCU'
                ]
            }
        }
        
        # Scoring weights
        self.scoring_weights = {
            'keyword_match': 3.0,
            'positive_pattern': 2.0,
            'negative_pattern': -3.0,
            'ticker_mentioned': 2.5,
            'recent_news': 1.5
        }
        
        logger.info(f"✅ SectorScanner initialized with {len(self.sector_keywords)} sectors")
    
    def scan_article(self, article: Dict) -> Dict:
        """
        Scan an article for sector relevance and potential stocks
        
        Returns:
        {
            'sectors_matched': [...],
            'potential_tickers': [...],
            'relevance_score': float,
            'sentiment': 'positive'|'negative'|'neutral',
            'key_findings': [...]
        }
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = f"{title} {description}"
        
        result = {
            'sectors_matched': [],
            'potential_tickers': set(),
            'relevance_score': 0.0,
            'sentiment': 'neutral',
            'key_findings': []
        }
        
        for sector_id, sector_data in self.sector_keywords.items():
            sector_score = 0
            keywords_found = []
            positive_patterns_found = []
            negative_patterns_found = []
            tickers_found = set()
            
            # Check keywords
            for keyword in sector_data['keywords']:
                if keyword.lower() in content:
                    sector_score += self.scoring_weights['keyword_match']
                    keywords_found.append(keyword)
            
            # Check positive patterns
            for pattern in sector_data['positive_patterns']:
                if pattern.lower() in content:
                    sector_score += self.scoring_weights['positive_pattern']
                    positive_patterns_found.append(pattern)
            
            # Check negative patterns
            for pattern in sector_data['negative_patterns']:
                if pattern.lower() in content:
                    sector_score += self.scoring_weights['negative_pattern']
                    negative_patterns_found.append(pattern)
            
            # Check tickers
            for ticker in sector_data['tickers_to_watch']:
                # Check for ticker mentions (with word boundaries)
                if f" {ticker.lower()} " in f" {content} " or f"${ticker.lower()}" in content:
                    sector_score += self.scoring_weights['ticker_mentioned']
                    tickers_found.add(ticker)
            
            # If sector is relevant, add to results
            if sector_score > 5.0:  # Threshold for relevance
                result['sectors_matched'].append({
                    'sector_id': sector_id,
                    'sector_name': sector_data['name'],
                    'score': sector_score,
                    'keywords_found': keywords_found[:5],  # Top 5
                    'positive_signals': len(positive_patterns_found),
                    'negative_signals': len(negative_patterns_found)
                })
                result['potential_tickers'].update(tickers_found)
                result['relevance_score'] += sector_score
            
            # Determine sentiment
            if positive_patterns_found and not negative_patterns_found:
                result['sentiment'] = 'positive'
            elif negative_patterns_found and not positive_patterns_found:
                result['sentiment'] = 'negative'
            elif len(positive_patterns_found) > len(negative_patterns_found):
                result['sentiment'] = 'positive'
            elif len(negative_patterns_found) > len(positive_patterns_found):
                result['sentiment'] = 'negative'
        
        # Convert set to list
        result['potential_tickers'] = list(result['potential_tickers'])
        
        # Sort sectors by score
        result['sectors_matched'].sort(key=lambda x: x['score'], reverse=True)
        
        return result
    
    async def scan_ticker_with_ml(self, ticker: str) -> Dict:
        """
        Scan a ticker using ML models to determine if it has potential
        
        Returns:
        {
            'ticker': str,
            'has_potential': bool,
            'ml_score': float,
            'ml_prediction': {...},
            'recommendation': str
        }
        """
        try:
            from financial.neural_networks import EnsembleNeuralNetwork
            from financial.market_data import financial_provider
            
            # Get stock data
            stock_data = await financial_provider.get_stock_data(ticker)
            if not stock_data:
                return {'ticker': ticker, 'has_potential': False, 'error': 'No data available'}
            
            # Create mock data for ML prediction (in production, use real historical data)
            import numpy as np
            current_price = stock_data.get('price', 100.0)
            sequence_length = 60
            data = np.random.randn(sequence_length, 14) * 10 + current_price
            data[-1, 0] = current_price
            
            # Get ML prediction
            ensemble = EnsembleNeuralNetwork()
            prediction = await ensemble.predict(data, ticker)
            
            ensemble_pred = prediction.get('ensemble_prediction', {})
            predicted_price = ensemble_pred.get('prediction', current_price)
            confidence = ensemble_pred.get('confidence', 0.5)
            
            # Calculate expected return
            expected_return = ((predicted_price - current_price) / current_price) * 100
            
            # Determine if it has potential
            has_potential = (
                expected_return > 2.0 and  # At least 2% expected gain
                confidence > 0.65 and  # High confidence
                prediction.get('model_agreement', {}).get('agreement_score', 0) > 0.7  # Models agree
            )
            
            # Generate recommendation
            if expected_return > 5 and confidence > 0.75:
                recommendation = "STRONG BUY"
            elif expected_return > 2 and confidence > 0.65:
                recommendation = "BUY"
            elif expected_return < -5:
                recommendation = "SELL"
            elif expected_return < -2:
                recommendation = "HOLD/SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'ticker': ticker,
                'has_potential': has_potential,
                'ml_score': confidence * 100,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'expected_return': expected_return,
                'confidence': confidence,
                'recommendation': recommendation,
                'ml_prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error scanning {ticker} with ML: {e}")
            return {'ticker': ticker, 'has_potential': False, 'error': str(e)}
    
    def get_all_sectors(self) -> List[Dict]:
        """Get list of all monitored sectors"""
        return [
            {
                'sector_id': sector_id,
                'sector_name': data['name'],
                'keywords_count': len(data['keywords']),
                'tickers_count': len(data['tickers_to_watch'])
            }
            for sector_id, data in self.sector_keywords.items()
        ]
    
    def get_sector_tickers(self, sector_id: str) -> List[str]:
        """Get tickers for a specific sector"""
        sector = self.sector_keywords.get(sector_id, {})
        return sector.get('tickers_to_watch', [])


# Global instance
sector_scanner = SectorScanner()
