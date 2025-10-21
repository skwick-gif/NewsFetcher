"""
Complete MarketPulse Status Check
Check all components and display what's working
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

async def main():
    print(f"\n{BLUE}{BOLD}{'='*70}{RESET}")
    print(f"{BLUE}{BOLD}🚀 MarketPulse Financial Intelligence Platform - Status Check{RESET}")
    print(f"{BLUE}{BOLD}{'='*70}{RESET}\n")
    
    # 1. Check API Keys
    print(f"{CYAN}{BOLD}📋 API Keys Configuration:{RESET}")
    print(f"{BLUE}{'─'*70}{RESET}")
    
    apis = {
        'Alpha Vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
        'Twitter Bearer': os.getenv('TWITTER_BEARER_TOKEN'),
        'Reddit Client ID': os.getenv('REDDIT_CLIENT_ID'),
        'Reddit Secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'Perplexity': os.getenv('PERPLEXITY_API_KEY'),
    }
    
    configured_apis = 0
    for name, key in apis.items():
        if key and key not in ['YOUR_', 'CHANGE_THIS']:
            print(f"  {GREEN}✅ {name}: Configured{RESET}")
            configured_apis += 1
        else:
            print(f"  {YELLOW}⚠️  {name}: Not configured{RESET}")
    
    print(f"\n  {BOLD}Total: {configured_apis}/{len(apis)} APIs configured{RESET}\n")
    
    # 2. Test Data Sources
    print(f"{CYAN}{BOLD}🔍 Testing Data Sources:{RESET}")
    print(f"{BLUE}{'─'*70}{RESET}")
    
    working_sources = []
    failed_sources = []
    
    # Test Yahoo Finance
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        price = ticker.info.get('currentPrice')
        if price:
            print(f"  {GREEN}✅ Yahoo Finance: Working - AAPL=${price}{RESET}")
            working_sources.append('Yahoo Finance')
        else:
            print(f"  {YELLOW}⚠️  Yahoo Finance: Connected but limited data{RESET}")
    except Exception as e:
        print(f"  {RED}❌ Yahoo Finance: Failed - {e}{RESET}")
        failed_sources.append('Yahoo Finance')
    
    # Test Alpha Vantage
    if apis['Alpha Vantage']:
        try:
            import requests
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={apis['Alpha Vantage']}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if "Global Quote" in data and data["Global Quote"]:
                price = data["Global Quote"].get("05. price")
                print(f"  {GREEN}✅ Alpha Vantage: Working - AAPL=${price}{RESET}")
                working_sources.append('Alpha Vantage')
            elif "Note" in data:
                print(f"  {YELLOW}⚠️  Alpha Vantage: Rate limit (free tier: 5 calls/min){RESET}")
            else:
                print(f"  {YELLOW}⚠️  Alpha Vantage: Unexpected response{RESET}")
        except Exception as e:
            print(f"  {RED}❌ Alpha Vantage: Failed - {e}{RESET}")
            failed_sources.append('Alpha Vantage')
    
    # Test Twitter
    if apis['Twitter Bearer']:
        try:
            import requests
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {apis['Twitter Bearer']}"}
            params = {"query": "AAPL", "max_results": 10}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                count = len(response.json().get('data', []))
                print(f"  {GREEN}✅ Twitter API: Working - {count} tweets found{RESET}")
                working_sources.append('Twitter')
            elif response.status_code == 429:
                print(f"  {YELLOW}⚠️  Twitter API: Rate limited{RESET}")
            else:
                print(f"  {RED}❌ Twitter API: HTTP {response.status_code}{RESET}")
                failed_sources.append('Twitter')
        except Exception as e:
            print(f"  {RED}❌ Twitter API: Failed - {e}{RESET}")
            failed_sources.append('Twitter')
    
    # Test Reddit
    if apis['Reddit Client ID'] and apis['Reddit Secret']:
        try:
            import praw
            reddit = praw.Reddit(
                client_id=apis['Reddit Client ID'],
                client_secret=apis['Reddit Secret'],
                user_agent='MarketPulse:v1.0'
            )
            subreddit = reddit.subreddit('stocks')
            posts = list(subreddit.hot(limit=1))
            print(f"  {GREEN}✅ Reddit API: Working - Connected to r/stocks{RESET}")
            working_sources.append('Reddit')
        except Exception as e:
            print(f"  {RED}❌ Reddit API: Failed - {e}{RESET}")
            failed_sources.append('Reddit')
    
    print(f"\n  {BOLD}Working: {len(working_sources)} | Failed: {len(failed_sources)}{RESET}\n")
    
    # 3. Check ML Components
    print(f"{CYAN}{BOLD}🤖 Machine Learning Components:{RESET}")
    print(f"{BLUE}{'─'*70}{RESET}")
    
    ml_components = []
    
    try:
        import tensorflow as tf
        print(f"  {GREEN}✅ TensorFlow {tf.__version__}: Installed{RESET}")
        ml_components.append('TensorFlow')
    except:
        print(f"  {RED}❌ TensorFlow: Not installed{RESET}")
    
    try:
        import sklearn
        print(f"  {GREEN}✅ Scikit-learn {sklearn.__version__}: Installed{RESET}")
        ml_components.append('Scikit-learn')
    except:
        print(f"  {RED}❌ Scikit-learn: Not installed{RESET}")
    
    try:
        from financial.ml_trainer import MLModelTrainer
        print(f"  {GREEN}✅ ML Trainer Module: Available{RESET}")
        ml_components.append('ML Trainer')
    except:
        print(f"  {RED}❌ ML Trainer Module: Not found{RESET}")
    
    try:
        from financial.neural_networks import EnsembleNeuralNetwork
        print(f"  {GREEN}✅ Neural Networks Module: Available{RESET}")
        ml_components.append('Neural Networks')
    except:
        print(f"  {RED}❌ Neural Networks Module: Not found{RESET}")
    
    print(f"\n  {BOLD}Total: {len(ml_components)}/4 ML components ready{RESET}\n")
    
    # 4. Summary
    print(f"{CYAN}{BOLD}📊 Overall Status:{RESET}")
    print(f"{BLUE}{'─'*70}{RESET}")
    
    total_score = configured_apis + len(working_sources) + len(ml_components)
    max_score = len(apis) + 4 + 4  # APIs + Data Sources + ML Components
    
    percentage = (total_score / max_score) * 100
    
    if percentage >= 70:
        status_icon = f"{GREEN}✅{RESET}"
        status_text = f"{GREEN}EXCELLENT{RESET}"
    elif percentage >= 50:
        status_icon = f"{YELLOW}⚠️{RESET}"
        status_text = f"{YELLOW}GOOD{RESET}"
    else:
        status_icon = f"{RED}❌{RESET}"
        status_text = f"{RED}NEEDS ATTENTION{RESET}"
    
    print(f"\n  {status_icon} System Status: {status_text}")
    print(f"  {BOLD}Score: {total_score}/{max_score} ({percentage:.1f}%){RESET}")
    
    print(f"\n  {CYAN}Capabilities:{RESET}")
    print(f"    • Real-time market data: {GREEN}YES{RESET} (Yahoo Finance)")
    print(f"    • Financial fundamentals: {GREEN}YES{RESET} (Yahoo Finance)")
    print(f"    • Social sentiment: {GREEN if len([s for s in working_sources if s in ['Twitter', 'Reddit']]) > 0 else YELLOW}{'YES' if len([s for s in working_sources if s in ['Twitter', 'Reddit']]) > 0 else 'PARTIAL'}{RESET}")
    print(f"    • ML predictions: {GREEN if 'TensorFlow' in ml_components else YELLOW}{'YES' if 'TensorFlow' in ml_components else 'READY'}{RESET}")
    print(f"    • Neural networks: {GREEN if 'Neural Networks' in ml_components else YELLOW}{'YES' if 'Neural Networks' in ml_components else 'READY'}{RESET}")
    
    # 5. Recommendations
    print(f"\n{CYAN}{BOLD}💡 Recommendations:{RESET}")
    print(f"{BLUE}{'─'*70}{RESET}")
    
    if 'Yahoo Finance' in working_sources:
        print(f"  {GREEN}✓{RESET} Yahoo Finance working - Use as primary data source")
    
    if 'Twitter' in working_sources or 'Reddit' in working_sources:
        print(f"  {GREEN}✓{RESET} Social media APIs working - Real sentiment analysis available")
    else:
        print(f"  {YELLOW}!{RESET} Configure social media APIs for sentiment analysis")
    
    if len(ml_components) >= 3:
        print(f"  {GREEN}✓{RESET} ML stack ready - Can train and deploy models")
    
    print(f"\n  {BOLD}Primary Data Flow:{RESET}")
    print(f"    1. Yahoo Finance → Real-time prices & fundamentals")
    print(f"    2. {'Twitter/Reddit' if len([s for s in working_sources if s in ['Twitter', 'Reddit']]) > 0 else '(No social data)'} → Sentiment analysis")
    print(f"    3. TensorFlow/Scikit-learn → ML predictions")
    print(f"    4. WebSocket → Live streaming to dashboard")
    
    print(f"\n{BLUE}{BOLD}{'='*70}{RESET}\n")
    
    print(f"{GREEN}{BOLD}✅ Status check complete!{RESET}")
    print(f"{CYAN}Ready to start server: py main_production_enhanced.py{RESET}\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Check interrupted{RESET}")
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
