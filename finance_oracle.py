import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import ta  # Using 'ta' library instead of pandas_ta
import yfinance as yf
import requests
import warnings
import time
import json
from fredapi import Fred
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import optuna
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np

# Check for CUDA availability and configure PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable TF32 precision on Ampere GPUs (like H100) for faster matrix multiplications
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN autotuning for better performance
torch.backends.cudnn.benchmark = True

# Print GPU information if available
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Memory Usage:")
    print(f"  Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
    print(f"  Cached:    {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")

# GPU Memory Management utility
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
        print(f"  Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
        print(f"  Cached:    {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")

# --------------------- News Fetcher with Polygon API ---------------------
def fetch_financial_news_polygon(ticker, start_date, end_date, date_index, api_key):
    """
    Fetch financial news for a ticker using the Polygon.io API.

    This function retrieves news articles from the Polygon.io News API endpoint.
    It then maps those articles to the nearest trading day in the date_index.

    The Polygon.io News API endpoint documentation:
    https://polygon.io/docs/rest/stocks/news

    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        date_index (pd.DatetimeIndex): Trading dates to map news to
        api_key (str): Polygon.io API key

    Returns:
        dict: Mapping of dates to lists of news articles
    """
    news_data = {}

    # Initialize empty lists for all dates
    for d in date_index:
        news_data[d] = []

    # Convert date strings to ISO format for API
    start_date_iso = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
    end_date_iso = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d')

    # Construct API URL
    base_url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "published_utc.gte": start_date_iso,
        "published_utc.lte": end_date_iso,
        "limit": 1000,  # Maximum number of results per request
        "sort": "published_utc",
        "order": "desc",
        "apiKey": api_key
    }

    try:
        # Make the API request
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            result = response.json()
            articles = result.get('results', [])

            print(f"Fetched {len(articles)} articles from Polygon API")

            # Process each article
            for article in articles:
                # Extract publication date and convert to datetime
                pub_date_str = article.get('published_utc', '')
                if not pub_date_str:
                    continue

                # Convert UTC string to datetime object
                try:
                    pub_date = datetime.strptime(pub_date_str[:10], '%Y-%m-%d')
                except ValueError:
                    continue

                # Find the matching trading day (use the next available trading day if not in date_index)
                matching_day = None
                for d in date_index:
                    if d.date() >= pub_date.date():
                        matching_day = d
                        break

                if matching_day is None and len(date_index) > 0:
                    matching_day = date_index[0]  # Use the first trading day if no match

                if matching_day:
                    # Extract title and summary to create a comprehensive article text
                    title = article.get('title', '')
                    description = article.get('description', '')

                    # Combine title and description
                    article_text = f"{title}. {description}"

                    # Add to the corresponding date
                    if matching_day in news_data:
                        news_data[matching_day].append(article_text)

        else:
            print(f"Polygon API returned status code {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Error fetching news from Polygon API: {e}")

    # Check if we got any articles
    total_articles = sum(len(articles) for articles in news_data.values())
    if total_articles == 0:
        print("Warning: No articles fetched from Polygon API. The model will rely only on market data.")

    return news_data

# --------------------- Historical News Fetcher ---------------------
def fetch_historical_news_polygon(ticker, years_back=3, api_key=None):
    """
    Fetch historical news data for a ticker from Polygon.io API,
    going back a specified number of years.

    Args:
        ticker (str): Stock ticker symbol
        years_back (int): Number of years to look back
        api_key (str): Polygon.io API key

    Returns:
        dict: A dictionary mapping dates to lists of news articles
    """
    if not api_key:
        raise ValueError("Polygon API key is required")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)

    # Format dates for API request
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Fetching news from {start_date_str} to {end_date_str} for {ticker}...")

    # Create date index for all trading days in the range
    # This will depend on market calendar, but for demo we'll use business days
    date_range = pd.date_range(start=start_date_str, end=end_date_str, freq='B')

    # Initialize result dictionary
    news_data = {d: [] for d in date_range}

    # Define pagination parameters
    limit = 1000  # Maximum allowed by Polygon
    base_url = "https://api.polygon.io/v2/reference/news"

    # Variables for pagination
    total_articles = 0
    next_url = None

    # Initial request parameters
    params = {
        "ticker": ticker,
        "published_utc.gte": start_date_str,
        "published_utc.lte": end_date_str,
        "limit": limit,
        "sort": "published_utc",
        "order": "desc",
        "apiKey": api_key
    }

    try:
        # Make initial request
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return news_data

        result = response.json()
        articles = result.get('results', [])
        total_articles += len(articles)

        # Process the articles from the first request
        process_articles(articles, news_data, date_range)

        # Check if there's a next_url for pagination
        next_url = result.get('next_url')

        # Continue paginating if there are more results
        while next_url and total_articles < 50000:  # Set a reasonable limit to prevent infinite loops
            # Append API key to next_url
            if '?' in next_url:
                paginated_url = f"{next_url}&apiKey={api_key}"
            else:
                paginated_url = f"{next_url}?apiKey={api_key}"

            response = requests.get(paginated_url)
            if response.status_code != 200:
                print(f"Error during pagination: {response.status_code}")
                break

            result = response.json()
            articles = result.get('results', [])

            if not articles:
                break

            total_articles += len(articles)
            process_articles(articles, news_data, date_range)

            # Update next_url for the next iteration
            next_url = result.get('next_url')

            # Print progress
            print(f"Fetched {total_articles} articles so far...")

            # Optional: Add a small delay to respect rate limits
            time.sleep(0.1)

    except Exception as e:
        print(f"Error fetching historical news: {e}")

    print(f"Total articles fetched: {total_articles}")

    # Count how many dates have articles
    dates_with_articles = sum(1 for articles in news_data.values() if articles)
    print(f"Dates with articles: {dates_with_articles} out of {len(date_range)}")

    return news_data

def process_articles(articles, news_data, date_range):
    """Helper function to process and map articles to trading dates"""
    for article in articles:
        # Extract publication date
        pub_date_str = article.get('published_utc', '')
        if not pub_date_str:
            continue

        # Convert UTC string to datetime object
        try:
            pub_date = datetime.strptime(pub_date_str[:10], '%Y-%m-%d')
        except ValueError:
            continue

        # Find the closest trading day (use the current day if it's a trading day,
        # otherwise use the next trading day)
        closest_date = None
        for d in date_range:
            if d.date() >= pub_date.date():
                closest_date = d
                break

        # If no future date found, use the last available date
        if closest_date is None and len(date_range) > 0:
            closest_date = date_range[-1]

        if closest_date:
            # Extract title and description
            title = article.get('title', '')
            description = article.get('description', '')

            # Combine title and description
            article_text = f"{title}. {description}"

            # Add to the corresponding date
            if closest_date in news_data:
                news_data[closest_date].append(article_text)

# --------------------- FinBERT Extractor ---------------------
class FinBERTFeatureExtractor:
    def __init__(self, model_name="yiyanghkust/finbert-tone", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = device  # Use global device (cuda if available)

        # Move model to GPU
        self.model = self.model.to(self.device)

    def extract_features(self, news_texts):
        """
        Extract FinBERT embeddings + simple sentiment for each article.
        If no articles are provided, returns zero embedding with neutral sentiment.
        """
        all_embeddings = []
        all_sentiments = []

        # If no articles, return zeros with neutral sentiment
        if len(news_texts) == 0:
            # Create a single zero embedding with neutral sentiment
            zero_embedding = np.zeros(768)
            neutral_sentiment = 0.5
            return {
                'embeddings': np.array([zero_embedding]),
                'sentiment_scores': np.array([[neutral_sentiment]])
            }

        for text in news_texts:
            # Skip empty texts
            if not text.strip():
                continue

            # Process inputs on GPU
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length,
                                    padding="max_length", truncation=True)

            # Move inputs to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # [CLS] token embedding - move back to CPU for numpy conversion
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embedding[0])

            # Basic sentiment from the average embedding
            pos_score = torch.sigmoid(torch.tensor(np.mean(cls_embedding), device=self.device)).cpu().item()
            all_sentiments.append(pos_score)

        # If after processing, we still have no embeddings, return zeros
        if len(all_embeddings) == 0:
            zero_embedding = np.zeros(768)
            neutral_sentiment = 0.5
            return {
                'embeddings': np.array([zero_embedding]),
                'sentiment_scores': np.array([[neutral_sentiment]])
            }

        # Convert to arrays
        emb_array = np.array(all_embeddings)
        sent_array = np.array(all_sentiments).reshape(-1, 1)

        return {
            'embeddings': emb_array,          # shape (num_articles, 768)
            'sentiment_scores': sent_array    # shape (num_articles, 1)
        }

# --------------------- Market Feature Extractor ---------------------
class MarketFeatureExtractor:
    def __init__(self, fred_api_key):
        self.scaler = StandardScaler()
        self.fred = Fred(api_key=fred_api_key)

    def get_technical_indicators(self, ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Flatten columns if multi-index
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data['Close'] = data['Close'].squeeze()

        print("DEBUG: data.columns =", data.columns)
        print("DEBUG: type(data['Close']) =", type(data['Close']))
        print("DEBUG: data['Close'].shape =", data['Close'].shape)

        # Must have at least 50 points for 50-day MAs
        if data.empty or len(data) < 50:
            raise ValueError(f"Not enough data available for {ticker} in the specified date range")

        # Calculate indicators
        data['MA5'] = ta.trend.sma_indicator(data['Close'], window=5)
        data['MA20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['MA50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['MA200'] = ta.trend.sma_indicator(data['Close'], window=200)

        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Diff'] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(data['Close'])
        data['BB_High'] = bb.bollinger_hband()
        data['BB_Low'] = bb.bollinger_lband()
        data['BB_Mid'] = bb.bollinger_mavg()

        data['Volume_Change'] = data['Volume'].pct_change()
        data['Volume_MA5'] = ta.trend.sma_indicator(data['Volume'], window=5)

        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
        data['Volatility'] = data['Close'].rolling(window=5).std()

        data = data.dropna()
        return data
        
    def enhance_features(self, market_data, macro_data):
        """Enhanced feature engineering with proven alpha factors"""
        
        # Mean reversion features
        market_data['price_zscore_5d'] = (market_data['Close'] - market_data['Close'].rolling(5).mean()) / market_data['Close'].rolling(5).std()
        market_data['price_zscore_20d'] = (market_data['Close'] - market_data['Close'].rolling(20).mean()) / market_data['Close'].rolling(20).std()
        
        # Momentum features with optimized lookbacks
        market_data['momentum_1m'] = market_data['Close'].pct_change(21)
        market_data['momentum_3m'] = market_data['Close'].pct_change(63)
        market_data['momentum_6m'] = market_data['Close'].pct_change(126)
        
        # Volatility regime features
        market_data['vol_ratio'] = market_data['Volatility'] / market_data['Volatility'].rolling(60).mean()
        
        # Cross-asset correlation factors
        if not macro_data.empty and 'VIXCLS' in macro_data.columns:
            # First resample macro data to match market_data's index
            macro_resampled = macro_data.reindex(market_data.index, method='ffill')
            market_data['rsi_vix_ratio'] = market_data['RSI'] / macro_resampled['VIXCLS']
        
        # Feature transformations for non-linear relationships
        market_data['log_volume'] = np.log1p(market_data['Volume'])
        market_data['volume_momentum'] = market_data['Volume'].pct_change(5) * market_data['Price_Change']
        
        # Market regime identification
        market_data['bull_market'] = (market_data['Close'] > market_data['MA200']).astype(int)
        market_data['high_volatility_regime'] = (market_data['Volatility'] > market_data['Volatility'].rolling(60).mean()).astype(int)
        
        # Price pattern recognition
        market_data['price_acceleration'] = market_data['Price_Change'].diff()
        market_data['momentum_divergence'] = ((market_data['Close'] > market_data['Close'].shift(10)) & 
                                             (market_data['RSI'] < market_data['RSI'].shift(10))).astype(int)
        
        # Volume-price relationship
        market_data['volume_price_trend'] = np.sign(market_data['Price_Change']) * market_data['Volume'] / market_data['Volume'].rolling(20).mean()
        
        # Mean reversion signals
        market_data['overbought'] = (market_data['RSI'] > 70).astype(int)
        market_data['oversold'] = (market_data['RSI'] < 30).astype(int)
        
        # Moving average crossovers
        market_data['ma_crossover_5_20'] = ((market_data['MA5'] > market_data['MA20']) & 
                                          (market_data['MA5'].shift(1) <= market_data['MA20'].shift(1))).astype(int)
        
        # Technical breakouts
        market_data['breakout_high'] = ((market_data['Close'] > market_data['High'].rolling(20).max().shift(1)) & 
                                      (market_data['Volume'] > market_data['Volume'].rolling(20).mean())).astype(int)
        
        # Fill NaN values with appropriate methods
        market_data = market_data.replace([np.inf, -np.inf], np.nan)
        market_data = market_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return market_data

    def get_macroeconomic_data(self, start_date, end_date):
        """
        Get macroeconomic data using FRED API
    
        Key indicators:
        - CPIAUCSL: Consumer Price Index (Inflation)
        - UNRATE: Unemployment Rate
        - FEDFUNDS: Federal Funds Rate (Interest rates)
        - T10Y2Y: 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity (Yield curve)
        - VIXCLS: CBOE Volatility Index (Market volatility)
        - DGS10: 10-Year Treasury Constant Maturity Rate
        - DTWEXBGS: Trade Weighted U.S. Dollar Index (Dollar strength)
        - M2SL: M2 Money Stock (Money supply)
        - INDPRO: Industrial Production Index (Economic output)
        """
        # Define list of FRED series to fetch
        fred_series = [
            'CPIAUCSL',  # Consumer Price Index
            'UNRATE',    # Unemployment Rate
            'FEDFUNDS',  # Federal Funds Rate
            'T10Y2Y',    # 10Y-2Y Treasury Spread
            'VIXCLS',    # VIX Volatility Index
            'DGS10',     # 10-Year Treasury Rate
            'DTWEXBGS',  # Dollar Index
            'M2SL',      # Money Supply
            'INDPRO',    # Industrial Production
            'PAYEMS',    # Non-farm Payrolls
            'UMCSENT',   # Consumer Sentiment
            'HOUST',     # Housing Starts
            'PCE',       # Personal Consumption Expenditures
            'PPIACO',    # Producer Price Index
            'RETAILSMNSA', # Retail Sales
            'ICSA',      # Initial Jobless Claims
            'PERMIT',    # Building Permits
            'BUSLOANS',  # Commercial and Industrial Loans
            'MORTGAGE30US', # 30-Year Fixed Rate Mortgage Average
            'JTSJOL'     # Job Openings
        ]
    
        # Convert date strings to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
        # Add extra buffer at the start to handle potential NA values
        # FRED data can be monthly or quarterly, so we need more history
        start_dt_buffer = start_dt - timedelta(days=365)
    
        print("Fetching FRED macroeconomic data...")
    
        # Fetch data for each series
        all_macro_data = {}
        for series_id in fred_series:
            try:
                series_data = self.fred.get_series(
                    series_id,
                    observation_start=start_dt_buffer.strftime('%Y-%m-%d'),
                    observation_end=end_dt.strftime('%Y-%m-%d')
                )
    
                if not series_data.empty:
                    # Store the series in our dictionary
                    all_macro_data[series_id] = series_data
                    print(f"  - Successfully fetched {series_id} ({len(series_data)} observations)")
                else:
                    print(f"  - No data found for {series_id}")
            except Exception as e:
                print(f"  - Error fetching {series_id}: {e}")
    
        # Create a DataFrame with all series
        if all_macro_data:
            # Combine all series into a single DataFrame
            macro_df = pd.DataFrame(all_macro_data)
    
            # We need to resample to daily frequency since FRED data has mixed frequencies
            # First, forward fill missing dates
            idx = pd.date_range(start=start_dt, end=end_dt)
            macro_df = macro_df.reindex(idx, method='ffill')
    
            # For any remaining NaNs, use backward fill
            macro_df = macro_df.fillna(method='bfill')
    
            # Calculate additional derived features
            for col in macro_df.columns:
                # Skip columns with too many NaNs
                if macro_df[col].isna().sum() > len(macro_df) * 0.3:
                    continue
    
                # Percent changes for different time windows
                macro_df[f'{col}_1m_change'] = macro_df[col].pct_change(periods=20)  # ~1 month
                macro_df[f'{col}_3m_change'] = macro_df[col].pct_change(periods=60)  # ~3 months
    
                # Calculate trends (slope)
                macro_df[f'{col}_trend'] = macro_df[col].rolling(window=30).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 5 else np.nan
                )
                
                # Try-except block for Z-scores to handle potential division by zero
                try:
                    # Z-scores to measure extremes
                    macro_df[f'{col}_zscore'] = (macro_df[col] - macro_df[col].rolling(252).mean()) / macro_df[col].rolling(252).std()
                except:
                    # If division by zero or other error, fill with zeros
                    macro_df[f'{col}_zscore'] = 0
                
                # Rate of change acceleration
                macro_df[f'{col}_acceleration'] = macro_df[f'{col}_1m_change'].diff(periods=20)
    
            # Economic health composite
            if all(col in macro_df.columns for col in ['UNRATE', 'CPIAUCSL', 'INDPRO']):
                # Make sure the z-scores are calculated first
                for col in ['UNRATE', 'CPIAUCSL', 'INDPRO']:
                    col_zscore = f'{col}_zscore'
                    if col_zscore not in macro_df.columns:
                        # Calculate it if not already present
                        macro_df[col_zscore] = (macro_df[col] - macro_df[col].rolling(252).mean()) / macro_df[col].rolling(252).std()
                
                # Now create the economic health composite            
                macro_df['economic_health'] = (
                    -1 * macro_df['UNRATE_zscore'] +  # Lower unemployment is better
                    -1 * macro_df['CPIAUCSL_zscore'] +  # Lower inflation is better
                    macro_df['INDPRO_zscore']  # Higher industrial production is better
                ) / 3
                    
            # Monetary policy composite
            if all(col in macro_df.columns for col in ['T10Y2Y', 'FEDFUNDS', 'DGS10']):
                # Make sure the trend columns are calculated first
                for col in ['FEDFUNDS', 'DGS10']:
                    col_trend = f'{col}_trend'
                    if col_trend not in macro_df.columns:
                        # Calculate it if not already present
                        macro_df[f'{col}_trend'] = macro_df[col].rolling(window=30).apply(
                            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 5 else np.nan
                        )
                
                # Now create the monetary policy composite
                macro_df['monetary_policy'] = (
                    macro_df['T10Y2Y'] +  # Steeper yield curve suggests expansionary
                    -1 * macro_df['FEDFUNDS_trend'] +  # Falling rates suggest expansionary
                    -1 * macro_df['DGS10_trend']  # Falling long rates suggest accommodative
                )
    
            # Risk sentiment composite
            if 'VIXCLS' in macro_df.columns:
                # Make sure the z-score is calculated first
                if 'VIXCLS_zscore' not in macro_df.columns:
                    macro_df['VIXCLS_zscore'] = (macro_df['VIXCLS'] - macro_df['VIXCLS'].rolling(252).mean()) / macro_df['VIXCLS'].rolling(252).std()
                
                # Now create the risk sentiment composite
                macro_df['risk_sentiment'] = -1 * macro_df['VIXCLS_zscore']  # Lower VIX implies risk-on
    
            # Fill any remaining NaNs with zeros
            macro_df = macro_df.fillna(0)
    
            print(f"Created macroeconomic feature dataframe with shape: {macro_df.shape}")
            return macro_df
        else:
            print("No macroeconomic data could be fetched from FRED")
            return pd.DataFrame()

    def get_market_microstructure(self, ticker, start_date, end_date, polygon_api_key):
        """
        Get market microstructure data using Polygon.io API
        """
        # Convert dates to required format
        start_date_iso = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        end_date_iso = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d')

        print(f"Fetching market microstructure from Polygon for {ticker}...")

        # 1. Get daily aggregates with adjusted close prices
        base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date_iso}/{end_date_iso}"
        params = {"apiKey": polygon_api_key, "adjusted": "true"}

        try:
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching microstructure data: {response.status_code}")
                print(f"Response: {response.text}")
                # Fall back to Yahoo Finance data
                return self._get_fallback_microstructure(ticker, start_date, end_date)

            data = response.json()
            results = data.get('results', [])

            if not results:
                print("No microstructure data found from Polygon")
                return self._get_fallback_microstructure(ticker, start_date, end_date)

            # Create DataFrame from results
            micro_data = pd.DataFrame(results)

            # Convert timestamp (in milliseconds) to datetime
            micro_data['timestamp'] = pd.to_datetime(micro_data['t'], unit='ms')
            micro_data.set_index('timestamp', inplace=True)

            # Rename columns to more meaningful names
            micro_data.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap',   # Volume-weighted average price
                'n': 'transactions'  # Number of transactions
            }, inplace=True)

            # 2. Calculate microstructure metrics
            # Volatility (Garman-Klass volatility estimator)
            micro_data['gk_volatility'] = np.sqrt(
                0.5 * np.log(micro_data['high'] / micro_data['low'])**2 -
                (2*np.log(2) - 1) * np.log(micro_data['close'] / micro_data['open'])**2
            )

            # Volume metrics
            micro_data['volume_ma5'] = micro_data['volume'].rolling(window=5).mean()
            micro_data['relative_volume'] = micro_data['volume'] / micro_data['volume_ma5']

            # Price range as proxy for liquidity
            micro_data['daily_range'] = (micro_data['high'] - micro_data['low']) / micro_data['close']
            micro_data['range_ma5'] = micro_data['daily_range'].rolling(window=5).mean()

            # VWAP distance as a proxy for price pressure
            micro_data['vwap_distance'] = (micro_data['close'] - micro_data['vwap']) / micro_data['vwap']

            # Transaction size
            micro_data['avg_trade_size'] = micro_data['volume'] / micro_data['transactions']

            # Other derived features
            micro_data['volume_per_range'] = micro_data['volume'] / (micro_data['high'] - micro_data['low'])
            micro_data['close_location'] = (micro_data['close'] - micro_data['low']) / (micro_data['high'] - micro_data['low'])

            # Advanced microstructure metrics
            # Price impact (Kyle's lambda) - approximation
            micro_data['price_impact'] = micro_data['close'].diff().abs() / micro_data['volume']
            
            # Gap analysis
            micro_data['overnight_gap'] = micro_data['open'] / micro_data['close'].shift(1) - 1
            micro_data['gap_fill'] = ((micro_data['open'] > micro_data['close'].shift(1)) & 
                                     (micro_data['low'] <= micro_data['close'].shift(1))).astype(int)
            
            # Market efficiency measures
            micro_data['high_low_range_ratio'] = (micro_data['high'] - micro_data['low']) / micro_data['close'].rolling(20).std()
            
            # Liquidity measures
            micro_data['amihud_illiquidity'] = micro_data['close'].pct_change().abs() / micro_data['volume']
            
            # Rolling features
            for window in [5, 10, 20]:
                # Volatility rolling windows
                micro_data[f'volatility_{window}d'] = micro_data['close'].pct_change().rolling(window=window).std()

                # Trading activity trend
                micro_data[f'volume_trend_{window}d'] = micro_data['volume'].rolling(window=window).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > window/2 else np.nan
                )

            # Fill NAs with forward fill then backward fill
            micro_data = micro_data.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Keep only the derived metrics, not the raw data
            derived_columns = [
                'gk_volatility', 'volume_ma5', 'relative_volume', 'daily_range',
                'range_ma5', 'vwap_distance', 'avg_trade_size', 'volume_per_range',
                'close_location', 'volatility_5d', 'volatility_10d', 'volatility_20d',
                'volume_trend_5d', 'volume_trend_10d', 'volume_trend_20d',
                'price_impact', 'overnight_gap', 'gap_fill', 'high_low_range_ratio', 
                'amihud_illiquidity'
            ]

            print(f"Successfully retrieved microstructure data from Polygon: {len(micro_data)} records")
            return micro_data[derived_columns]

        except Exception as e:
            print(f"Error in Polygon microstructure: {e}")
            # Fall back to Yahoo Finance
            return self._get_fallback_microstructure(ticker, start_date, end_date)

    def _get_fallback_microstructure(self, ticker, start_date, end_date):
        """Fallback method to get microstructure data from Yahoo Finance"""
        print("Falling back to Yahoo Finance for microstructure data...")
        # Get base price data
        price_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, interval='1d')

        if price_data.empty:
            print(f"No price data found for {ticker}")
            return pd.DataFrame()

        micro_data = pd.DataFrame(index=price_data.index)

        # Calculate actual microstructure metrics from price data
        # 1. Volatility
        micro_data['volatility_daily'] = price_data['Close'].pct_change().rolling(window=5).std()
        micro_data['volatility_weekly'] = price_data['Close'].pct_change().rolling(window=20).std()

        # 2. Volume metrics
        micro_data['volume'] = price_data['Volume']
        micro_data['volume_ma5'] = price_data['Volume'].rolling(window=5).mean()
        micro_data['relative_volume'] = price_data['Volume'] / micro_data['volume_ma5']

        # 3. Price range as proxy for liquidity
        micro_data['daily_range'] = (price_data['High'] - price_data['Low']) / price_data['Close']
        micro_data['daily_range_ma5'] = micro_data['daily_range'].rolling(window=5).mean()

        # 4. Gap as a proxy for overnight information
        micro_data['gap'] = (price_data['Open'] - price_data['Close'].shift(1)) / price_data['Close'].shift(1)

        # 5. Intraday volatility proxy
        micro_data['intraday_vol'] = (price_data['High'] - price_data['Low']) / (price_data['High'] + price_data['Low']) * 2

        # 6. Calculate candle body ratio (proxy for conviction)
        micro_data['body_ratio'] = abs(price_data['Close'] - price_data['Open']) / (price_data['High'] - price_data['Low'])
        
        # 7. Enhanced metrics
        micro_data['price_impact'] = price_data['Close'].diff().abs() / price_data['Volume']
        micro_data['overnight_gap'] = price_data['Open'] / price_data['Close'].shift(1) - 1
        micro_data['gap_fill'] = ((price_data['Open'] > price_data['Close'].shift(1)) & 
                                 (price_data['Low'] <= price_data['Close'].shift(1))).astype(int)
        
        # Volatility regimes
        for window in [5, 10, 20]:
            micro_data[f'volatility_{window}d'] = price_data['Close'].pct_change().rolling(window=window).std()
            micro_data[f'volume_trend_{window}d'] = price_data['Volume'].rolling(window=window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > window/2 else np.nan
            )

        # Fill missing values
        micro_data = micro_data.fillna(method='ffill').fillna(0)

        return micro_data

    def extract_features(self, ticker, start_date, end_date, polygon_api_key):
        """Extract and combine all features for the model"""
        try:
            # Get data from different sources
            print(f"Getting technical indicators for {ticker} from {start_date} to {end_date}")
            tech = self.get_technical_indicators(ticker, start_date, end_date)
            print(f"Technical data shape: {tech.shape}")
            
            print("Getting macroeconomic data")
            macro = self.get_macroeconomic_data(start_date, end_date)
            print(f"Macro data shape: {macro.shape if not macro.empty else '(empty)'}")
            
            print("Getting microstructure data")
            micro = self.get_market_microstructure(ticker, start_date, end_date, polygon_api_key)
            print(f"Micro data shape: {micro.shape if not micro.empty else '(empty)'}")
            
            # Apply enhanced feature engineering
            print("Applying feature engineering")
            tech = self.enhance_features(tech, macro)
            print(f"Enhanced tech data shape: {tech.shape}")
            
            # Ensure all dataframes have the same index (trading days)
            common_idx = tech.index
            
            # Reindex macro and micro data to match technical data
            if not macro.empty:
                macro = macro.reindex(common_idx, method='ffill')
            
            if not micro.empty:
                micro = micro.reindex(common_idx, method='ffill')
            
            # Extract basic features
            feature_cols = [
                'MA5', 'MA20', 'MA50', 'MA200',
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
                'BB_High', 'BB_Low', 'BB_Mid',
                'Volume_Change', 'Volume_MA5',
                'Price_Change', 'Price_Change_5d', 'Volatility'
            ]
            
            # Add enhanced features
            enhanced_cols = [
                'price_zscore_5d', 'price_zscore_20d',
                'momentum_1m', 'momentum_3m', 'momentum_6m',
                'vol_ratio', 'log_volume', 'volume_momentum',
                'bull_market', 'high_volatility_regime',
                'price_acceleration', 'momentum_divergence',
                'volume_price_trend', 'overbought', 'oversold',
                'ma_crossover_5_20', 'breakout_high'
            ]
            
            # Add these columns if they exist
            for col in enhanced_cols:
                if col in tech.columns:
                    feature_cols.append(col)
            
            # Create the combined feature set
            features_list = [tech[feature_cols]]
            
            # Only add non-empty dataframes
            if not macro.empty:
                features_list.append(macro)
            
            if not micro.empty:
                features_list.append(micro)
            
            # Combine all features into a single dataframe
            all_features = pd.concat(features_list, axis=1)
            
            # Check if the dataframe is empty and handle it
            if all_features.empty or len(all_features) == 0:
                print("WARNING: No features available in the specified date range!")
                # Create a default feature set with zeros
                default_features = pd.DataFrame(
                    np.zeros((1, len(all_features.columns) if not all_features.empty else len(feature_cols))), 
                    columns=all_features.columns if not all_features.empty else feature_cols,
                    index=[datetime.now()]
                )
                all_features = default_features
            
            # Drop any rows with missing values
            all_features = all_features.dropna()
            
            # Print feature information for debugging
            print(f"Combined features shape: {all_features.shape}")
            print(f"Feature columns: {all_features.columns.tolist()}")
            
            # Check again after dropping NaN values
            if all_features.empty or len(all_features) == 0:
                print("WARNING: All features were dropped due to NaN values. Creating default features.")
                default_features = pd.DataFrame(
                    np.zeros((1, len(feature_cols))), 
                    columns=feature_cols,
                    index=[datetime.now()]
                )
                all_features = default_features
            
            # Clean up any infinity or extremely large values before scaling
            all_features = all_features.replace([np.inf, -np.inf], np.nan)
            all_features = all_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Additional check for extremely large values
            for col in all_features.columns:
                # Replace values larger than a threshold with the column median
                extreme_mask = np.abs(all_features[col]) > 1e10
                if extreme_mask.any():
                    print(f"Replacing extreme values in column: {col}")
                    all_features.loc[extreme_mask, col] = all_features[col].median()
            
            # Clean up any infinity or extremely large values before scaling
            all_features = all_features.replace([np.inf, -np.inf], np.nan)
            
            # Print columns with NaN values to debug
            nan_columns = all_features.columns[all_features.isna().any()].tolist()
            if nan_columns:
                print(f"Columns with NaN values: {nan_columns}")
            
            # Fill NaN values
            all_features = all_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Replace extremely large values with column medians
            for col in all_features.columns:
                # Calculate column median for non-extreme values
                median_val = all_features[col][np.abs(all_features[col]) < 1e10].median()
                if pd.isna(median_val):
                    median_val = 0
                    
                # Replace extreme values
                extreme_mask = np.abs(all_features[col]) > 1e10
                extreme_count = extreme_mask.sum()
                if extreme_count > 0:
                    print(f"Replacing {extreme_count} extreme values in column: {col}")
                    all_features.loc[extreme_mask, col] = median_val
            
            print("Data preprocessing complete, proceeding with scaling...")
            
            # Now scale the cleaned data
            scaled = self.scaler.fit_transform(all_features)
            
            return scaled, all_features.index
        
        except Exception as e:
            print(f"ERROR in feature extraction: {e}")
            # Create default features in case of error
            print("Creating default feature set due to error...")
            feature_dim = 100  # Reasonable default
            default_features = np.zeros((1, feature_dim))
            default_dates = pd.DatetimeIndex([datetime.now()])
            return default_features, default_dates

    def get_enhanced_data(self, ticker, start_date, end_date, polygon_api_key):
        """Get enhanced alternative data sources"""
        
        # Get existing data
        market_data = self.get_technical_indicators(ticker, start_date, end_date)
        
        # Add options data if available
        try:
            options_data = self.get_options_metrics(ticker, start_date, end_date, polygon_api_key)
            market_data = pd.merge(
                market_data, 
                options_data, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
        except Exception as e:
            print(f"Could not retrieve options data: {e}")
        
        # Add sector rotation metrics
        try:
            sector_data = self.get_sector_performance(start_date, end_date)
            market_data = pd.merge(
                market_data, 
                sector_data, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
        except Exception as e:
            print(f"Could not retrieve sector data: {e}")
        
        # Fill missing values using appropriate methods
        market_data = self.handle_missing_data(market_data)
        
        return market_data

    def handle_missing_data(self, data):
        """Intelligent missing data handling"""
        # Handle missing values based on column type
        numeric_cols = data.select_dtypes(include='number').columns
        
        # For price-based metrics, use forward fill then backward fill
        price_cols = [col for col in numeric_cols if 'price' in col.lower() 
                      or 'close' in col.lower() or 'open' in col.lower() 
                      or 'high' in col.lower() or 'low' in col.lower()]
        
        for col in price_cols:
            data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        # For momentum and rate of change metrics, use 0
        momentum_cols = [col for col in numeric_cols if 'momentum' in col.lower() 
                        or 'change' in col.lower() or 'pct' in col.lower()]
        
        for col in momentum_cols:
            data[col] = data[col].fillna(0)
        
        # For volume metrics, use the median
        volume_cols = [col for col in numeric_cols if 'volume' in col.lower()]
        
        for col in volume_cols:
            data[col] = data[col].fillna(data[col].median())
        
        # Finally, replace any remaining NaNs with column medians
        for col in numeric_cols:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].median())
        
        return data

    def get_options_metrics(self, ticker, start_date, end_date, polygon_api_key):
        """
        Fetch options data from Polygon and calculate options-based metrics.
        This is an advanced method that requires options data access.
        """
        # This is a placeholder for a real implementation
        # In practice, you would fetch options chains and calculate metrics like:
        # - Put/Call ratio
        # - Implied volatility skew
        # - Options volume
        # - Open interest
        
        # For now, return an empty DataFrame
        return pd.DataFrame()
        
    def get_sector_performance(self, start_date, end_date):
        """
        Get sector rotation data by analyzing sector ETFs
        """
        # Sector ETFs tickers
        sectors = {
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLE': 'Energy',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLK': 'Technology',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        # This is a placeholder implementation
        # In practice, you would download data for all sectors,
        # calculate relative performance metrics
        
        return pd.DataFrame()


# --------------------- Model Definitions ---------------------
class ImprovedMarketModel(nn.Module):
    def __init__(self, market_feature_dim, news_embedding_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super(ImprovedMarketModel, self).__init__()
        self.market_feature_dim = market_feature_dim
        self.news_embedding_dim = news_embedding_dim
        self.hidden_dim = hidden_dim
        
        # Transformer encoder for time series
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=market_feature_dim,
            nhead=4 if market_feature_dim >= 8 else 1,  # Ensure nhead divides feature dimension
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced attention for news
        self.news_attention = nn.Sequential(
            nn.Linear(news_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Project news to match market features dimension first
        self.news_projection = nn.Linear(news_embedding_dim + 1, market_feature_dim)
        
        # Project both to hidden dimension separately
        self.market_projection = nn.Linear(market_feature_dim, hidden_dim)
        self.news_hidden_projection = nn.Linear(market_feature_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Main sequence processing with GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Hierarchical attention for time steps
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Multiple prediction heads
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # DOWN, NEUTRAL, UP
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, market_features, news_embeddings, news_sentiment):
        batch_size, seq_len, _ = market_features.size()
        
        # Apply transformer to market features
        market_encoded = self.transformer(market_features)
        
        # Process news with attention
        news_attention_weights = self.news_attention(news_embeddings)
        attended_news = torch.sum(news_embeddings * news_attention_weights, dim=1)
        
        # Include sentiment information
        avg_sentiment = torch.mean(news_sentiment, dim=1)
        news_with_sentiment = torch.cat([attended_news, avg_sentiment], dim=1)
        
        # Project news to match market features dimension
        news_projected = self.news_projection(news_with_sentiment)
        
        # Repeat for sequence length
        news_projected = news_projected.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Project both to hidden dimension
        market_hidden = self.market_projection(market_encoded)
        news_hidden = self.news_hidden_projection(news_projected)
        
        # Concatenate for gating
        combined = torch.cat([market_hidden, news_hidden], dim=2)
        gate_values = self.gate(combined)
        
        # Apply gating
        fused_features = market_hidden * gate_values + news_hidden * (1 - gate_values)
        
        # Process with GRU
        gru_out, _ = self.gru(fused_features)
        
        # Apply temporal attention
        time_weights = self.time_attention(gru_out).transpose(1, 2)
        context_vector = torch.bmm(time_weights, gru_out).squeeze(1)
        
        # Get predictions from multiple heads
        direction_logits = self.direction_head(context_vector)
        uncertainty = self.uncertainty_head(context_vector)
        
        return direction_logits, uncertainty

# --------------------- Original model for backward compatibility --------- 
class MarketPredictionModel(nn.Module):
    def __init__(self, market_feature_dim, news_embedding_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(MarketPredictionModel, self).__init__()
        self.market_feature_dim = market_feature_dim
        self.news_embedding_dim = news_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.news_attention = nn.Sequential(
            nn.Linear(news_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # +1 for sentiment dimension
        self.fusion_layer = nn.Sequential(
            nn.Linear(news_embedding_dim + 1 + market_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 classes: up, down, neutral
        )

    def forward(self, market_features, news_embeddings, news_sentiment):
        # market_features: (batch, seq_len, market_feature_dim)
        # news_embeddings: (batch, MAX_ARTICLES, embed_dim)
        # news_sentiment: (batch, MAX_ARTICLES, 1)

        batch_size, seq_len, _ = market_features.size()

        # Attention across the articles dimension
        news_attention_weights = self.news_attention(news_embeddings)  # (batch, MAX_ARTICLES, 1)
        attended_news = torch.sum(news_embeddings * news_attention_weights, dim=1)  # (batch, embed_dim)

        # Average sentiment
        attended_sentiment = torch.mean(news_sentiment, dim=1)  # (batch, 1)

        # Combine
        attended_news = torch.cat([attended_news, attended_sentiment], dim=1)  # (batch, embed_dim+1)

        # Repeat for each time step
        attended_news = attended_news.unsqueeze(1).repeat(1, seq_len, 1)

        combined_features = torch.cat([market_features, attended_news], dim=2)

        fused = self.fusion_layer(combined_features)  # (batch, seq_len, hidden_dim)

        lstm_out, _ = self.lstm(fused)  # (batch, seq_len, hidden_dim*2)

        predictions = self.output_layer(lstm_out)  # (batch, seq_len, 3)
        return predictions


# --------------------- Ensemble Model ---------------------
class EnsemblePredictor:
    """Ensemble of multiple models for better generalization"""
    def __init__(self, models_list, weights=None, voting='soft'):
        self.models = models_list
        self.voting = voting
        
        if weights is None:
            self.weights = [1/len(models_list)] * len(models_list)
        else:
            total = sum(weights)
            self.weights = [w/total for w in weights]
    
    def predict(self, market_feats, news_embeds, news_sents):
        self.models_output = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if isinstance(model, MarketPredictionModel):
                    # Original model
                    output = model(market_feats, news_embeds, news_sents)
                    output = output[:, -1, :]  # Take last time step
                else:
                    # Improved model
                    output, _ = model(market_feats, news_embeds, news_sents)
                
                self.models_output.append(output)
        
        if self.voting == 'hard':
            # Get class predictions from each model
            predictions = [torch.argmax(out, dim=1) for out in self.models_output]
            # Stack predictions and find mode (most common class)
            stacked = torch.stack(predictions)
            final_pred = torch.mode(stacked, dim=0).values
        else:
            # Weighted average of probabilities
            probs = [F.softmax(out, dim=1) for out in self.models_output]
            weighted_probs = [p * w for p, w in zip(probs, self.weights)]
            avg_prob = torch.sum(torch.stack(weighted_probs), dim=0)
            final_pred = torch.argmax(avg_prob, dim=1)
            
        return final_pred, avg_prob


# --------------------- PyTorch Dataset ---------------------
class TensorDataset(Dataset):
    def __init__(self, market_features, news_embeddings, news_sentiment, labels):
        self.market_features = market_features
        self.news_embeddings = news_embeddings
        self.news_sentiment = news_sentiment
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.market_features[idx],
            self.news_embeddings[idx],
            self.news_sentiment[idx],
            self.labels[idx]
        )


# --------------------- Loss Functions and Training Utilities ---------------------
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def hyperparameter_optimization(X_train, X_test, nf_train, nf_test, y_train, y_test, 
                               market_feature_dim, news_embedding_dim, model_type='improved', seed=42):
    """
    Perform hyperparameter optimization using Optuna
    
    Args:
        X_train, X_test: Market features for training and testing
        nf_train, nf_test: News features for training and testing  
        y_train, y_test: Target labels
        market_feature_dim: Dimension of market features
        news_embedding_dim: Dimension of news embeddings
        model_type: Which model architecture to use ('original' or 'improved')
        seed: Random seed for reproducibility
        
    Returns:
        dict: Best hyperparameters
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor([f['embeddings'] for f in nf_train], dtype=torch.float32),
        torch.tensor([f['sentiment_scores'] for f in nf_train], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor([f['embeddings'] for f in nf_test], dtype=torch.float32),
        torch.tensor([f['sentiment_scores'] for f in nf_test], dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    def objective(trial):
        # Define the hyperparameters to search
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 768])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        
        # Class weights
        down_weight = trial.suggest_float("down_weight", 0.5, 3.0)
        neutral_weight = trial.suggest_float("neutral_weight", 0.5, 3.0)
        up_weight = trial.suggest_float("up_weight", 0.5, 3.0)
        
        # Focal loss gamma parameter
        gamma = trial.suggest_float("gamma", 0.5, 5.0)
        
        # Print the current trial configuration
        print(f"Trial {trial.number}: Testing {hidden_dim=}, {num_layers=}, {dropout=}, {learning_rate=}, {batch_size=}")
        print(f"Class weights: DOWN={down_weight}, NEUTRAL={neutral_weight}, UP={up_weight}, Gamma={gamma}")
        
        # Create model with these hyperparameters
        if model_type == 'improved':
            model = ImprovedMarketModel(
                market_feature_dim=market_feature_dim,
                news_embedding_dim=news_embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            model = MarketPredictionModel(
                market_feature_dim=market_feature_dim,
                news_embedding_dim=news_embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        
        # Create dataloaders with the selected batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
        
        # Set up loss function with class weights
        class_weights = torch.tensor([down_weight, neutral_weight, up_weight], device=device)
        focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        
        # Set up optimizer with the selected learning rate
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Move model to device
        model.to(device)
        
        # Train (we'll do fewer epochs for the optimization)
        max_epochs = 30
        patience = 5  # Early stopping patience
        best_val_loss = float('inf')
        best_val_f1 = 0
        no_improve_epochs = 0
        
        # Early stopping tracker
        early_stopper = EarlyStopping(patience=patience, min_delta=0.005)
        
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]
                
                if model_type == 'improved':
                    logits, uncertainty = model(market_feats, news_embeds, news_sents)
                    loss = focal_loss(logits, label_batch)
                else:
                    outputs = model(market_feats, news_embeds, news_sents)
                    loss = focal_loss(outputs[:, -1, :], label_batch)
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}")
            
            # Validation
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]
                    
                    if model_type == 'improved':
                        logits, _ = model(market_feats, news_embeds, news_sents)
                        loss = focal_loss(logits, label_batch)
                        _, predicted = torch.max(logits, 1)
                    else:
                        outputs = model(market_feats, news_embeds, news_sents)
                        loss = focal_loss(outputs[:, -1, :], label_batch)
                        _, predicted = torch.max(outputs[:, -1, :], 1)
                    
                    val_loss += loss.item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(label_batch.cpu().numpy())
            
            avg_val_loss = val_loss / len(test_loader)
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            print(f"Validation Loss: {avg_val_loss:.4f}, F1 Score: {val_f1:.4f}")
            
            # Early stopping check
            early_stopper(1.0 - val_f1)  # We want to maximize F1, so we convert to a minimization problem
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save best model state based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
        
        # Final evaluation
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"Trial {trial.number} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        return f1  # Optimize for F1 score, which balances precision and recall

    # Run hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # Adjust number of trials based on time constraints
    
    # Print best parameters
    print("Best hyperparameters:", study.best_params)
    print("Best F1 score:", study.best_value)
    
    return study.best_params


# --------------------- Prediction Calibration ---------------------
def calibrate_predictions(model, val_loader, temperature=1.0):
    """Calibrate prediction probabilities and estimate uncertainty"""
    model.eval()
    
    # Setup calibration data
    val_probs = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]
            
            if isinstance(model, ImprovedMarketModel):
                logits, uncertainty = model(market_feats, news_embeds, news_sents)
            else:
                logits = model(market_feats, news_embeds, news_sents)[:, -1, :]
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=1)
            
            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(label_batch.cpu().numpy())
    
    # Calculate calibration metrics
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    
    # Expected Calibration Error
    confidences = np.max(val_probs, axis=1)
    predictions = np.argmax(val_probs, axis=1)
    accuracies = (predictions == val_labels)
    
    # Bin the confidences
    M = 10  # Number of bins
    bins = np.linspace(0, 1, M+1)
    
    # Calculate ECE
    ece = 0
    for m in range(M):
        in_bin = np.logical_and(confidences > bins[m], confidences <= bins[m+1])
        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * (np.sum(in_bin) / len(confidences))
    
    print(f"Expected Calibration Error: {ece:.4f}")
    
    # If ECE is high, adjust temperature
    if ece > 0.05:
        print("Calibration needed. Trying different temperature values...")
        best_ece = ece
        best_temp = temperature
        
        for temp in [0.5, 0.8, 1.2, 1.5, 2.0, 3.0]:
            temp_tensor = torch.tensor(temp, device=device)
            scaled_logits = logits / temp_tensor
            new_probs = F.softmax(scaled_logits, dim=1).cpu().numpy()
            new_confidences = np.max(new_probs, axis=1)
            
            # Calculate new ECE
            new_ece = 0
            for m in range(M):
                in_bin = np.logical_and(new_confidences > bins[m], new_confidences <= bins[m+1])
                if np.sum(in_bin) > 0:
                    accuracy_in_bin = np.mean(accuracies[in_bin])
                    confidence_in_bin = np.mean(new_confidences[in_bin])
                    new_ece += np.abs(accuracy_in_bin - confidence_in_bin) * (np.sum(in_bin) / len(new_confidences))
            
            if new_ece < best_ece:
                best_ece = new_ece
                best_temp = temp
        
        print(f"Best temperature: {best_temp:.2f}, ECE: {best_ece:.4f}")
        return best_temp
    
    return temperature


# --------------------- Kelly Position Sizing ---------------------
def kelly_position_sizing(prediction_probs, historical_win_rate=None, risk_aversion=0.5):
    """
    Calculate position size using the Kelly Criterion
    Args:
        prediction_probs: Model's probability predictions
        historical_win_rate: Optional historical win rate for calibration
        risk_aversion: Fraction of Kelly to use (0.5 = Half Kelly)
    
    Returns:
        optimal_fraction: Kelly optimal fraction of capital to risk
    """
    # Get the highest probability class and its probability
    max_prob = np.max(prediction_probs)
    predicted_class = np.argmax(prediction_probs)
    
    # Adjust probability based on historical model performance
    if historical_win_rate is not None:
        # Scale the raw probability by historical accuracy
        adjusted_prob = max_prob * historical_win_rate
    else:
        adjusted_prob = max_prob
    
    # Set odds based on the direction
    if predicted_class == 2:  # UP
        # Calculate asymmetric payoff (typically market moves up slowly)
        odds = 1.5  # Example: risk 1 to make 0.5
    elif predicted_class == 0:  # DOWN
        # Markets often fall faster than they rise
        odds = 2.0  # Example: risk 1 to make 1
    else:  # NEUTRAL
        # If prediction is neutral, use minimal position
        return 0.01
    
    # Kelly formula: f* = (bp - q) / b
    # where p = probability of winning, q = 1-p, b = odds
    win_prob = adjusted_prob
    lose_prob = 1 - win_prob
    
    kelly_fraction = (odds * win_prob - lose_prob) / odds
    
    # Apply a fraction of Kelly for safety (risk_aversion parameter)
    fractional_kelly = kelly_fraction * risk_aversion
    
    # Ensure position size is reasonable
    if fractional_kelly < 0:
        return 0  # No position if Kelly is negative
    elif fractional_kelly > 0.2:
        return 0.2  # Cap at 20% for risk management
    
    return fractional_kelly


# --------------------- Main Training + Backtest ---------------------
def train_and_backtest_model(ticker='SPY', start_date='2022-03-30', end_date='2025-03-29',
                             seq_length=10, max_articles=5, polygon_api_key=None,
                             historical_news_data=None, fred_api_key=None, 
                             run_optimization=False, model_type='improved'):
    """
    1) Fetch price data + technical indicators for the entire date range (must >= 50 days).
    2) Use provided historical news data or fetch news from Polygon API.
    3) Create 10-day sequences for LSTM. Each sequence has up to `max_articles` articles (padded/truncated).
    4) Perform a time-based 'backtest' by using the first 80% of sequences (chronologically) as train,
        and the final 20% as test.
    5) Train and evaluate the model.

    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        seq_length (int): Number of days in each sequence for LSTM
        max_articles (int): Maximum number of articles per sequence
        polygon_api_key (str): Polygon.io API key
        historical_news_data (dict): Optional pre-fetched historical news data
        fred_api_key (str): API key for FRED economic data
        run_optimization (bool): Whether to run hyperparameter optimization
        model_type (str): Type of model to use ('original' or 'improved')

    Returns:
        tuple: Trained model, accuracy, market features, news data, and dates
    """
    if fred_api_key is None:
        raise ValueError("FRED API key is required for macroeconomic data")

    # --- Extract Market Features ---
    market_extractor = MarketFeatureExtractor(fred_api_key=fred_api_key)
    market_features, dates = market_extractor.extract_features(ticker, start_date, end_date, polygon_api_key)
    print(f"Successfully extracted features for {len(dates)} trading days")

    # --- Use provided historical news or fetch news using Polygon API ---
    if historical_news_data:
        print("Using pre-fetched historical news data")
        # Filter the historical news to match our date range
        news_data = {d: historical_news_data.get(d, []) for d in dates}
    else:
        print("Fetching financial news using Polygon API...")
        news_data = {}

        # Initialize empty lists for all dates
        for d in dates:
            news_data[d] = []

        if polygon_api_key and polygon_api_key != "YOUR_ACTUAL_POLYGON_API_KEY_HERE":
            try:
                news_data = fetch_financial_news_polygon(ticker, start_date, end_date, dates, polygon_api_key)
            except Exception as e:
                print(f"Error during news fetching: {e}")
                print("Continuing with empty news data...")
        else:
            print("No valid Polygon API key provided. Proceeding without news data.")

    print(f"Successfully processed news for {len(news_data)} days")

    # --- Build labels (up=2, neutral=1, down=0) ---
    price_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    price_data = price_data[['Open','High','Low','Close','Volume']]
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = price_data.columns.droplevel(1)
    price_data['Close'] = price_data['Close'].squeeze()
    price_data = price_data.reindex(dates)

    atr = ta.volatility.average_true_range(price_data['High'], price_data['Low'], price_data['Close'], window=14)
    threshold = 0.5 * atr / price_data['Close']
    daily_returns = price_data['Close'].pct_change().fillna(0.0).values
    labels = np.zeros(len(daily_returns))
    labels[daily_returns > threshold] = 2  # UP
    labels[np.abs(daily_returns) <= threshold] = 1  # NEUTRAL
    labels[daily_returns < -threshold] = 0  # DOWN

    # --- Create Sequences ---
    X_sequences = []
    news_sequences = []
    y_sequences = []

    for i in range(len(market_features) - seq_length):
        X_sequences.append(market_features[i : i+seq_length])

        # Accumulate news from those sequence days
        seq_news = []
        for j in range(i, i+seq_length):
            d = dates[j]
            articles_for_day = news_data.get(d, [])
            seq_news.extend(articles_for_day)

        news_sequences.append(seq_news)
        y_sequences.append(labels[i+seq_length])  # label for the day after the sequence window

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    print("DEBUG: Total sequences:", len(X_sequences))
    if len(X_sequences) == 0:
        raise ValueError("No data sequences found. Possibly no valid data or an invalid date range.")

    # --- Convert news articles to FinBERT embeddings, then pad/truncate to max_articles ---
    news_extractor = FinBERTFeatureExtractor()
    news_features = []

    for seq_news in news_sequences:
        feats = news_extractor.extract_features(seq_news)
        emb = feats['embeddings']          # shape (n_articles, 768)
        snt = feats['sentiment_scores']    # shape (n_articles, 1)

        n_articles = emb.shape[0]

        # TRUNCATE or PAD to exactly max_articles
        if n_articles > max_articles:
            # Truncate
            emb = emb[:max_articles]
            snt = snt[:max_articles]
        elif n_articles < max_articles:
            # Pad
            pad_len = max_articles - n_articles
            emb = np.concatenate([emb, np.zeros((pad_len, 768))], axis=0)
            snt = np.concatenate([snt, 0.5 * np.ones((pad_len, 1))], axis=0)  # neutral sentiment for padded

        news_features.append({
            'embeddings': emb,             # shape (max_articles, 768)
            'sentiment_scores': snt        # shape (max_articles, 1)
        })

    # --- Time-based backtest split (80% train, 20% test) ---
    # We'll do a chronological split by index, not random.
    split_idx = int(len(X_sequences) * 0.8)

    X_train = X_sequences[:split_idx]
    X_test  = X_sequences[split_idx:]

    nf_train = news_features[:split_idx]
    nf_test  = news_features[split_idx:]

    y_train = y_sequences[:split_idx]
    y_test  = y_sequences[split_idx:]

    print("DEBUG: len(X_train) =", len(X_train), "len(X_test) =", len(X_test))
    if len(X_train) == 0:
        raise ValueError("No training samples in train split. Possibly the date range is too small.")

    # Check the shape of the first train embeddings
    print("DEBUG: First train embeddings shape:", nf_train[0]['embeddings'].shape)

    # --- Build the model ---
    market_feature_dim = X_train.shape[2]  # e.g. how many features after scaling
    news_embedding_dim = nf_train[0]['embeddings'].shape[1]  # 768 typically

    if run_optimization:
        print(f"Running hyperparameter optimization for {model_type} model...")
        best_params = hyperparameter_optimization(
            X_train, X_test, nf_train, nf_test, y_train, y_test,
            market_feature_dim, news_embedding_dim, model_type=model_type
        )
        
        hidden_dim = best_params["hidden_dim"]
        num_layers = best_params["num_layers"]
        dropout = best_params["dropout"]
        learning_rate = best_params["learning_rate"]
        batch_size = best_params["batch_size"]
        
        # Class weights
        down_weight = best_params["down_weight"]
        neutral_weight = best_params["neutral_weight"]
        up_weight = best_params["up_weight"]
        
        # Focal loss gamma
        gamma = best_params.get("gamma", 2.0)  # Default to 2.0 if not in params
        
        print(f"Using optimized parameters: {hidden_dim=}, {num_layers=}, {dropout=}, {learning_rate=}, {batch_size=}")
        print(f"Class weights: DOWN={down_weight}, NEUTRAL={neutral_weight}, UP={up_weight}, Gamma={gamma}")
    else:
        # Default parameters
        hidden_dim = 512 if model_type == 'improved' else 256
        num_layers = 2
        dropout = 0.3
        learning_rate = 0.001
        batch_size = 32
        down_weight = 2.0  # Give more weight to DOWN class
        neutral_weight = 1.5  # Give more weight to NEUTRAL class (often underrepresented)
        up_weight = 1.0
        gamma = 2.0  # Default focal loss gamma
    
    # --- Build the model based on type ---
    if model_type == 'improved':
        model = ImprovedMarketModel(
            market_feature_dim=market_feature_dim,
            news_embedding_dim=news_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        print("Using improved model architecture with Transformer + GRU")
    else:
        model = MarketPredictionModel(
            market_feature_dim=market_feature_dim,
            news_embedding_dim=news_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        print("Using original model architecture with LSTM")
    

    # --- Build PyTorch datasets ---
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor([f['embeddings'] for f in nf_train], dtype=torch.float32),
        torch.tensor([f['sentiment_scores'] for f in nf_train], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor([f['embeddings'] for f in nf_test], dtype=torch.float32),
        torch.tensor([f['sentiment_scores'] for f in nf_test], dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)

    # --- Train ---
    # Set up loss function with focal loss
    class_weights = torch.tensor([down_weight, neutral_weight, up_weight], device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=gamma)
    
    # Optimizer with weight decay and gradient clipping
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopper = EarlyStopping(patience=10, min_delta=0.005)

    # Move model to GPU
    model.to(device)

    # Setup mixed precision training if on GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    num_epochs = 100  # Increase epochs since we have early stopping
    start_time = time.time()
    
    best_val_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]

            # Use mixed precision for faster training if on GPU
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    if model_type == 'improved':
                        direction_logits, uncertainty = model(market_feats, news_embeds, news_sents)
                        loss = criterion(direction_logits, label_batch)
                    else:
                        outputs = model(market_feats, news_embeds, news_sents)
                        loss = criterion(outputs[:, -1, :], label_batch)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                if model_type == 'improved':
                    direction_logits, uncertainty = model(market_feats, news_embeds, news_sents)
                    loss = criterion(direction_logits, label_batch)
                else:
                    outputs = model(market_feats, news_embeds, news_sents)
                    loss = criterion(outputs[:, -1, :], label_batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]
                
                if model_type == 'improved':
                    direction_logits, _ = model(market_feats, news_embeds, news_sents)
                    _, predicted = torch.max(direction_logits, 1)
                else:
                    outputs = model(market_feats, news_embeds, news_sents)
                    _, predicted = torch.max(outputs[:, -1, :], 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(label_batch.cpu().numpy())
        
        # Calculate validation metrics
        val_accuracy = np.mean(np.array(val_preds) == np.array(val_labels))
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        # Update learning rate based on validation performance
        scheduler.step(val_f1)
        
        # Early stopping check
        early_stopper(1.0 - val_f1)  # We want to maximize F1, so invert for minimization
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"New best model with F1 score: {val_f1:.4f}")

        # Print GPU memory usage if available
        if torch.cuda.is_available():
            gpu_memory_allocated = round(torch.cuda.memory_allocated()/1024**3, 2)
            gpu_memory_reserved = round(torch.cuda.memory_reserved()/1024**3, 2)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_accuracy:.4f}, GPU Memory: {gpu_memory_allocated}GB allocated, {gpu_memory_reserved}GB reserved")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_accuracy:.4f}")
            
        # Check for early stopping
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model for evaluation")

    # --- Evaluate (backtest) ---
    model.eval()
    test_preds = []
    test_labels = []
    
    # For calibration, collect probability outputs
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]

            # Get predictions based on model type
            if model_type == 'improved':
                logits, uncertainty = model(market_feats, news_embeds, news_sents)
                probs = F.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
            else:
                outputs = model(market_feats, news_embeds, news_sents)
                probs = F.softmax(outputs[:, -1, :], dim=1)
                _, predicted = torch.max(outputs[:, -1, :], 1)

            # Store predictions, probabilities and true labels
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(label_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(test_preds) == np.array(test_labels))
    print(f"Backtest Accuracy (time-based split): {accuracy:.4f}")

    # Calculate precision & recall for each class
    print("\nDetailed classification report:")
    class_names = ["DOWN", "NEUTRAL", "UP"]
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Return everything including probabilities for calibration
    return model, accuracy, market_features, news_data, dates, (test_preds, test_labels, all_probs, test_loader)


# --------------------- Real-time Prediction ---------------------
def predict_next_day(model, ticker, last_n_days=10, max_articles=5, polygon_api_key=None, 
                    fred_api_key=None, temperature=1.0, model_type='improved'):
    """
    Make a prediction for the next trading day based on the most recent data.

    Args:
        model: Trained model (either MarketPredictionModel or ImprovedMarketModel)
        ticker: Stock ticker symbol
        last_n_days: Number of days to use for the sequence (should match training)
        max_articles: Maximum number of articles per sequence
        polygon_api_key: Polygon API key for news and microstructure data
        fred_api_key: FRED API key for macroeconomic data
        temperature: Temperature for calibrating predictions
        model_type: Type of model ('original' or 'improved')

    Returns:
        prediction: Class prediction (0=DOWN, 1=NEUTRAL, 2=UP)
        confidence: Confidence scores for each class
        uncertainty: Model uncertainty score
        latest_dates: Dates used for the prediction
    """
    if fred_api_key is None:
        raise ValueError("FRED API key is required for macroeconomic data")

    # Calculate date range for the last n days
    end_date = datetime.now()
    # Use a wider date range to ensure we have enough data
    start_date = end_date - timedelta(days=300)  # Get 300 days for indicators instead of 100

    # Format dates for API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Fetching latest {last_n_days} days of data for {ticker}...")

    try:
        # Get market features
        market_extractor = MarketFeatureExtractor(fred_api_key=fred_api_key)
        market_features, dates = market_extractor.extract_features(ticker, start_date_str, end_date_str, polygon_api_key)
        
        # Check if we got any valid data
        if len(market_features) == 0 or len(dates) == 0:
            print("WARNING: No features extracted. Creating default feature set.")
            # Create a default feature set (zeros) with the right dimensions
            if model_type == 'improved':
                market_feature_dim = model.market_feature_dim
            else:
                market_feature_dim = model.market_feature_dim
                
            market_features = np.zeros((last_n_days, market_feature_dim))
            dates = [end_date - timedelta(days=i) for i in range(last_n_days, 0, -1)]
    except Exception as e:
        print(f"Error extracting features: {e}")
        print("Creating default feature set...")
        if model_type == 'improved':
            market_feature_dim = model.market_feature_dim
        else:
            market_feature_dim = model.market_feature_dim
            
        market_features = np.zeros((last_n_days, market_feature_dim))
        dates = [end_date - timedelta(days=i) for i in range(last_n_days, 0, -1)]

    # Check if we have enough data
    if len(market_features) < last_n_days:
        print(f"WARNING: Not enough recent data available. Need {last_n_days} days, got {len(market_features)}")
        print("Padding with zeros...")
        # Pad with zeros at the beginning to reach the required sequence length
        padding_needed = last_n_days - len(market_features)
        if len(market_features) > 0:
            padding_shape = (padding_needed, market_features.shape[1])
            market_features = np.vstack([np.zeros(padding_shape), market_features])
            # Add padding dates
            first_date = dates[0] if len(dates) > 0 else end_date
            padding_dates = [first_date - timedelta(days=i+1) for i in range(padding_needed)]
            dates = padding_dates + (dates if isinstance(dates, list) else dates.tolist())
        else:
            # If we have no data at all, create a full sequence of zeros
            if model_type == 'improved':
                market_feature_dim = model.market_feature_dim
            else:
                market_feature_dim = model.market_feature_dim
                
            market_features = np.zeros((last_n_days, market_feature_dim))
            dates = [end_date - timedelta(days=i) for i in range(last_n_days, 0, -1)]

    # Get the most recent n days of features
    if len(market_features) > last_n_days:
        latest_features = market_features[-last_n_days:]
        if isinstance(dates, pd.DatetimeIndex):
            latest_dates = dates[-last_n_days:]
        else:
            latest_dates = dates[-last_n_days:]
    else:
        latest_features = market_features
        latest_dates = dates

    # Get the expected feature count from the model's market_feature_dim
    if model_type == 'improved':
        expected_feature_dim = model.market_feature_dim
    else:
        expected_feature_dim = model.market_feature_dim
    
    # Check if we have a mismatch
    if latest_features.shape[1] != expected_feature_dim:
        print(f"Feature dimension mismatch: got {latest_features.shape[1]}, expected {expected_feature_dim}")
        print("Adjusting feature dimensions to match training data...")
        
        # Create a properly sized array with zeros
        adjusted_features = np.zeros((latest_features.shape[0], expected_feature_dim))
        
        # Copy available features (use the minimum to avoid index errors)
        feat_count = min(latest_features.shape[1], expected_feature_dim)
        adjusted_features[:, :feat_count] = latest_features[:, :feat_count]
        
        # Replace original features with adjusted ones
        latest_features = adjusted_features
        print(f"Adjusted feature shape: {latest_features.shape}")

    # Fetch latest news
    print("Fetching latest news...")
    latest_news = []

    if polygon_api_key:
        try:
            # Convert dates to strings for API if needed
            if not isinstance(latest_dates[0], str):
                start_date_str = latest_dates[0].strftime('%Y-%m-%d') if hasattr(latest_dates[0], 'strftime') else str(latest_dates[0])
                end_date_str = latest_dates[-1].strftime('%Y-%m-%d') if hasattr(latest_dates[-1], 'strftime') else str(latest_dates[-1])
            else:
                start_date_str = latest_dates[0]
                end_date_str = latest_dates[-1]
                
            news_data = fetch_financial_news_polygon(
                ticker,
                start_date_str,
                end_date_str,
                pd.DatetimeIndex(latest_dates) if not isinstance(latest_dates, pd.DatetimeIndex) else latest_dates,
                polygon_api_key
            )

            # Accumulate all news across the days
            for date in latest_dates:
                latest_news.extend(news_data.get(date, []))

            print(f"Found {len(latest_news)} recent news articles")
        except Exception as e:
            print(f"Error fetching news: {e}")

    # Extract FinBERT features for the news
    news_extractor = FinBERTFeatureExtractor()
    news_feats = news_extractor.extract_features(latest_news)

    emb = news_feats['embeddings']
    snt = news_feats['sentiment_scores']

    # Pad or truncate to max_articles
    if emb.shape[0] > max_articles:
        emb = emb[:max_articles]
        snt = snt[:max_articles]
    elif emb.shape[0] < max_articles:
        pad_len = max_articles - emb.shape[0]
        emb = np.concatenate([emb, np.zeros((pad_len, 768))], axis=0)
        snt = np.concatenate([snt, 0.5*np.ones((pad_len, 1))], axis=0)

    # Convert to tensors and add batch dimension
    market_tensor = torch.tensor(latest_features.reshape(1, last_n_days, -1), dtype=torch.float32)
    news_embeds = torch.tensor(emb.reshape(1, max_articles, 768), dtype=torch.float32)
    news_sents = torch.tensor(snt.reshape(1, max_articles, 1), dtype=torch.float32)

    # Move to device
    market_tensor = market_tensor.to(device)
    news_embeds = news_embeds.to(device)
    news_sents = news_sents.to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        try:
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    if model_type == 'improved':
                        logits, uncertainty_score = model(market_tensor, news_embeds, news_sents)
                    else:
                        logits = model(market_tensor, news_embeds, news_sents)[:, -1, :]
                        uncertainty_score = torch.tensor([0.5], device=device)  # Placeholder for original model
                    
                    # Apply temperature scaling for calibration
                    scaled_logits = logits / temperature
                    probs = torch.nn.functional.softmax(scaled_logits, dim=1)[0]
                    _, pred = torch.max(scaled_logits, 1)
            else:
                if model_type == 'improved':
                    logits, uncertainty_score = model(market_tensor, news_embeds, news_sents)
                else:
                    logits = model(market_tensor, news_embeds, news_sents)[:, -1, :]
                    uncertainty_score = torch.tensor([0.5], device=device)  # Placeholder for original model
                    
                # Apply temperature scaling for calibration
                scaled_logits = logits / temperature
                probs = torch.nn.functional.softmax(scaled_logits, dim=1)[0]
                _, pred = torch.max(scaled_logits, 1)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            # Default to neutral with high uncertainty
            pred = torch.tensor([1], device=device)  # NEUTRAL
            probs = torch.tensor([0.2, 0.6, 0.2], device=device)  # Higher prob for NEUTRAL
            uncertainty_score = torch.tensor([0.8], device=device)  # High uncertainty

    # Convert to numpy
    prediction = pred.item()
    confidence = probs.cpu().numpy()
    uncertainty = uncertainty_score.cpu().numpy()[0] if hasattr(uncertainty_score, 'cpu') else 0.5

    # Map prediction to label
    label_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

    print(f"Prediction for next day: {label_map[prediction]}")
    print(f"Confidence: DOWN={confidence[0]:.2f}, NEUTRAL={confidence[1]:.2f}, UP={confidence[2]:.2f}")
    print(f"Model uncertainty: {uncertainty:.2f} (lower is better)")

    return prediction, confidence, uncertainty, latest_dates


# --------------------- Create Ensemble of Models ---------------------
def create_and_train_ensemble(X_train, X_test, nf_train, nf_test, y_train, y_test,
                             market_feature_dim, news_embedding_dim, num_models=3,
                             device=device):
    """
    Create and train an ensemble of models with different architectures
    
    Args:
        X_train, X_test: Market features for training and testing
        nf_train, nf_test: News features for training and testing
        y_train, y_test: Target labels
        market_feature_dim: Dimension of market features
        news_embedding_dim: Dimension of news embeddings
        num_models: Number of models in the ensemble
        device: Training device
        
    Returns:
        ensemble: Trained ensemble model
    """
    print(f"Creating ensemble of {num_models} models...")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor([f['embeddings'] for f in nf_train], dtype=torch.float32),
        torch.tensor([f['sentiment_scores'] for f in nf_train], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor([f['embeddings'] for f in nf_test], dtype=torch.float32),
        torch.tensor([f['sentiment_scores'] for f in nf_test], dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize models list
    models = []
    weights = []
    
    # Define different configurations for ensemble models
    configs = [
        # Model 1: Original LSTM with more layers
        {
            'type': 'original',
            'hidden_dim': 512,
            'num_layers': 3,
            'dropout': 0.3,
            'learning_rate': 0.001
        },
        # Model 2: Improved model with transformer
        {
            'type': 'improved',
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001
        },
        # Model 3: Improved model with focus on news
        {
            'type': 'improved',
            'hidden_dim': 384,
            'num_layers': 1,
            'dropout': 0.2,
            'learning_rate': 0.002
        }
    ]
    
    # Train each model
    for i, config in enumerate(configs[:num_models]):
        print(f"\nTraining model {i+1}/{num_models} with config: {config}")
        
        # Create model based on type
        if config['type'] == 'improved':
            model = ImprovedMarketModel(
                market_feature_dim=market_feature_dim,
                news_embedding_dim=news_embedding_dim,
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
        else:
            model = MarketPredictionModel(
                market_feature_dim=market_feature_dim,
                news_embedding_dim=news_embedding_dim,
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
            
        # Move model to device
        model.to(device)
        
        # Setup loss function (Focal Loss) and optimizer
        criterion = FocalLoss(
            alpha=torch.tensor([2.0, 1.5, 1.0], device=device),
            gamma=2.0
        )
        optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
        
        # Train for a fixed number of epochs
        epochs = 30
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]
                
                if config['type'] == 'improved':
                    logits, _ = model(market_feats, news_embeds, news_sents)
                    loss = criterion(logits, label_batch)
                else:
                    outputs = model(market_feats, news_embeds, news_sents)
                    loss = criterion(outputs[:, -1, :], label_batch)
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if (epoch + 1) % 5 == 0:
                model.eval()
                val_preds = []
                val_labels = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]
                        
                        if config['type'] == 'improved':
                            logits, _ = model(market_feats, news_embeds, news_sents)
                            _, predicted = torch.max(logits, 1)
                        else:
                            outputs = model(market_feats, news_embeds, news_sents)
                            _, predicted = torch.max(outputs[:, -1, :], 1)
                        
                        val_preds.extend(predicted.cpu().numpy())
                        val_labels.extend(label_batch.cpu().numpy())
                
                val_f1 = f1_score(val_labels, val_preds, average='weighted')
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Final evaluation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]
                
                if config['type'] == 'improved':
                    logits, _ = model(market_feats, news_embeds, news_sents)
                    _, predicted = torch.max(logits, 1)
                else:
                    outputs = model(market_feats, news_embeds, news_sents)
                    _, predicted = torch.max(outputs[:, -1, :], 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(label_batch.cpu().numpy())
        
        # Calculate F1 score
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"Model {i+1} final F1 score: {val_f1:.4f}")
        
        # Add model to ensemble with weight proportional to its F1 score
        models.append(model)
        weights.append(val_f1)
    
    # Create ensemble with weights proportional to F1 scores
    ensemble = EnsemblePredictor(models, weights=weights)
    print(f"Created ensemble with {len(models)} models")
    
    return ensemble


# --------------------- Example Usage ---------------------
if __name__ == "__main__":
    # The range 2022-03-30 to 2025-03-29 yields ~752 trading days
    start_date = "2022-03-30"
    end_date   = "2025-03-29"

    print(f"Running on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    print(f"Fetching data from {start_date} to {end_date} for backtest...")

    # API keys
    polygon_api_key = "uXdWonBRqeQUs_kvEmx7IhvO1ktP8LH1"
    fred_api_key = "e27aeafdd08f0e92830315de82579e94"

    # For production, use environment variables instead
    # import os
    # polygon_api_key = os.environ.get("POLYGON_API_KEY")
    # fred_api_key = os.environ.get("FRED_API_KEY")

    # If you want to fetch and cache historical news data for multiple runs
    print("\nFetching historical news for the past 3 years...")
    historical_news = fetch_historical_news_polygon("SPY", years_back=3, api_key=polygon_api_key)

    # Count total articles fetched
    total_articles = sum(len(articles) for articles in historical_news.values())
    print(f"Successfully fetched {total_articles} historical news articles for SPY")

    # Set model type and whether to run hyperparameter optimization
    model_type = 'improved'  # Use 'improved' or 'original'
    run_optimization = True  # Set to False after finding optimal parameters
    
    # Train the model and get evaluation results
    model, accuracy, market_feats, all_news, all_dates, evaluation_data = train_and_backtest_model(
        ticker="SPY",
        start_date=start_date,
        end_date=end_date,
        seq_length=10,
        max_articles=5,
        polygon_api_key=polygon_api_key,
        historical_news_data=historical_news,
        fred_api_key=fred_api_key,
        run_optimization=run_optimization,
        model_type=model_type
    )

    print(f"\nFinal Backtest Accuracy: {accuracy:.4f}")
    
    # Unpack evaluation data
    test_preds, test_labels, test_probs, test_loader = evaluation_data
    
    # Calibrate model predictions
    print("\nCalibrating model predictions...")
    temperature = calibrate_predictions(model, test_loader, temperature=1.0)
    print(f"Using temperature scaling with T={temperature:.2f}")

    # Make a prediction for the next trading day
    print("\nPredicting next trading day...")
    prediction, confidence, uncertainty, dates = predict_next_day(
        model=model,
        ticker="SPY",
        last_n_days=10,
        max_articles=5,
        polygon_api_key=polygon_api_key,
        fred_api_key=fred_api_key,
        temperature=temperature,
        model_type=model_type
    )
    
    # Calculate position size using Kelly Criterion
    historical_accuracy = accuracy  # Use backtest accuracy as a proxy
    position_size = kelly_position_sizing(confidence, historical_accuracy, risk_aversion=0.5)
    print(f"Recommended position size: {position_size:.2%} of capital")
    
    # Print trading recommendation
    label_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
    print("\n=== TRADING RECOMMENDATION ===")
    print(f"Direction: {label_map[prediction]}")
    print(f"Confidence: {confidence[prediction]:.2f}")
    print(f"Uncertainty: {uncertainty:.2f}")
    print(f"Position size: {position_size:.2%} of capital")
    print("==============================")

    # Save model (optional)
    model_path = f"market_prediction_model_{model_type}_{datetime.now().strftime('%Y%m%d')}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved as {model_path}")
