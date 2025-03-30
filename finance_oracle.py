import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
            'RETAILSMNSA' # Retail Sales
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
                'volume_trend_5d', 'volume_trend_10d', 'volume_trend_20d'
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

        # Fill missing values
        micro_data = micro_data.fillna(method='ffill').fillna(0)

        return micro_data

    def extract_features(self, ticker, start_date, end_date, polygon_api_key):
        """Extract and combine all features for the model"""
        # Get data from different sources
        tech = self.get_technical_indicators(ticker, start_date, end_date)
        macro = self.get_macroeconomic_data(start_date, end_date)
        micro = self.get_market_microstructure(ticker, start_date, end_date, polygon_api_key)

        # Ensure all dataframes have the same index (trading days)
        common_idx = tech.index

        # Reindex macro and micro data to match technical data
        if not macro.empty:
            macro = macro.reindex(common_idx, method='ffill')

        if not micro.empty:
            micro = micro.reindex(common_idx, method='ffill')

        # Basic technical features
        feature_cols = [
            'MA5', 'MA20', 'MA50',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
            'BB_High', 'BB_Low', 'BB_Mid',
            'Volume_Change', 'Volume_MA5',
            'Price_Change', 'Price_Change_5d', 'Volatility'
        ]

        # Create the combined feature set
        features_list = [tech[feature_cols]]

        # Only add non-empty dataframes
        if not macro.empty:
            features_list.append(macro)

        if not micro.empty:
            features_list.append(micro)

        # Combine all features into a single dataframe
        all_features = pd.concat(features_list, axis=1)

        # Drop any rows with missing values
        all_features = all_features.dropna()

        # Print feature information for debugging
        print(f"Combined features shape: {all_features.shape}")
        print(f"Feature columns: {all_features.columns.tolist()}")

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


# --------------------- Model ---------------------
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


# --------------------- Main Training + Backtest ---------------------
def train_and_backtest_model(ticker='SPY', start_date='2022-03-30', end_date='2025-03-29',
                                seq_length=10, max_articles=5, polygon_api_key=None,
                                historical_news_data=None, fred_api_key=None):
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

    threshold = 0.001
    daily_returns = price_data['Close'].pct_change().fillna(0.0).values
    labels = np.zeros(len(daily_returns))
    labels[daily_returns > threshold] = 2
    labels[np.abs(daily_returns) <= threshold] = 1
    labels[daily_returns < -threshold] = 0

    # --- Create Sequences ---
    X_sequences = []
    news_sequences = []
    y_sequences = []

    for i in range(len(market_features) - seq_length):
        X_sequences.append(market_features[i : i+seq_length])

        # Accumulate news from those 10 days
        seq_news = []
        for j in range(i, i+seq_length):
            d = dates[j]
            articles_for_day = news_data.get(d, [])
            seq_news.extend(articles_for_day)

        news_sequences.append(seq_news)
        y_sequences.append(labels[i+seq_length])  # label for the day after the 10-day window

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

    # --- Build the LSTM model ---
    market_feature_dim = X_train.shape[2]  # e.g. how many features after scaling
    news_embedding_dim = nf_train[0]['embeddings'].shape[1]  # 768 typically

    model = MarketPredictionModel(
        market_feature_dim=market_feature_dim,
        news_embedding_dim=news_embedding_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )

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

    # Increase batch size for H100 GPU
    batch_size = 128 if torch.cuda.is_available() else 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)

    # --- Train ---
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001)

    # Move model to GPU
    model.to(device)

    # Setup mixed precision training if on GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    num_epochs = 30
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]

            # Use mixed precision for faster training if on GPU
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(market_feats, news_embeds, news_sents)
                    # Use only the last time-step
                    loss = criterion(outputs[:, -1, :], label_batch)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(market_feats, news_embeds, news_sents)
                # Use only the last time-step
                loss = criterion(outputs[:, -1, :], label_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Print GPU memory usage if available
        if torch.cuda.is_available():
            gpu_memory_allocated = round(torch.cuda.memory_allocated()/1024**3, 2)
            gpu_memory_reserved = round(torch.cuda.memory_reserved()/1024**3, 2)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, GPU Memory: {gpu_memory_allocated}GB allocated, {gpu_memory_reserved}GB reserved")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # --- Evaluate (backtest) ---
    model.eval()
    correct = 0
    total = 0

    # Track predictions and true labels for more detailed evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            market_feats, news_embeds, news_sents, label_batch = [b.to(device) for b in batch]

            # Use mixed precision for inference as well
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(market_feats, news_embeds, news_sents)
            else:
                outputs = model(market_feats, news_embeds, news_sents)

            # last step
            _, predicted = torch.max(outputs[:, -1, :], 1)

            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())

            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()

    # Calculate accuracy
    accuracy = correct / total
    print(f"Backtest Accuracy (time-based split): {accuracy:.4f}")

    # Calculate precision & recall for each class
    from sklearn.metrics import classification_report
    print("\nDetailed classification report:")
    class_names = ["DOWN", "NEUTRAL", "UP"]
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Return everything in case you want further analysis
    return model, accuracy, market_features, news_data, dates, (all_preds, all_labels)


# --------------------- Real-time Prediction ---------------------
def predict_next_day(model, ticker, last_n_days=10, max_articles=5, polygon_api_key=None, fred_api_key=None):
    """
    Make a prediction for the next trading day based on the most recent data.

    Args:
        model: Trained MarketPredictionModel
        ticker: Stock ticker symbol
        last_n_days: Number of days to use for the sequence (should match training)
        max_articles: Maximum number of articles per sequence
        polygon_api_key: Polygon API key for news and microstructure data
        fred_api_key: FRED API key for macroeconomic data

    Returns:
        prediction: Class prediction (0=DOWN, 1=NEUTRAL, 2=UP)
        confidence: Confidence scores for each class
    """
    if fred_api_key is None:
        raise ValueError("FRED API key is required for macroeconomic data")

    # Calculate date range for the last n days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)  # Get 100 days for indicators

    # Format dates for API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Fetching latest {last_n_days} days of data for {ticker}...")

    # Get market features
    market_extractor = MarketFeatureExtractor(fred_api_key=fred_api_key)
    market_features, dates = market_extractor.extract_features(ticker, start_date_str, end_date_str, polygon_api_key)

    if len(market_features) < last_n_days:
        raise ValueError(f"Not enough recent data available. Need {last_n_days} days, got {len(market_features)}")

    # Get the most recent n days of features
    latest_features = market_features[-last_n_days:]
    latest_dates = dates[-last_n_days:]

    # Fetch latest news
    print("Fetching latest news...")
    latest_news = []

    if polygon_api_key:
        try:
            news_data = fetch_financial_news_polygon(
                ticker,
                latest_dates[0].strftime('%Y-%m-%d'),
                latest_dates[-1].strftime('%Y-%m-%d'),
                latest_dates,
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
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(market_tensor, news_embeds, news_sents)
        else:
            outputs = model(market_tensor, news_embeds, news_sents)

        # Get prediction for last step
        probs = torch.nn.functional.softmax(outputs[:, -1, :], dim=1)[0]
        _, pred = torch.max(outputs[:, -1, :], 1)

    # Convert to numpy
    prediction = pred.item()
    confidence = probs.cpu().numpy()

    # Map prediction to label
    label_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

    print(f"Prediction for next day: {label_map[prediction]}")
    print(f"Confidence: DOWN={confidence[0]:.2f}, NEUTRAL={confidence[1]:.2f}, UP={confidence[2]:.2f}")

    return prediction, confidence, latest_dates


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

    # Run model training with all data sources
    model, accuracy, market_feats, all_news, all_dates, evaluation = train_and_backtest_model(
        ticker="SPY",
        start_date=start_date,
        end_date=end_date,
        seq_length=10,
        max_articles=5,
        polygon_api_key=polygon_api_key,
        historical_news_data=historical_news,
        fred_api_key=fred_api_key
    )

    print(f"\nFinal Backtest Accuracy: {accuracy:.4f}")

    # Make a prediction for the next trading day
    print("\nPredicting next trading day...")
    prediction, confidence, dates = predict_next_day(
        model=model,
        ticker="SPY",
        last_n_days=10,
        max_articles=5,
        polygon_api_key=polygon_api_key,
        fred_api_key=fred_api_key
    )

    # Save model (optional)
    torch.save(model.state_dict(), f"market_prediction_model_{datetime.now().strftime('%Y%m%d')}.pt")
    print("\nModel saved successfully.")
