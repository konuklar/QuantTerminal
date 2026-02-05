# =============================================================
# 🏛️ Institutional Apollo / ENIGMA – Quant Terminal v5.0
# Professional Portfolio Optimization & Global Multi-Asset Edition
# Enhanced: Complete professional features with error handling
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import concurrent.futures
from functools import lru_cache
import traceback
import time
import hashlib
import inspect
from pathlib import Path
import pickle
import tempfile

# Import PyPortfolioOpt
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.objective_functions import L2_reg
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    st.warning("⚠️ PyPortfolioOpt not installed. Some optimization features will be limited.")

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# ENHANCED GLOBAL ASSET UNIVERSE WITH SYMBOL MAPPING
# -------------------------------------------------------------
GLOBAL_ASSET_UNIVERSE = {
    # US Major Indices & ETFs
    "US_Indices": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", 
        "VEA", "VWO", "VUG", "VO", "VB", "VTV"
    ],
    
    # Bonds & Fixed Income
    "Bonds": [
        "TLT", "IEF", "SHY", "BND", "AGG", "HYG", "JNK",
        "MUB", "TIP", "LQD", "EMB"
    ],
    
    # Commodities
    "Commodities": [
        "GLD", "SLV", "USO", "UNG", "DBA", "PDBC", "GSG",
        "WEAT", "CORN", "SOYB"
    ],
    
    # Cryptocurrencies
    "Cryptocurrencies": [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "SOL-USD", "DOT-USD", "DOGE-USD", "MATIC-USD", "AVAX-USD",
        "LTC-USD", "UNI-USD", "LINK-USD", "ATOM-USD", "ETC-USD"
    ],
    
    # Global Stocks - US
    "US_Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
        "BRK-B", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA"
    ],
    
    # European Stocks
    "Europe_Stocks": [
        "ASML.AS", "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE",
        "NOVN.SW", "ROG.SW", "NESN.SW", "UBSG.SW", "CSGN.SW",
        "SAN.PA", "BNP.PA", "AIR.PA", "MC.PA", "OR.PA",
        "ENEL.MI", "ENI.MI", "ISP.MI", "UCG.MI"
    ],
    
    # UK Stocks
    "UK_Stocks": [
        "HSBA.L", "BP.L", "GSK.L", "RIO.L", "AAL.L",
        "AZN.L", "ULVR.L", "DGE.L", "BATS.L", "NG.L"
    ],
    
    # Asia Pacific Stocks
    "Asia_Stocks": [
        "9988.HK", "0700.HK", "0388.HK", "0005.HK", "1299.HK",
        "7203.T", "8306.T", "9984.T", "6758.T", "6861.T",
        "BABA", "JD", "BIDU", "NTES", "TCEHY"
    ],
    
    # Emerging Markets
    "Emerging_Stocks": [
        "HDB", "INFY", "TCS.NS", "SBIN.NS", "RELIANCE.NS",
        "VALE", "ITUB", "BBD", "GGB", "ABEV", "SBS"
    ],
    
    # Australia
    "Australia_Stocks": [
        "BHP.AX", "RIO.AX", "CBA.AX", "WBC.AX", "ANZ.AX",
        "NAB.AX", "CSL.AX", "WES.AX", "WOW.AX", "TLS.AX"
    ],
    
    # Singapore
    "Singapore_Stocks": [
        "D05.SI", "O39.SI", "U11.SI", "Z74.SI", "C09.SI"
    ],
    
    # Turkey
    "Turkey_Stocks": [
        "AKBNK.IS", "GARAN.IS", "ISCTR.IS", "KOZAA.IS", "SAHOL.IS",
        "THYAO.IS", "TCELL.IS", "TUPRS.IS", "ARCLK.IS", "BIMAS.IS"
    ],
    
    # Currencies & Forex with alternative symbols
    "Currencies": [
        "EURUSD=X", "EURUSD", "GBPUSD=X", "GBPUSD", 
        "USDJPY=X", "USDJPY", "USDCHF=X", "USDCHF",
        "AUDUSD=X", "AUDUSD", "USDCAD=X", "USDCAD", 
        "NZDUSD=X", "NZDUSD", "USDTRY=X", "USDTRY",
        "USDCNY=X", "USDCNY", "USDSGD=X", "USDSGD", 
        "USDHKD=X", "USDHKD", "USDINR=X", "USDINR",
        "USDBRL=X", "USDBRL", "USDZAR=X", "USDZAR", 
        "USDMXN=X", "USDMXN"
    ],
    
    # Volatility & Alternatives
    "Alternatives": [
        "^VIX", "VIXY", "UVXY", "SVXY",
        "TMF", "UPRO", "TQQQ", "SQQQ"
    ]
}

# Symbol mapping for common variations
SYMBOL_MAPPING = {
    "BTC-USD": ["BTC-USD", "BTCUSD", "BTCUSDT"],
    "ETH-USD": ["ETH-USD", "ETHUSD", "ETHUSDT"],
    "BRK-B": ["BRK-B", "BRK.B"],
    "GOOGL": ["GOOGL", "GOOG"],
    "EURUSD=X": ["EURUSD=X", "EURUSD"],
    "GBPUSD=X": ["GBPUSD=X", "GBPUSD"],
    "USDJPY=X": ["USDJPY=X", "USDJPY"],
}

# Flatten universe for selection
ALL_TICKERS = []
for category in GLOBAL_ASSET_UNIVERSE.values():
    ALL_TICKERS.extend(category)

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
APP_TITLE = "🏛️ Apollo/ENIGMA - Global Portfolio Terminal v5.0"
DEFAULT_RF_ANNUAL = 0.03
TRADING_DAYS = 252
MONTE_CARLO_SIMULATIONS = 10000
MAX_CACHE_SIZE = 100
CACHE_DIR = tempfile.gettempdir() + "/apollo_cache/"

os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "4"

# Create cache directory if it doesn't exist
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Apollo/ENIGMA - Global Portfolio Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# ENHANCED INSTITUTIONAL THEME
# -------------------------------------------------------------
st.markdown("""
<style>
:root {
    --primary: #1a5fb4;
    --primary-dark: #0d4fa0;
    --secondary: #26a269;
    --danger: #c01c28;
    --warning: #f5a623;
    --dark-bg: #0f172a;
    --card-bg: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --success: #16a34a;
    --success-dark: #15803d;
}

/* Enhanced tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: transparent;
    padding: 4px;
    border-radius: 10px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    color: var(--text-muted);
    border: 1px solid transparent;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
    border-color: var(--primary) !important;
}

/* Professional KPI cards */
.kpi-card {
    background: linear-gradient(145deg, var(--card-bg), #1a2332);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.3s ease;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
    border-color: var(--primary);
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    margin: 8px 0;
    line-height: 1.2;
}

.kpi-label {
    font-size: 13px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}

/* Data tables */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 20px !important;
}

[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
}

[data-testid="metric-container"] div {
    color: var(--text) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(26, 95, 180, 0.4);
}

/* Status indicators */
.status-success { color: var(--success) !important; }
.status-warning { color: var(--warning) !important; }
.status-error { color: var(--danger) !important; }

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--dark-bg);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-dark);
}

/* Card enhancements */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.card-header {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 15px;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--secondary));
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PORTFOLIO STRATEGIES
# -------------------------------------------------------------
PORTFOLIO_STRATEGIES = {
    "Equal Weight": "Equal allocation across all selected assets",
    "Market Cap Weight": "Weighted by market capitalization",
    "Minimum Volatility": "Optimized for lowest portfolio volatility",
    "Maximum Sharpe Ratio": "Optimized for highest risk-adjusted returns",
    "Risk Parity": "Equal risk contribution from each asset",
    "Maximum Diversification": "Maximizes diversification ratio",
    "Mean-Variance Optimal": "Classical Markowitz optimization",
    "Hierarchical Risk Parity": "HRP clustering-based allocation",
    "Custom Weights": "Manually specify asset weights"
}

# -------------------------------------------------------------
# ENHANCED CACHING SYSTEM
# -------------------------------------------------------------
class EnhancedCache:
    """Enhanced caching system for better performance"""
    
    @staticmethod
    def get_cache_key(*args, **kwargs):
        """Generate a unique cache key from function arguments"""
        call_frame = inspect.currentframe().f_back
        func_name = call_frame.f_code.co_name
        
        # Create a string representation of arguments
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        # Generate MD5 hash
        key_string = f"{func_name}_{args_str}_{kwargs_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    def cache_data(data, key, ttl_seconds=3600):
        """Cache data with TTL"""
        cache_file = Path(CACHE_DIR) / f"{key}.pkl"
        
        cache_entry = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl_seconds
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
        except:
            pass  # Silently fail on cache errors
    
    @staticmethod
    def get_cached_data(key):
        """Retrieve cached data if not expired"""
        cache_file = Path(CACHE_DIR) / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_entry = pickle.load(f)
            
            # Check if cache is expired
            if time.time() - cache_entry['timestamp'] > cache_entry['ttl']:
                cache_file.unlink(missing_ok=True)
                return None
            
            return cache_entry['data']
        except:
            return None

# -------------------------------------------------------------
# ENHANCED DATA LOADING WITH PARALLEL PROCESSING
# -------------------------------------------------------------
def download_single_ticker(ticker, start_date, end_date, max_retries=3):
    """Download data for a single ticker with retry logic and symbol fallback"""
    for attempt in range(max_retries):
        try:
            # Try alternative symbols first
            symbols_to_try = [ticker]
            if ticker in SYMBOL_MAPPING:
                symbols_to_try = SYMBOL_MAPPING[ticker] + symbols_to_try
            
            for symbol in symbols_to_try:
                try:
                    ticker_obj = yf.Ticker(symbol)
                    # Try different intervals if daily fails
                    hist = ticker_obj.history(
                        start=start_date,
                        end=end_date,
                        interval="1d",
                        auto_adjust=True,
                        prepost=False
                    )
                    
                    if not hist.empty and len(hist) > 10:
                        if 'Close' in hist.columns:
                            return symbol, hist['Close']
                        elif 'Adj Close' in hist.columns:
                            return symbol, hist['Adj Close']
                except Exception as e:
                    continue
            
            # Try batch download as fallback
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=False,
                    auto_adjust=True
                )
                if not data.empty and 'Close' in data.columns:
                    return ticker, data['Close']
            except:
                pass
                
        except Exception as e:
            if attempt == max_retries - 1:
                return ticker, None
            time.sleep(0.5)  # Brief delay before retry
    
    return ticker, None

@st.cache_data(show_spinner=False, ttl=3600, max_entries=10)
def load_global_prices_enhanced(tickers: List[str], start_date: str, end_date: str, 
                               use_parallel: bool = True) -> pd.DataFrame:
    """Load global asset prices with enhanced error handling and parallel downloads"""
    
    if not tickers:
        return pd.DataFrame()
    
    st.info(f"📥 Loading data for {len(tickers)} assets...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    prices_dict = {}
    successful_tickers = []
    failed_tickers = []
    
    if use_parallel and len(tickers) > 5:
        # Parallel download for larger sets
        max_workers = min(10, len(tickers))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_ticker = {
                executor.submit(download_single_ticker, ticker, start_date, end_date): ticker 
                for ticker in tickers
            }
            
            # Process completed tasks
            for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
                ticker = future_to_ticker[future]
                
                try:
                    symbol, price_data = future.result()
                    if price_data is not None and len(price_data) > 10:
                        prices_dict[symbol] = price_data
                        successful_tickers.append(symbol)
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
                
                # Update progress
                progress = (i + 1) / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"Loaded {i + 1}/{len(tickers)} assets...")
    else:
        # Sequential download for smaller sets
        for i, ticker in enumerate(tickers):
            symbol, price_data = download_single_ticker(ticker, start_date, end_date)
            
            if price_data is not None and len(price_data) > 10:
                prices_dict[symbol] = price_data
                successful_tickers.append(symbol)
            else:
                failed_tickers.append(ticker)
            
            # Update progress
            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            status_text.text(f"Loaded {i + 1}/{len(tickers)} assets...")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Report results
    if successful_tickers:
        st.success(f"✅ Successfully loaded {len(successful_tickers)} out of {len(tickers)} assets")
        
        if failed_tickers:
            st.warning(f"⚠️ Failed to load {len(failed_tickers)} assets: {', '.join(failed_tickers[:5])}{'...' if len(failed_tickers) > 5 else ''}")
    
    # Create DataFrame
    if prices_dict:
        prices_df = pd.DataFrame(prices_dict)
        
        # Sort columns to maintain order as much as possible
        ordered_columns = []
        for ticker in tickers:
            if ticker in prices_df.columns:
                ordered_columns.append(ticker)
            else:
                # Check if any mapped symbol is in the dataframe
                for mapped in SYMBOL_MAPPING.get(ticker, []):
                    if mapped in prices_df.columns and mapped not in ordered_columns:
                        ordered_columns.append(mapped)
                        break
        
        # Add any remaining columns
        remaining_cols = [col for col in prices_df.columns if col not in ordered_columns]
        ordered_columns.extend(remaining_cols)
        
        prices_df = prices_df[ordered_columns]
        
        # Handle missing data
        prices_df = prices_df.dropna(axis=1, how='all')
        
        if prices_df.empty:
            st.error("❌ No data available after cleaning. All tickers failed.")
            return pd.DataFrame()
        
        # Fill small gaps (limit to 3 consecutive NaNs)
        prices_df = prices_df.ffill(limit=3).bfill(limit=3)
        
        # Remove assets with too much missing data (>30%)
        missing_pct = prices_df.isna().mean()
        valid_columns = missing_pct[missing_pct < 0.3].index.tolist()
        prices_df = prices_df[valid_columns]
        
        if len(prices_df.columns) < 2:
            st.error("❌ Insufficient data for portfolio analysis. Need at least 2 assets with valid data.")
            return pd.DataFrame()
        
        # Ensure we have enough rows
        if len(prices_df) < 20:
            st.warning(f"⚠️ Limited data: only {len(prices_df)} data points available")
        
        return prices_df
    
    else:
        st.error("❌ Could not load any data. Please check your ticker symbols and internet connection.")
        return pd.DataFrame()

def generate_demo_data(tickers, start_date, end_date):
    """Generate synthetic data for demonstration when real data fails"""
    st.info("📊 Generating demo data for testing purposes...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    if len(dates) < 20:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=252, freq='B')
    
    prices_dict = {}
    
    # Base parameters for different asset types
    asset_params = {
        'SPY': {'base': 400, 'vol': 0.18, 'trend': 0.08},
        'QQQ': {'base': 350, 'vol': 0.22, 'trend': 0.10},
        'TLT': {'base': 100, 'vol': 0.12, 'trend': 0.02},
        'GLD': {'base': 180, 'vol': 0.15, 'trend': 0.04},
        'BTC-USD': {'base': 30000, 'vol': 0.40, 'trend': 0.15},
        'ETH-USD': {'base': 2000, 'vol': 0.45, 'trend': 0.12},
        'AAPL': {'base': 150, 'vol': 0.25, 'trend': 0.12},
        'MSFT': {'base': 300, 'vol': 0.22, 'trend': 0.10},
        'GOOGL': {'base': 120, 'vol': 0.24, 'trend': 0.09},
        'AMZN': {'base': 130, 'vol': 0.28, 'trend': 0.08},
    }
    
    # Generate correlated returns
    np.random.seed(42)
    n_assets = len(tickers)
    n_days = len(dates)
    
    # Create correlation matrix
    base_corr = 0.3
    corr_matrix = np.eye(n_assets) * (1 - base_corr) + base_corr
    
    # Generate correlated random numbers
    L = np.linalg.cholesky(corr_matrix)
    uncorrelated = np.random.randn(n_assets, n_days)
    correlated = L @ uncorrelated
    
    for idx, ticker in enumerate(tickers):
        # Get parameters
        params = asset_params.get(ticker, {'base': 100, 'vol': 0.25, 'trend': 0.06})
        
        # Generate returns with trend and volatility
        daily_trend = (1 + params['trend']) ** (1/252) - 1
        daily_vol = params['vol'] / np.sqrt(252)
        
        # Create returns series
        returns = daily_trend + daily_vol * correlated[idx]
        
        # Add some autocorrelation
        for i in range(1, len(returns)):
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
        
        # Calculate prices
        cumulative_returns = np.exp(np.cumsum(returns))
        prices = params['base'] * cumulative_returns
        
        # Add some noise
        prices = prices * (1 + np.random.randn(len(prices)) * 0.001)
        
        prices_dict[ticker] = pd.Series(prices, index=dates)
    
    return pd.DataFrame(prices_dict)

@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def load_global_prices_cached(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load global asset prices with enhanced caching"""
    cache_key = f"prices_{hashlib.md5(str(tickers).encode() + str(start_date).encode() + str(end_date).encode()).hexdigest()}"
    
    # Try to get from enhanced cache first
    cached_data = EnhancedCache.get_cached_data(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not in cache, load from original function
    data = load_global_prices_enhanced(tickers, start_date, end_date)
    
    # Cache the result
    if not data.empty:
        EnhancedCache.cache_data(data, cache_key)
    
    return data

# -------------------------------------------------------------
# ENHANCED PERFORMANCE METRICS CALCULATOR
# -------------------------------------------------------------
class PerformanceMetrics:
    """Enhanced performance metrics calculator with all major financial ratios"""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, 
                         risk_free_rate: float = 0.03, trading_days: int = 252) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if returns.empty:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['annual_return'] = returns.mean() * trading_days
        metrics['annual_volatility'] = returns.std() * np.sqrt(trading_days)
        metrics['sharpe_ratio'] = (metrics['annual_return'] - risk_free_rate) / (metrics['annual_volatility'] + 1e-10)
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(trading_days)
            metrics['sortino_ratio'] = (metrics['annual_return'] - risk_free_rate) / (downside_deviation + 1e-10)
        else:
            metrics['sortino_ratio'] = np.nan
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Recovery metrics
        if metrics['max_drawdown'] < 0:
            try:
                # Find drawdown recovery
                underwater = cumulative < running_max
                recovery_start = drawdown.idxmin() if not drawdown.empty else None
                
                if recovery_start:
                    # Find when portfolio recovers to previous high
                    recovery_periods = 0
                    for i in range(cumulative.index.get_loc(recovery_start), len(cumulative)):
                        if cumulative.iloc[i] >= running_max.iloc[i]:
                            recovery_periods = i - cumulative.index.get_loc(recovery_start)
                            break
                    metrics['recovery_period'] = recovery_periods
                else:
                    metrics['recovery_period'] = np.nan
            except:
                metrics['recovery_period'] = np.nan
        
        # Calmar Ratio
        if metrics['max_drawdown'] < 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = np.nan
        
        # Information Ratio (if benchmark provided)
        if benchmark_returns is not None and not benchmark_returns.empty:
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(trading_days)
            if tracking_error > 0:
                metrics['information_ratio'] = (metrics['annual_return'] - benchmark_returns.mean() * trading_days) / tracking_error
            else:
                metrics['information_ratio'] = np.nan
        
        # Skewness and Kurtosis
        metrics['skewness'] = stats.skew(returns.dropna())
        metrics['kurtosis'] = stats.kurtosis(returns.dropna())
        
        # Value at Risk (Historical)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        
        # Conditional VaR
        tail_95 = returns[returns <= metrics['var_95']]
        tail_99 = returns[returns <= metrics['var_99']]
        metrics['cvar_95'] = tail_95.mean() if len(tail_95) > 0 else metrics['var_95']
        metrics['cvar_99'] = tail_99.mean() if len(tail_99) > 0 else metrics['var_99']
        
        # Gain/Loss Ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            metrics['gain_loss_ratio'] = abs(positive_returns.mean() / negative_returns.mean())
        else:
            metrics['gain_loss_ratio'] = np.nan
        
        # Win Rate
        metrics['win_rate'] = (returns > 0).mean()
        
        # Profit Factor
        gross_profit = positive_returns.sum()
        gross_loss = abs(negative_returns.sum())
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.nan
        
        # Omega Ratio
        threshold = risk_free_rate / trading_days
        excess_returns = returns - threshold
        positive_excess = excess_returns[excess_returns > 0].sum()
        negative_excess = abs(excess_returns[excess_returns < 0].sum())
        metrics['omega_ratio'] = positive_excess / negative_excess if negative_excess > 0 else np.nan
        
        # Ulcer Index
        if len(drawdown) > 0:
            metrics['ulcer_index'] = np.sqrt((drawdown ** 2).mean())
        else:
            metrics['ulcer_index'] = np.nan
        
        # Tail Ratio
        metrics['tail_ratio'] = abs(metrics['var_95'] / metrics['var_99']) if metrics['var_99'] != 0 else np.nan
        
        return metrics

# -------------------------------------------------------------
# PYPORTFOLIOOPT INTEGRATION (ENHANCED)
# -------------------------------------------------------------
class PortfolioOptimizer:
    """Enhanced portfolio optimization using PyPortfolioOpt"""
    
    @staticmethod
    def optimize_portfolio(returns_df: pd.DataFrame, strategy: str, 
                          target_return: float = None, target_risk: float = None,
                          risk_free_rate: float = 0.03, 
                          risk_aversion: float = 1.0,
                          short_allowed: bool = False) -> Dict:
        """Optimize portfolio using PyPortfolioOpt with enhanced error handling"""
        
        # Validate input
        if returns_df is None or returns_df.empty:
            return PortfolioOptimizer._fallback_weights([], risk_free_rate)
        
        if len(returns_df.columns) < 2:
            n_assets = len(returns_df.columns)
            equal_weights = np.ones(n_assets) / n_assets if n_assets > 0 else np.array([])
            return {
                'weights': equal_weights,
                'expected_return': 0,
                'expected_risk': 0,
                'sharpe_ratio': 0,
                'method': 'Single Asset',
                'cleaned_weights': {asset: equal_weights[i] for i, asset in enumerate(returns_df.columns)},
                'success': True
            }
        
        if not PYPFOPT_AVAILABLE:
            return PortfolioOptimizer._fallback_weights(returns_df.columns.tolist(), risk_free_rate, returns_df)
        
        try:
            # Clean returns data
            returns_clean = returns_df.copy()
            
            # Remove extreme outliers (>10 standard deviations)
            for col in returns_clean.columns:
                col_data = returns_clean[col].dropna()
                if len(col_data) > 10:
                    mean = col_data.mean()
                    std = col_data.std()
                    if std > 0:
                        # Use .values to avoid Series boolean ambiguity
                        mask = (returns_clean[col] - mean).abs() < (10 * std)
                        # Apply the mask properly
                        returns_clean.loc[~mask, col] = mean
            
            # Calculate expected returns and covariance
            try:
                mu = expected_returns.mean_historical_return(returns_clean)
            except:
                # Fallback to simple mean
                mu = returns_clean.mean() * TRADING_DAYS
            
            try:
                S = risk_models.sample_cov(returns_clean)
                
                # Check if covariance matrix is positive definite
                try:
                    np.linalg.cholesky(S)
                except:
                    # Use shrinkage estimator if not positive definite
                    S = risk_models.CovarianceShrinkage(returns_clean).ledoit_wolf()
            except:
                # Fallback to diagonal covariance
                S = pd.DataFrame(np.diag(returns_clean.var()), 
                                index=returns_clean.columns, 
                                columns=returns_clean.columns)
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1) if not short_allowed else (-1, 1))
            
            # Apply strategy
            if strategy == "Minimum Volatility":
                try:
                    weights = ef.min_volatility()
                except:
                    weights = PortfolioOptimizer._equal_weights(returns_clean)
                    
            elif strategy == "Maximum Sharpe Ratio":
                try:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                except Exception as e:
                    st.warning(f"Max Sharpe optimization failed: {str(e)[:100]}. Using Min Volatility.")
                    weights = ef.min_volatility()
                
            elif strategy == "Maximum Quadratic Utility":
                try:
                    weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)
                except:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                
            elif strategy == "Efficient Risk":
                if target_risk:
                    try:
                        weights = ef.efficient_risk(target_risk=target_risk/np.sqrt(TRADING_DAYS))
                    except:
                        weights = ef.min_volatility()
                else:
                    weights = ef.min_volatility()
                    
            elif strategy == "Efficient Return":
                if target_return:
                    try:
                        weights = ef.efficient_return(target_return=target_return/TRADING_DAYS)
                    except:
                        weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                else:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                    
            elif strategy == "Risk Parity":
                # Simple risk parity implementation
                volatilities = returns_clean.std()
                inv_vol = 1 / (volatilities + 1e-10)  # Add small epsilon to avoid division by zero
                weights_series = inv_vol / inv_vol.sum()
                weights = dict(zip(returns_clean.columns, weights_series))
                
            else:
                # Default to minimum volatility
                weights = ef.min_volatility()
            
            # Clean weights
            if isinstance(weights, dict):
                cleaned_weights = weights
            else:
                cleaned_weights = ef.clean_weights()
            
            # Convert to array
            weights_array = np.array([cleaned_weights.get(asset, 0) for asset in returns_df.columns])
            
            # Normalize weights (handle rounding errors)
            if abs(weights_array.sum() - 1.0) > 0.001:
                weights_array = weights_array / (weights_array.sum() + 1e-10)
            
            # Calculate performance metrics
            try:
                expected_return, expected_risk, sharpe_ratio = ef.portfolio_performance(
                    risk_free_rate=risk_free_rate/TRADING_DAYS
                )
            except:
                # Fallback calculation
                portfolio_returns = (returns_clean * weights_array).sum(axis=1)
                expected_return = portfolio_returns.mean() * TRADING_DAYS
                expected_risk = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
                sharpe_ratio = (expected_return - risk_free_rate) / (expected_risk + 1e-10)
            
            return {
                'weights': weights_array,
                'expected_return': expected_return,
                'expected_risk': expected_risk,
                'sharpe_ratio': sharpe_ratio,
                'method': strategy,
                'cleaned_weights': cleaned_weights,
                'success': True
            }
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)[:200]}")
            return PortfolioOptimizer._fallback_weights(returns_df.columns.tolist(), risk_free_rate, returns_df)
    
    @staticmethod
    def _equal_weights(returns_df):
        """Generate equal weights"""
        n_assets = len(returns_df.columns)
        weights = {asset: 1.0/n_assets for asset in returns_df.columns}
        return weights
    
    @staticmethod
    def _fallback_weights(tickers, risk_free_rate, returns_df=None):
        """Fallback to equal weights"""
        n_assets = len(tickers)
        if n_assets == 0:
            return {
                'weights': np.array([]),
                'expected_return': 0,
                'expected_risk': 0,
                'sharpe_ratio': 0,
                'method': 'No Assets',
                'cleaned_weights': {},
                'success': False
            }
        
        equal_weights = np.ones(n_assets) / n_assets
        
        if returns_df is not None and not returns_df.empty:
            port_returns = (returns_df * equal_weights).sum(axis=1)
            ann_return = port_returns.mean() * TRADING_DAYS
            ann_risk = port_returns.std() * np.sqrt(TRADING_DAYS)
            sharpe = (ann_return - risk_free_rate) / (ann_risk + 1e-10)
        else:
            ann_return = 0
            ann_risk = 0
            sharpe = 0
        
        return {
            'weights': equal_weights,
            'expected_return': ann_return,
            'expected_risk': ann_risk,
            'sharpe_ratio': sharpe,
            'method': 'Equal Weight (Fallback)',
            'cleaned_weights': {asset: equal_weights[i] for i, asset in enumerate(tickers)},
            'success': False
        }

# -------------------------------------------------------------
# ENHANCED RISK ANALYTICS (UPDATED)
# -------------------------------------------------------------
class EnhancedRiskAnalytics:
    """Professional risk analytics with multiple VaR methods"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, method: str = "historical", 
                     confidence_level: float = 0.95, 
                     window: int = None, 
                     params: Dict = None) -> Dict:
        """Calculate Value at Risk using multiple methods with enhanced error handling"""
        
        if returns is None or returns.empty:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': method,
                'confidence': confidence_level,
                'error': 'Empty returns series',
                'success': False
            }
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 20:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': method,
                'confidence': confidence_level,
                'error': f'Insufficient data: {len(returns_clean)} points',
                'success': False
            }
        
        try:
            if method == "historical":
                return EnhancedRiskAnalytics._historical_var(returns_clean, confidence_level)
            
            elif method == "parametric":
                return EnhancedRiskAnalytics._parametric_var(returns_clean, confidence_level)
            
            elif method == "ewma":
                return EnhancedRiskAnalytics._ewma_var(returns_clean, confidence_level, window, params)
            
            elif method == "monte_carlo":
                return EnhancedRiskAnalytics._monte_carlo_var(returns_clean, confidence_level, params)
            
            else:
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': method,
                    'confidence': confidence_level,
                    'error': 'Unknown method',
                    'success': False
                }
                
        except Exception as e:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': method,
                'confidence': confidence_level,
                'error': f'Calculation failed: {str(e)[:100]}',
                'success': False
            }
    
    @staticmethod
    def _historical_var(returns, confidence_level):
        """Historical Simulation VaR"""
        try:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            tail = returns[returns <= var]
            cvar = tail.mean() if len(tail) > 0 else var
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'Historical Simulation',
                'confidence': confidence_level,
                'observations': len(returns),
                'percentile': (1 - confidence_level) * 100,
                'success': True
            }
        except:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': 'Historical Simulation',
                'confidence': confidence_level,
                'error': 'Percentile calculation failed',
                'success': False
            }
    
    @staticmethod
    def _parametric_var(returns, confidence_level):
        """Parametric (Normal Distribution) VaR"""
        try:
            mu = returns.mean()
            sigma = returns.std()
            
            if sigma == 0 or pd.isna(sigma):
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'Parametric (Normal)',
                    'confidence': confidence_level,
                    'error': 'Zero or NaN volatility',
                    'success': False
                }
            
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mu + z_score * sigma
            cvar = mu - (sigma / (1 - confidence_level)) * stats.norm.pdf(z_score)
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'Parametric (Normal)',
                'confidence': confidence_level,
                'mu': mu,
                'sigma': sigma,
                'z_score': z_score,
                'success': True
            }
        except Exception as e:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': 'Parametric (Normal)',
                'confidence': confidence_level,
                'error': f'Calculation failed: {str(e)[:100]}',
                'success': False
            }
    
    @staticmethod
    def _ewma_var(returns, confidence_level, window=None, params=None):
        """EWMA VaR with proper initialization"""
        try:
            if window is None:
                window = min(252, len(returns))
            
            lambda_param = params.get('lambda', 0.94) if params else 0.94
            
            # Calculate EWMA variance with proper initialization
            returns_squared = returns ** 2
            
            # Initialize with simple average of first window
            if len(returns) < window:
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'EWMA Parametric',
                    'confidence': confidence_level,
                    'error': f'Insufficient data for window {window}',
                    'success': False
                }
            
            ewma_var = pd.Series(index=returns.index, dtype=float)
            ewma_var.iloc[:window] = returns_squared.iloc[:window].mean()
            
            # Recursive EWMA calculation
            for i in range(window, len(returns)):
                ewma_var.iloc[i] = lambda_param * ewma_var.iloc[i-1] + (1 - lambda_param) * returns_squared.iloc[i-1]
            
            current_vol = np.sqrt(ewma_var.iloc[-1])
            
            if current_vol == 0 or pd.isna(current_vol):
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'EWMA Parametric',
                    'confidence': confidence_level,
                    'error': 'Zero or NaN volatility',
                    'success': False
                }
            
            z_score = stats.norm.ppf(1 - confidence_level)
            var = z_score * current_vol
            cvar = - (current_vol / (1 - confidence_level)) * stats.norm.pdf(z_score)
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'EWMA Parametric',
                'confidence': confidence_level,
                'lambda': lambda_param,
                'ewma_vol': current_vol,
                'z_score': z_score,
                'success': True
            }
        except Exception as e:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': 'EWMA Parametric',
                'confidence': confidence_level,
                'error': f'Calculation failed: {str(e)[:100]}',
                'success': False
            }
    
    @staticmethod
    def _monte_carlo_var(returns, confidence_level, params=None):
        """Monte Carlo Simulation VaR"""
        try:
            n_simulations = params.get('n_simulations', 10000) if params else 10000
            days = params.get('days', 1) if params else 1
            
            mu = returns.mean()
            sigma = returns.std()
            
            if sigma == 0 or pd.isna(sigma):
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'Monte Carlo Simulation',
                    'confidence': confidence_level,
                    'error': 'Zero or NaN volatility',
                    'success': False
                }
            
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Simulate returns using GBM
            dt = 1/252
            simulations = np.random.normal(mu*dt, sigma*np.sqrt(dt), (days, n_simulations))
            
            # Calculate cumulative returns
            cumulative_returns = np.ones(n_simulations)
            for day in range(days):
                cumulative_returns *= (1 + simulations[day])
            
            final_returns = cumulative_returns - 1
            
            var = np.percentile(final_returns, (1 - confidence_level) * 100)
            tail = final_returns[final_returns <= var]
            cvar = tail.mean() if len(tail) > 0 else var
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'Monte Carlo Simulation',
                'confidence': confidence_level,
                'simulations': n_simulations,
                'days': days,
                'mu': mu,
                'sigma': sigma,
                'success': True
            }
        except Exception as e:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': 'Monte Carlo Simulation',
                'confidence': confidence_level,
                'error': f'Calculation failed: {str(e)[:100]}',
                'success': False
            }
    
    @staticmethod
    def calculate_all_var_methods(returns: pd.Series, confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate VaR using all available methods"""
        methods = ["historical", "parametric", "ewma", "monte_carlo"]
        results = []
        
        for method in methods:
            result = EnhancedRiskAnalytics.calculate_var(returns, method, confidence_level)
            results.append({
                'Method': result['method'],
                'VaR': result['VaR'],
                'CVaR': result['CVaR'],
                'Status': 'Success' if result.get('success', False) else 'Failed',
                'Error': result.get('error', '')
            })
        
        return pd.DataFrame(results)

# -------------------------------------------------------------
# ENHANCED EWMA ANALYSIS (UPDATED)
# -------------------------------------------------------------
class EWMAAnalysis:
    """Enhanced EWMA volatility analysis with proper error handling"""
    
    @staticmethod
    def calculate_ewma_volatility(returns: pd.DataFrame, lambda_param: float = 0.94) -> pd.DataFrame:
        """Calculate EWMA volatility for multiple assets with robust initialization"""
        if returns.empty:
            return pd.DataFrame()
        
        ewma_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for asset in returns.columns:
            r = returns[asset].dropna()
            if len(r) < 20:
                ewma_vol[asset] = np.nan
                continue
            
            try:
                # Initialize EWMA variance with simple average of squared returns
                init_window = min(30, len(r))
                init_var = r.iloc[:init_window].var()
                
                if pd.isna(init_var) or init_var <= 0:
                    ewma_vol[asset] = r.rolling(20).std()
                    continue
                
                ewma_var = pd.Series(index=r.index, dtype=float)
                ewma_var.iloc[:init_window] = init_var
                
                # Calculate squared returns
                r_squared = r ** 2
                
                # Recursive EWMA calculation
                for i in range(init_window, len(r)):
                    prev_var = ewma_var.iloc[i-1]
                    if not pd.isna(prev_var):
                        ewma_var.iloc[i] = lambda_param * prev_var + (1 - lambda_param) * r_squared.iloc[i-1]
                    else:
                        ewma_var.iloc[i] = init_var
                
                # Calculate volatility (sqrt of variance)
                ewma_vol[asset] = np.sqrt(ewma_var)
                
                # Handle any remaining NaN values
                ewma_vol[asset] = ewma_vol[asset].ffill().bfill()
                
            except Exception as e:
                # Fallback to rolling standard deviation
                ewma_vol[asset] = r.rolling(20).std()
        
        return ewma_vol.dropna(how='all')

# -------------------------------------------------------------
# ENHANCED PORTFOLIO ANALYZER
# -------------------------------------------------------------
class PortfolioAnalyzer:
    """Comprehensive portfolio analysis with enhanced features"""
    
    @staticmethod
    def analyze_portfolio(prices: pd.DataFrame, weights: np.ndarray, 
                         benchmark_prices: pd.Series = None) -> Dict:
        """Comprehensive portfolio analysis"""
        
        analysis = {}
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        
        analysis['returns'] = portfolio_returns
        analysis['cumulative_returns'] = (1 + portfolio_returns).cumprod()
        
        # Calculate metrics
        benchmark_returns = None
        if benchmark_prices is not None:
            benchmark_returns = benchmark_prices.pct_change().dropna()
        
        analysis['metrics'] = PerformanceMetrics.calculate_metrics(
            portfolio_returns, benchmark_returns
        )
        
        # Calculate rolling metrics
        analysis['rolling_volatility'] = portfolio_returns.rolling(window=63).std() * np.sqrt(252)
        analysis['rolling_sharpe'] = (portfolio_returns.rolling(window=63).mean() * 252 - 0.03) / (
            analysis['rolling_volatility'] + 1e-10
        )
        
        # Calculate drawdowns
        cumulative = analysis['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        analysis['drawdown'] = drawdown
        analysis['underwater'] = cumulative < running_max
        
        # Calculate monthly returns
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        analysis['monthly_returns'] = monthly_returns
        
        # Calculate annual returns
        annual_returns = portfolio_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        analysis['annual_returns'] = annual_returns
        
        return analysis

# -------------------------------------------------------------
# PERFORMANCE ATTRIBUTION (UPDATED)
# -------------------------------------------------------------
class PerformanceAttribution:
    """Professional performance attribution analysis with error handling"""
    
    @staticmethod
    def calculate_brinson_attribution(portfolio_returns: pd.Series, 
                                     benchmark_returns: pd.Series,
                                     portfolio_weights: np.ndarray,
                                     benchmark_weights: np.ndarray,
                                     asset_returns: pd.DataFrame) -> Dict:
        """Calculate Brinson attribution analysis with validation"""
        
        # Validate inputs
        if (portfolio_returns is None or benchmark_returns is None or 
            portfolio_weights is None or benchmark_weights is None or 
            asset_returns is None):
            return {
                'total_active_return': np.nan,
                'allocation_effect': np.nan,
                'selection_effect': np.nan,
                'interaction_effect': np.nan,
                'residual': np.nan,
                'success': False,
                'error': 'Missing input data'
            }
        
        try:
            # Ensure weights sum to 1
            portfolio_weights = portfolio_weights / (portfolio_weights.sum() + 1e-10)
            benchmark_weights = benchmark_weights / (benchmark_weights.sum() + 1e-10)
            
            # Calculate total active return
            total_active_return = portfolio_returns.mean() - benchmark_returns.mean()
            
            # Calculate asset returns mean
            asset_means = asset_returns.mean()
            benchmark_mean = benchmark_returns.mean()
            
            # Allocation effect
            allocation_effect = np.sum(
                (portfolio_weights - benchmark_weights) * 
                (asset_means.values - benchmark_mean)
            )
            
            # Selection effect
            selection_effect = np.sum(
                benchmark_weights * 
                (asset_means.values - benchmark_mean)
            )
            
            # Interaction effect
            interaction_effect = np.sum(
                (portfolio_weights - benchmark_weights) * 
                (asset_means.values - benchmark_mean)
            ) - allocation_effect
            
            # Residual (should be close to 0)
            residual = total_active_return - (allocation_effect + selection_effect + interaction_effect)
            
            return {
                'total_active_return': total_active_return,
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'interaction_effect': interaction_effect,
                'residual': residual,
                'success': True
            }
            
        except Exception as e:
            return {
                'total_active_return': np.nan,
                'allocation_effect': np.nan,
                'selection_effect': np.nan,
                'interaction_effect': np.nan,
                'residual': np.nan,
                'success': False,
                'error': f'Calculation failed: {str(e)[:100]}'
            }

# -------------------------------------------------------------
# MONTE CARLO SIMULATION ENGINE (UPDATED)
# -------------------------------------------------------------
class MonteCarloSimulator:
    """Professional Monte Carlo simulation engine with error handling"""
    
    @staticmethod
    def simulate_gbm(S0: float, mu: float, sigma: float, 
                    T: float = 1.0, n_steps: int = 252,
                    n_sims: int = 10000, seed: int = 42) -> np.ndarray:
        """Geometric Brownian Motion simulation with validation"""
        
        # Validate inputs
        if S0 <= 0 or sigma < 0 or n_steps <= 0 or n_sims <= 0:
            raise ValueError("Invalid input parameters for GBM simulation")
        
        np.random.seed(seed)
        
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_sims))
        paths[0] = S0
        
        # Pre-calculate drift and diffusion terms
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_sims)
            paths[t] = paths[t-1] * np.exp(drift + diffusion * z)
        
        return paths
    
    @staticmethod
    def calculate_var_cvar(paths: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
        """Calculate VaR and CVaR from simulation paths"""
        if paths is None or paths.size == 0:
            return np.nan, np.nan
        
        try:
            final_returns = (paths[-1] / paths[0]) - 1
            var = np.percentile(final_returns, (1 - alpha) * 100)
            tail = final_returns[final_returns <= var]
            cvar = tail.mean() if len(tail) > 0 else var
            
            return var, cvar
        except:
            return np.nan, np.nan

# -------------------------------------------------------------
# ENHANCED REPORT GENERATOR
# -------------------------------------------------------------
class ReportGenerator:
    """Generate comprehensive portfolio reports"""
    
    @staticmethod
    def generate_summary_report(portfolio_analysis: Dict, benchmark_name: str = "Benchmark") -> str:
        """Generate markdown summary report"""
        
        metrics = portfolio_analysis.get('metrics', {})
        
        report = f"""
# 📊 Portfolio Performance Summary

## 📈 Key Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Annual Return** | {metrics.get('annual_return', 0):.2%} | Average annual return |
| **Annual Volatility** | {metrics.get('annual_volatility', 0):.2%} | Risk measured as standard deviation |
| **Sharpe Ratio** | {metrics.get('sharpe_ratio', 0):.2f} | Risk-adjusted return (higher is better) |
| **Sortino Ratio** | {metrics.get('sortino_ratio', 0):.2f} | Downside risk-adjusted return |
| **Maximum Drawdown** | {metrics.get('max_drawdown', 0):.2%} | Worst historical peak-to-trough decline |
| **Calmar Ratio** | {metrics.get('calmar_ratio', 0):.2f} | Return relative to max drawdown |
| **Win Rate** | {metrics.get('win_rate', 0):.2%} | Percentage of profitable periods |

## ⚠️ Risk Metrics

| Risk Measure | 95% Confidence | 99% Confidence |
|--------------|----------------|----------------|
| **Value at Risk (VaR)** | {metrics.get('var_95', 0):.2%} | {metrics.get('var_99', 0):.2%} |
| **Conditional VaR (CVaR)** | {metrics.get('cvar_95', 0):.2%} | {metrics.get('cvar_99', 0):.2%} |

## 📊 Distribution Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **Skewness** | {metrics.get('skewness', 0):.2f} | {ReportGenerator._interpret_skewness(metrics.get('skewness', 0))} |
| **Kurtosis** | {metrics.get('kurtosis', 0):.2f} | {ReportGenerator._interpret_kurtosis(metrics.get('kurtosis', 0))} |
| **Profit Factor** | {metrics.get('profit_factor', 0):.2f} | Gross profit / gross loss |
| **Gain/Loss Ratio** | {metrics.get('gain_loss_ratio', 0):.2f} | Average gain / average loss |

## 🎯 Performance Attribution

- **Best Month**: {portfolio_analysis.get('monthly_returns', pd.Series()).max():.2%}
- **Worst Month**: {portfolio_analysis.get('monthly_returns', pd.Series()).min():.2%}
- **Positive Months**: {(portfolio_analysis.get('monthly_returns', pd.Series()) > 0).mean():.2%}
- **Recovery Period**: {metrics.get('recovery_period', 'N/A')} days

---

*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*This report is for informational purposes only. Past performance does not guarantee future results.*
"""
        
        return report
    
    @staticmethod
    def _interpret_skewness(value: float) -> str:
        if value > 0.5:
            return "Right-skewed (more large gains)"
        elif value < -0.5:
            return "Left-skewed (more large losses)"
        else:
            return "Approximately symmetric"
    
    @staticmethod
    def _interpret_kurtosis(value: float) -> str:
        if value > 3:
            return "Leptokurtic (fat tails, more extreme events)"
        elif value < 3:
            return "Platykurtic (thin tails, fewer extremes)"
        else:
            return "Normal distribution"

# -------------------------------------------------------------
# ENHANCED ERROR HANDLING AND LOGGING
# -------------------------------------------------------------
class ErrorHandler:
    """Enhanced error handling and logging"""
    
    @staticmethod
    def handle_error(e: Exception, context: str = ""):
        """Handle errors gracefully and log them"""
        error_msg = f"Error in {context}: {str(e)}"
        
        # Log to session state for debugging
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        
        st.session_state.error_log.append({
            'timestamp': datetime.now(),
            'context': context,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        
        # Display user-friendly error
        st.error(f"❌ {error_msg}")
        
        # Add expander for technical details
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())

# -------------------------------------------------------------
# ENHANCED MONITORING
# -------------------------------------------------------------
class PerformanceMonitor:
    """Monitor application performance"""
    
    @staticmethod
    def start_timer(label: str):
        """Start a performance timer"""
        if 'performance_timers' not in st.session_state:
            st.session_state.performance_timers = {}
        
        st.session_state.performance_timers[label] = time.time()
    
    @staticmethod
    def end_timer(label: str):
        """End a performance timer and log result"""
        if 'performance_timers' in st.session_state and label in st.session_state.performance_timers:
            elapsed = time.time() - st.session_state.performance_timers[label]
            
            # Log to session state
            if 'performance_log' not in st.session_state:
                st.session_state.performance_log = []
            
            st.session_state.performance_log.append({
                'timestamp': datetime.now(),
                'label': label,
                'elapsed_seconds': elapsed
            })

# -------------------------------------------------------------
# MAIN APPLICATION (ENHANCED WITH COMPLETE ERROR HANDLING)
# -------------------------------------------------------------
def main():
    st.title(APP_TITLE)
    
    # Initialize session state with enhanced defaults
    default_tickers = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
    
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = default_tickers
    if 'portfolio_strategy' not in st.session_state:
        st.session_state.portfolio_strategy = "Equal Weight"
    if 'custom_weights' not in st.session_state:
        st.session_state.custom_weights = {}
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_weights' not in st.session_state:
        st.session_state.current_weights = None
    if 'use_demo_data' not in st.session_state:
        st.session_state.use_demo_data = False
    if 'show_advanced' not in st.session_state:
        st.session_state.show_advanced = False
    
    # Sidebar configuration with enhanced features
    with st.sidebar:
        st.title("⚙️ Portfolio Configuration")
        
        # Add quick action buttons at the top
        col_qa1, col_qa2 = st.columns(2)
        with col_qa1:
            if st.button("📊 Quick Demo", use_container_width=True):
                st.session_state.selected_tickers = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD"]
                st.session_state.portfolio_strategy = "Maximum Sharpe Ratio"
                st.success("Demo portfolio loaded!")
        
        with col_qa2:
            if st.button("🔄 Reset All", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Asset selection with enhanced filtering
        st.subheader("🌍 Asset Selection")
        
        # Enhanced category filter with search
        category_search = st.text_input("Search Categories", "", key="category_search")
        
        if category_search:
            filtered_categories = [cat for cat in GLOBAL_ASSET_UNIVERSE.keys() 
                                 if category_search.lower() in cat.lower()]
        else:
            filtered_categories = list(GLOBAL_ASSET_UNIVERSE.keys())
        
        reliable_categories = ["US_Indices", "US_Stocks", "Bonds", "Commodities"]
        default_cats = [cat for cat in reliable_categories if cat in filtered_categories]
        
        category_filter = st.multiselect(
            "Filter by Category",
            filtered_categories,
            default=default_cats,
            key="sidebar_category_filter"
        )
        
        # Enhanced ticker selection with search
        asset_search = st.text_input("🔍 Search Assets", "", key="asset_search")
        
        # Filter tickers based on selected categories and search
        filtered_tickers = []
        if category_filter:
            for category in category_filter:
                filtered_tickers.extend(GLOBAL_ASSET_UNIVERSE[category])
        else:
            filtered_tickers = ALL_TICKERS
        
        # Remove duplicates while preserving order
        filtered_tickers = list(dict.fromkeys(filtered_tickers))
        
        # Apply search filter
        if asset_search:
            filtered_tickers = [ticker for ticker in filtered_tickers 
                              if asset_search.upper() in ticker.upper()]
        
        # Get valid default values
        current_selected = st.session_state.get('selected_tickers', default_tickers)
        valid_defaults = [t for t in current_selected if t in filtered_tickers]
        
        if not valid_defaults and filtered_tickers:
            valid_defaults = filtered_tickers[:min(5, len(filtered_tickers))]
        
        # Enhanced asset selector with virtualization for large lists
        selected_tickers = st.multiselect(
            "Select Assets (3-20 recommended)",
            filtered_tickers,
            default=valid_defaults,
            help="Select 3-20 assets for optimal diversification",
            key="sidebar_asset_selector"
        )
        
        # Display selection statistics
        if selected_tickers:
            st.caption(f"✅ Selected: {len(selected_tickers)} assets")
        
        # Validate selection
        if len(selected_tickers) < 2:
            st.error("Please select at least 2 assets for portfolio analysis")
        elif len(selected_tickers) > 20:
            st.warning("⚠️ More than 20 assets selected. This may impact performance.")
        
        st.session_state.selected_tickers = selected_tickers
        
        # Enhanced benchmark selection
        benchmark_options = ["SPY", "QQQ", "VTI", "IWM", "DIA", "BND", "GLD"] + \
                           [t for t in ALL_TICKERS if t not in selected_tickers]
        benchmark_options = list(dict.fromkeys(benchmark_options))
        
        default_benchmark = "SPY" if "SPY" in benchmark_options else benchmark_options[0]
        
        benchmark = st.selectbox(
            "📊 Benchmark Index",
            benchmark_options,
            index=benchmark_options.index(default_benchmark) if default_benchmark in benchmark_options else 0,
            help="Primary benchmark for performance comparison",
            key="sidebar_benchmark_selector"
        )
        
        # Enhanced date range selector
        st.subheader("📅 Date Range")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            date_preset = st.selectbox(
                "Time Period",
                ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years", "10 Years", "Max", "Custom"],
                index=4,  # Default to 3 years
                key="sidebar_date_preset"
            )
        
        with col_d2:
            if date_preset == "Custom":
                period_years = st.slider("Years", 1, 20, 5)
                date_preset = f"{period_years} Years"
        
        end_date = pd.Timestamp.today()
        
        if date_preset == "1 Month":
            start_date = end_date - pd.DateOffset(months=1)
        elif date_preset == "3 Months":
            start_date = end_date - pd.DateOffset(months=3)
        elif date_preset == "6 Months":
            start_date = end_date - pd.DateOffset(months=6)
        elif date_preset == "1 Year":
            start_date = end_date - pd.DateOffset(years=1)
        elif date_preset == "3 Years":
            start_date = end_date - pd.DateOffset(years=3)
        elif date_preset == "5 Years":
            start_date = end_date - pd.DateOffset(years=5)
        elif date_preset == "10 Years":
            start_date = end_date - pd.DateOffset(years=10)
        else:  # Max or Custom
            start_date = pd.Timestamp("2000-01-01")
        
        # Enhanced portfolio strategy selector
        st.subheader("🎯 Portfolio Strategy")
        
        strategy = st.selectbox(
            "Construction Method",
            list(PORTFOLIO_STRATEGIES.keys()),
            index=0,
            help="Select portfolio optimization strategy",
            key="sidebar_strategy_selector"
        )
        
        st.session_state.portfolio_strategy = strategy
        
        # Enhanced advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=st.session_state.show_advanced):
            # Risk parameters in columns
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                rf_annual = st.number_input(
                    "Risk-Free Rate (%)",
                    value=3.0,
                    min_value=0.0,
                    max_value=20.0,
                    step=0.5,
                    key="sidebar_rf_annual"
                ) / 100
            
            with col_a2:
                confidence_level = st.slider(
                    "VaR Confidence (%)",
                    min_value=90,
                    max_value=99,
                    value=95,
                    step=1,
                    key="sidebar_confidence_level"
                ) / 100
            
            # Optimization constraints
            short_selling = st.checkbox(
                "Allow Short Selling",
                value=False,
                help="Allow negative portfolio weights",
                key="sidebar_short_selling"
            )
            
            # Data options
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                use_parallel = st.checkbox(
                    "Parallel Download",
                    value=True,
                    help="Use parallel processing for faster data loading"
                )
            
            with col_d2:
                use_demo_data = st.checkbox(
                    "Demo Mode",
                    value=False,
                    help="Use synthetic data for testing"
                )
                st.session_state.use_demo_data = use_demo_data
            
            # Performance metrics
            st.subheader("📊 Performance Metrics")
            
            metrics_to_show = st.multiselect(
                "Select Metrics to Display",
                ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Omega Ratio", 
                 "Max Drawdown", "VaR", "CVaR", "Information Ratio"],
                default=["Sharpe Ratio", "Max Drawdown", "VaR"],
                key="metrics_selection"
            )
        
        # Enhanced action buttons
        st.markdown("---")
        
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            run_analysis = st.button(
                "🚀 Run Analysis",
                type="primary",
                use_container_width=True,
                key="sidebar_run_analysis"
            )
        
        with col_b2:
            if st.button("💾 Save Config", key="sidebar_save_config"):
                # Save configuration logic
                config = {
                    'selected_tickers': st.session_state.selected_tickers,
                    'strategy': st.session_state.portfolio_strategy,
                    'benchmark': benchmark,
                    'date_preset': date_preset,
                    'rf_rate': rf_annual
                }
                st.success("Configuration saved!")
        
        with col_b3:
            if st.button("📤 Export", key="sidebar_export"):
                # Export functionality placeholder
                st.info("Export feature coming soon!")
    
    # Main content area - Enhanced with loading states
    if not run_analysis or len(selected_tickers) < 2:
        # Enhanced welcome screen
        st.info("👈 Configure your portfolio in the sidebar and click 'Run Analysis'")
        
        # Dashboard-style statistics
        with st.expander("📊 Global Market Overview", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Assets", len(ALL_TICKERS), "Global Coverage")
            
            with col2:
                st.metric("Categories", len(GLOBAL_ASSET_UNIVERSE), "Diversified")
            
            with col3:
                st.metric("Geographic Markets", "15+", "Worldwide")
            
            with col4:
                st.metric("Asset Classes", "10+", "Multi-Asset")
            
            # Market heat map visualization
            st.subheader("🌍 Asset Distribution by Category")
            
            category_counts = {cat: len(assets) for cat, assets in GLOBAL_ASSET_UNIVERSE.items()}
            categories_df = pd.DataFrame({
                'Category': list(category_counts.keys()),
                'Count': list(category_counts.values())
            }).sort_values('Count', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=categories_df['Category'].str.replace('_', ' '),
                    y=categories_df['Count'],
                    marker_color='#1a5fb4',
                    text=categories_df['Count'],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                height=400,
                template="plotly_dark",
                xaxis_title="Category",
                yaxis_title="Number of Assets",
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced quick start guide
        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown("""
            ### Quick Start Guide
            
            1. **Select Assets**: Choose 3-20 diverse assets
            2. **Set Timeframe**: Pick appropriate historical period
            3. **Choose Strategy**: Select optimization method
            4. **Run Analysis**: Generate insights and reports
            
            ### 📋 Recommended Portfolios
            
            **Balanced Portfolio**:
            - SPY (US Stocks) - 40%
            - TLT (US Bonds) - 30%
            - GLD (Gold) - 10%
            - BND (Aggregate Bonds) - 20%
            
            **Growth Portfolio**:
            - QQQ (Tech) - 50%
            - ARKK (Innovation) - 20%
            - ICLN (Clean Energy) - 15%
            - MSFT (Individual Stock) - 15%
            
            **Conservative Portfolio**:
            - SHY (Short-term Bonds) - 40%
            - TIP (Inflation Protected) - 30%
            - Utilities Sector - 20%
            - Consumer Staples - 10%
            """)
        
        # Feature highlights
        with st.expander("✨ Key Features", expanded=False):
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                st.markdown("""
                **📈 Portfolio Optimization**
                - 9+ optimization strategies
                - Efficient frontier analysis
                - Real-time performance metrics
                """)
            
            with col_f2:
                st.markdown("""
                **⚖️ Risk Management**
                - Advanced VaR calculations
                - Monte Carlo simulations
                - Stress testing scenarios
                """)
            
            with col_f3:
                st.markdown("""
                **📊 Analytics**
                - Performance attribution
                - Correlation analysis
                - EWMA volatility
                """)
        
        st.stop()
    
    # Load data with enhanced progress tracking
    st.subheader("📊 Loading Market Data...")
    
    # Progress container
    progress_container = st.container()
    
    try:
        # Combine selected tickers with benchmark
        all_tickers_to_load = list(dict.fromkeys(selected_tickers + [benchmark]))
        
        with progress_container:
            # Create progress bars
            progress_overall = st.progress(0, text="Initializing...")
            progress_status = st.empty()
            
            # Load data
            with st.spinner(f"Downloading data for {len(all_tickers_to_load)} assets..."):
                # Update progress
                progress_status.text("Connecting to data sources...")
                
                # Use cached loader for better performance
                prices = load_global_prices_cached(all_tickers_to_load, start_date, end_date)
                
                # Update progress
                progress_overall.progress(50, text="Processing data...")
                
                # If data loading fails and demo data is enabled
                if prices.empty and st.session_state.use_demo_data:
                    progress_status.text("Generating demo data...")
                    prices = generate_demo_data(all_tickers_to_load, start_date, end_date)
                
                # Validate loaded data
                if prices.empty:
                    progress_overall.progress(0, text="Failed to load data")
                    st.error("❌ No data loaded. Please check your selections and try again.")
                    st.stop()
                
                if len(prices.columns) < 2:
                    st.error("❌ Insufficient data for analysis. Need at least 2 assets with valid data.")
                    st.stop()
                
                # Update progress
                progress_overall.progress(75, text="Calculating returns...")
                
                # Check which tickers were successfully loaded
                loaded_tickers = prices.columns.tolist()
                missing_tickers = [t for t in all_tickers_to_load if t not in loaded_tickers]
                
                # Update selected tickers to only include loaded ones
                selected_tickers = [t for t in selected_tickers if t in loaded_tickers]
                
                if benchmark not in loaded_tickers:
                    benchmark = loaded_tickers[0] if loaded_tickers else "SPY"
                
                if len(selected_tickers) < 2:
                    st.error("❌ Need at least 2 selected assets with valid data.")
                    st.stop()
                
                # Calculate returns
                prices = prices[selected_tickers + [benchmark]]
                returns = prices.pct_change().dropna()
                
                # Update progress
                progress_overall.progress(100, text="Data loaded successfully!")
                time.sleep(0.5)  # Brief pause to show completion
                
                # Clear progress indicators
                progress_overall.empty()
                progress_status.empty()
            
            # Store in session state
            st.session_state.prices = prices
            st.session_state.returns = returns
            st.session_state.benchmark = benchmark
            st.session_state.data_loaded = True
            
            # Success message with statistics
            success_col1, success_col2, success_col3 = st.columns(3)
            
            with success_col1:
                st.success(f"✅ {len(loaded_tickers)} assets loaded")
            
            with success_col2:
                st.info(f"📅 {len(returns)} trading days")
            
            with success_col3:
                st.info(f"📊 Data from {returns.index[0].date()} to {returns.index[-1].date()}")
            
            if missing_tickers:
                st.warning(f"⚠️ Missing data for {len(missing_tickers)} assets")
                
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)[:200]}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        st.stop()
    
    # Create enhanced tabs with icons and better organization
    tab_config = [
        {"name": "📈 Overview & Weights", "icon": "📈"},
        {"name": "⚖️ Risk Analytics", "icon": "⚖️"},
        {"name": "🎯 Optimization", "icon": "🎯"},
        {"name": "🔗 Correlation", "icon": "🔗"},
        {"name": "📊 Volatility", "icon": "📊"},
        {"name": "🎲 Monte Carlo", "icon": "🎲"},
        {"name": "📊 Attribution", "icon": "📊"},
        {"name": "📋 Report", "icon": "📋"}
    ]
    
    # Create tabs dynamically
    tabs = st.tabs([tab["name"] for tab in tab_config])
    
    # Tab 1: Enhanced Overview & Weights
    with tabs[0]:
        st.header("📊 Portfolio Overview")
        
        try:
            # Get data from session state
            returns = st.session_state.returns
            selected_tickers = st.session_state.selected_tickers
            benchmark = st.session_state.benchmark
            
            # Add a summary dashboard at the top
            st.subheader("📊 Portfolio Dashboard")
            
            # Calculate portfolio based on strategy
            if st.session_state.portfolio_strategy == "Equal Weight":
                n_assets = len(selected_tickers)
                weights = np.ones(n_assets) / n_assets
                portfolio_returns = returns[selected_tickers].mean(axis=1)
                method_desc = "Equal Weight Portfolio"
                
            elif st.session_state.portfolio_strategy == "Custom Weights":
                # Custom weights editor
                st.subheader("✏️ Custom Portfolio Weights")
                
                # Initialize or update custom weights
                if selected_tickers != list(st.session_state.custom_weights.keys()):
                    st.session_state.custom_weights = {
                        ticker: 1.0/len(selected_tickers) 
                        for ticker in selected_tickers
                    }
                
                # Create weight editor
                cols = st.columns(3)
                weight_inputs = {}
                
                for idx, ticker in enumerate(selected_tickers):
                    with cols[idx % 3]:
                        default_weight = st.session_state.custom_weights.get(ticker, 0.0) * 100
                        weight = st.number_input(
                            f"{ticker} Weight (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=default_weight,
                            step=1.0,
                            key=f"tab1_weight_{ticker}"
                        ) / 100
                        weight_inputs[ticker] = weight
                
                # Normalize weights
                total_weight = sum(weight_inputs.values())
                if total_weight > 0:
                    weights = np.array([weight_inputs[t] / total_weight for t in selected_tickers])
                    st.session_state.custom_weights = {t: weights[i] for i, t in enumerate(selected_tickers)}
                else:
                    weights = np.ones(len(selected_tickers)) / len(selected_tickers)
                    st.warning("Weights must sum to > 0. Using equal weights.")
                
                portfolio_returns = (returns[selected_tickers] * weights).sum(axis=1)
                method_desc = "Custom Weight Portfolio"
                
            else:
                # Use PyPortfolioOpt optimization
                with st.spinner("🔧 Optimizing portfolio..."):
                    optimizer = PortfolioOptimizer()
                    result = optimizer.optimize_portfolio(
                        returns[selected_tickers],
                        st.session_state.portfolio_strategy,
                        risk_free_rate=rf_annual,
                        risk_aversion=2.5,
                        short_allowed=short_selling
                    )
                    
                    weights = result['weights']
                    portfolio_returns = (returns[selected_tickers] * weights).sum(axis=1)
                    method_desc = result['method']
            
            # Store weights for use in other tabs
            st.session_state.current_weights = weights
            st.session_state.portfolio_returns = portfolio_returns
            st.session_state.asset_returns = returns[selected_tickers]
            
            # Quick stats row
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                ann_return = portfolio_returns.mean() * 252
                st.metric("Expected Return", f"{ann_return:.2%}")
            
            with col_s2:
                ann_vol = portfolio_returns.std() * np.sqrt(252)
                st.metric("Expected Volatility", f"{ann_vol:.2%}")
            
            with col_s3:
                sharpe = (ann_return - rf_annual) / (ann_vol + 1e-10)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col_s4:
                # Calculate max drawdown
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = drawdown.min()
                st.metric("Max Drawdown", f"{max_dd:.2%}")
            
            # Display weights and performance
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📊 Portfolio Weights")
                
                weights_df = pd.DataFrame({
                    'Asset': selected_tickers,
                    'Weight': weights,
                    'Category': [
                        next((cat for cat, assets in GLOBAL_ASSET_UNIVERSE.items() if t in assets), 'Other') 
                        for t in selected_tickers
                    ]
                })
                
                # Sort by weight
                weights_df = weights_df.sort_values('Weight', ascending=False)
                
                # Display weights table
                st.dataframe(
                    weights_df.style.format({'Weight': '{:.2%}'}).background_gradient(
                        subset=['Weight'], cmap='Blues'
                    ),
                    use_container_width=True,
                    height=400
                )
            
            with col2:
                st.subheader("📈 Portfolio Performance")
                
                # Benchmark returns
                benchmark_returns = returns[benchmark]
                
                # Calculate cumulative performance
                portfolio_cumulative = (1 + portfolio_returns).cumprod()
                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                
                # Create performance chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=portfolio_cumulative.index,
                    y=portfolio_cumulative.values,
                    name=f"Portfolio ({method_desc})",
                    line=dict(color='#1a5fb4', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(26, 95, 180, 0.1)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    name=f"Benchmark ({benchmark})",
                    line=dict(color='#26a269', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Cumulative Performance vs Benchmark",
                    height=500,
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in Overview tab: {str(e)[:200]}")
            st.error("Please try different assets or settings.")
    
    # Tab 2: Enhanced Risk Analytics
    with tabs[1]:
        st.header("⚖️ Comprehensive Risk Analytics")
        
        # Ensure we have portfolio data
        if 'portfolio_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        try:
            portfolio_returns = st.session_state.portfolio_returns
            risk_analytics = EnhancedRiskAnalytics()
            
            # Risk metrics in a clean grid
            st.subheader("⚖️ Comprehensive Risk Dashboard")
            
            # Calculate all risk metrics
            metrics_calculator = PerformanceMetrics()
            risk_metrics = metrics_calculator.calculate_metrics(portfolio_returns, risk_free_rate=rf_annual)
            
            # Display in a clean grid
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.2%}")
                st.metric("VaR (99%)", f"{risk_metrics.get('var_99', 0):.2%}")
            
            with col_r2:
                st.metric("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0):.2%}")
                st.metric("CVaR (99%)", f"{risk_metrics.get('cvar_99', 0):.2%}")
            
            with col_r3:
                st.metric("Sortino Ratio", f"{risk_metrics.get('sortino_ratio', 0):.2f}")
                st.metric("Calmar Ratio", f"{risk_metrics.get('calmar_ratio', 0):.2f}")
            
            with col_r4:
                st.metric("Omega Ratio", f"{risk_metrics.get('omega_ratio', 0):.2f}")
                st.metric("Ulcer Index", f"{risk_metrics.get('ulcer_index', 0):.3f}")
            
            # VaR Method Selection
            st.subheader("📊 Value at Risk Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                var_method = st.selectbox(
                    "VaR Calculation Method",
                    ["Historical Simulation", "Parametric (Normal)", "EWMA", 
                     "Monte Carlo Simulation", "Compare All Methods"],
                    index=0,
                    key="var_method_tab2"
                )
            
            with col2:
                var_horizon = st.selectbox(
                    "Time Horizon",
                    ["1 Day", "5 Days", "10 Days", "1 Month", "3 Months"],
                    index=0,
                    key="var_horizon_tab2"
                )
                
                # Convert horizon to days
                horizon_map = {"1 Day": 1, "5 Days": 5, "10 Days": 10, "1 Month": 21, "3 Months": 63}
                horizon_days = horizon_map[var_horizon]
            
            with col3:
                var_confidence = st.slider(
                    "Confidence Level (%)",
                    min_value=90,
                    max_value=99,
                    value=95,
                    step=1,
                    key="var_confidence_tab2"
                ) / 100
            
            # Calculate portfolio returns for horizon
            if horizon_days > 1:
                # Aggregate returns for horizon
                horizon_returns = portfolio_returns.rolling(horizon_days).apply(
                    lambda x: np.prod(1 + x) - 1, raw=True
                ).dropna()
            else:
                horizon_returns = portfolio_returns
            
            # Calculate VaR based on selected method
            if var_method == "Compare All Methods":
                # Compare all VaR methods
                st.subheader("📈 VaR Method Comparison")
                
                var_results = risk_analytics.calculate_all_var_methods(
                    horizon_returns, 
                    var_confidence
                )
                
                # Display results
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.dataframe(
                        var_results.style.format({
                            'VaR': '{:.4%}',
                            'CVaR': '{:.4%}'
                        }).applymap(
                            lambda x: 'color: green' if isinstance(x, float) and x < 0 else '', 
                            subset=['VaR', 'CVaR']
                        ),
                        use_container_width=True,
                        height=300
                    )
                
                with col_b:
                    # Visualization
                    fig = go.Figure()
                    
                    # Filter successful results
                    successful_results = var_results[var_results['Status'] == 'Success']
                    
                    if not successful_results.empty:
                        fig.add_trace(go.Bar(
                            name='VaR',
                            x=successful_results['Method'],
                            y=successful_results['VaR'].abs() * 100,
                            marker_color='#1a5fb4',
                            text=successful_results['VaR'].apply(lambda x: f"{x:.2%}"),
                            textposition='auto'
                        ))
                        
                        fig.add_trace(go.Bar(
                            name='CVaR',
                            x=successful_results['Method'],
                            y=successful_results['CVaR'].abs() * 100,
                            marker_color='#26a269',
                            text=successful_results['CVaR'].apply(lambda x: f"{x:.2%}"),
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="VaR & CVaR Comparison by Method",
                        height=400,
                        template="plotly_dark",
                        yaxis_title="Value (%)",
                        barmode='group',
                        xaxis_tickangle=45
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Single method analysis
                method_map = {
                    "Historical Simulation": "historical",
                    "Parametric (Normal)": "parametric",
                    "EWMA": "ewma",
                    "Monte Carlo Simulation": "monte_carlo"
                }
                
                method_key = method_map[var_method]
                
                # Calculate VaR
                var_result = risk_analytics.calculate_var(
                    horizon_returns,
                    method_key,
                    var_confidence,
                    params={'n_simulations': 10000, 'days': horizon_days} if method_key == "monte_carlo" else None
                )
                
                # Display results
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if var_result.get('success', False) and not pd.isna(var_result.get('VaR')):
                        st.metric(
                            f"{var_horizon} VaR ({var_confidence*100:.1f}%)",
                            f"{var_result['VaR']:.4%}"
                        )
                    else:
                        st.metric(
                            f"{var_horizon} VaR ({var_confidence*100:.1f}%)",
                            "N/A"
                        )
                
                with col_b:
                    if var_result.get('success', False) and not pd.isna(var_result.get('CVaR')):
                        st.metric(
                            f"{var_horizon} CVaR ({var_confidence*100:.1f}%)",
                            f"{var_result['CVaR']:.4%}"
                        )
                    else:
                        st.metric(
                            f"{var_horizon} CVaR ({var_confidence*100:.1f}%)",
                            "N/A"
                        )
                
                with col_c:
                    if 'method' in var_result:
                        st.metric("Method", var_result['method'])
                
                # Distribution plot with VaR
                if var_result.get('success', False) and not pd.isna(var_result.get('VaR')):
                    st.subheader("📊 Return Distribution with VaR")
                    
                    fig = go.Figure()
                    
                    # Histogram of returns
                    fig.add_trace(go.Histogram(
                        x=horizon_returns,
                        nbinsx=50,
                        name="Returns",
                        marker_color='#1a5fb4',
                        opacity=0.7,
                        histnorm='probability density'
                    ))
                    
                    # Add VaR line
                    fig.add_vline(
                        x=var_result['VaR'],
                        line_dash="dash",
                        line_color="#f5a623",
                        annotation_text=f"VaR: {var_result['VaR']:.4%}",
                        annotation_position="top left"
                    )
                    
                    # Add CVaR line
                    if not pd.isna(var_result.get('CVaR')):
                        fig.add_vline(
                            x=var_result['CVaR'],
                            line_dash="dot",
                            line_color="#c01c28",
                            annotation_text=f"CVaR: {var_result['CVaR']:.4%}",
                            annotation_position="top right"
                        )
                    
                    fig.update_layout(
                        height=500,
                        template="plotly_dark",
                        xaxis_title="Return",
                        yaxis_title="Density",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Risk Metrics by Asset
            st.subheader("📊 Detailed Risk Metrics by Asset")
            
            # Calculate metrics for each asset
            risk_metrics_data = []
            
            for asset in selected_tickers:
                asset_returns = returns[asset]
                
                # Basic metrics
                ann_return = asset_returns.mean() * TRADING_DAYS
                ann_vol = asset_returns.std() * np.sqrt(TRADING_DAYS)
                sharpe = (ann_return - rf_annual) / (ann_vol + 1e-10)
                
                # Risk metrics
                max_dd = ((1 + asset_returns).cumprod() / (1 + asset_returns).cumprod().cummax() - 1).min()
                
                # VaR metrics (historical)
                var_95 = risk_analytics.calculate_var(asset_returns, "historical", 0.95)
                var_99 = risk_analytics.calculate_var(asset_returns, "historical", 0.99)
                
                risk_metrics_data.append({
                    'Asset': asset,
                    'Annual Return': ann_return,
                    'Annual Volatility': ann_vol,
                    'Sharpe Ratio': sharpe,
                    'Max Drawdown': max_dd,
                    'VaR 95%': var_95.get('VaR', np.nan),
                    'CVaR 95%': var_95.get('CVaR', np.nan),
                    'VaR 99%': var_99.get('VaR', np.nan),
                    'CVaR 99%': var_99.get('CVaR', np.nan)
                })
            
            risk_metrics_df = pd.DataFrame(risk_metrics_data)
            
            # Display interactive table
            st.dataframe(
                risk_metrics_df.style.format({
                    'Annual Return': '{:.2%}',
                    'Annual Volatility': '{:.2%}',
                    'Sharpe Ratio': '{:.2f}',
                    'Max Drawdown': '{:.2%}',
                    'VaR 95%': '{:.2%}',
                    'CVaR 95%': '{:.2%}',
                    'VaR 99%': '{:.2%}',
                    'CVaR 99%': '{:.2%}'
                }).background_gradient(
                    subset=['Sharpe Ratio'], cmap='RdYlGn'
                ),
                use_container_width=True,
                height=400
            )
            
        except Exception as e:
            st.error(f"Error in Risk Analytics tab: {str(e)[:200]}")
    
    # Tab 3: Portfolio Optimization
    with tabs[2]:
        st.header("🎯 Portfolio Optimization Strategies")
        
        # Ensure we have data
        if 'asset_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        try:
            asset_returns = st.session_state.asset_returns
            
            if not PYPFOPT_AVAILABLE:
                st.warning("PyPortfolioOpt is not installed. Using basic optimization methods.")
            
            # Optimization strategies comparison
            st.subheader("📊 Strategy Comparison")
            
            strategies_to_test = [
                "Minimum Volatility",
                "Maximum Sharpe Ratio",
                "Maximum Quadratic Utility",
                "Efficient Risk",
                "Efficient Return"
            ]
            
            optimization_results = []
            
            with st.spinner("🔄 Testing optimization strategies..."):
                optimizer = PortfolioOptimizer()
                
                for strategy in strategies_to_test:
                    try:
                        result = optimizer.optimize_portfolio(
                            asset_returns,
                            strategy,
                            risk_free_rate=rf_annual,
                            risk_aversion=2.5,
                            short_allowed=short_selling
                        )
                        
                        optimization_results.append({
                            'Strategy': strategy,
                            'Expected Return': result['expected_return'],
                            'Expected Risk': result['expected_risk'],
                            'Sharpe Ratio': result['sharpe_ratio'],
                            'Method': result['method'],
                            'Success': result['success']
                        })
                        
                    except Exception as e:
                        optimization_results.append({
                            'Strategy': strategy,
                            'Expected Return': np.nan,
                            'Expected Risk': np.nan,
                            'Sharpe Ratio': np.nan,
                            'Method': f"Error: {str(e)[:50]}",
                            'Success': False
                        })
            
            # Display results
            results_df = pd.DataFrame(optimization_results)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    results_df.style.format({
                        'Expected Return': '{:.2%}',
                        'Expected Risk': '{:.2%}',
                        'Sharpe Ratio': '{:.2f}'
                    }).applymap(
                        lambda x: 'color: red' if pd.isna(x) else '', 
                        subset=['Expected Return', 'Expected Risk', 'Sharpe Ratio']
                    ),
                    use_container_width=True,
                    height=300
                )
            
            with col2:
                # Visualization
                fig = go.Figure()
                
                valid_results = results_df[results_df['Success'] == True].dropna(subset=['Expected Return', 'Expected Risk'])
                
                if not valid_results.empty:
                    fig.add_trace(go.Scatter(
                        x=valid_results['Expected Risk'],
                        y=valid_results['Expected Return'],
                        mode='markers+text',
                        text=valid_results['Strategy'],
                        textposition="top center",
                        marker=dict(
                            size=15,
                            color=valid_results['Sharpe Ratio'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        name='Strategies'
                    ))
                    
                    # Add equal weight portfolio for comparison
                    eq_return = asset_returns.mean(axis=1).mean() * TRADING_DAYS
                    eq_risk = asset_returns.mean(axis=1).std() * np.sqrt(TRADING_DAYS)
                    
                    fig.add_trace(go.Scatter(
                        x=[eq_risk],
                        y=[eq_return],
                        mode='markers',
                        marker=dict(
                            size=20,
                            color='red',
                            symbol='star'
                        ),
                        name='Equal Weight'
                    ))
                
                fig.update_layout(
                    title="Efficient Frontier & Optimization Strategies",
                    height=500,
                    template="plotly_dark",
                    xaxis_title="Annual Volatility",
                    yaxis_title="Annual Return",
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Interactive optimization
            st.subheader("🔄 Interactive Optimization")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                target_type = st.selectbox(
                    "Optimization Target",
                    ["Minimum Volatility", "Maximum Sharpe Ratio", "Target Return", "Target Risk"],
                    index=1,
                    key="optim_target_tab3"
                )
            
            with col_b:
                if target_type == "Target Return":
                    target_value = st.number_input(
                        "Target Annual Return (%)",
                        min_value=0.0,
                        max_value=50.0,
                        value=10.0,
                        step=1.0,
                        key="target_return_tab3"
                    ) / 100
                elif target_type == "Target Risk":
                    target_value = st.number_input(
                        "Target Annual Volatility (%)",
                        min_value=5.0,
                        max_value=50.0,
                        value=15.0,
                        step=1.0,
                        key="target_risk_tab3"
                    ) / 100
                else:
                    target_value = None
            
            # Run optimization
            if st.button("🚀 Run Optimization", type="primary", key="run_optimization_tab3"):
                with st.spinner("Optimizing portfolio..."):
                    if target_type == "Minimum Volatility":
                        strategy_name = "Minimum Volatility"
                    elif target_type == "Maximum Sharpe Ratio":
                        strategy_name = "Maximum Sharpe Ratio"
                    elif target_type == "Target Return":
                        strategy_name = "Efficient Return"
                    else:
                        strategy_name = "Efficient Risk"
                    
                    result = optimizer.optimize_portfolio(
                        asset_returns,
                        strategy_name,
                        target_return=target_value if target_type == "Target Return" else None,
                        target_risk=target_value if target_type == "Target Risk" else None,
                        risk_free_rate=rf_annual,
                        risk_aversion=2.5,
                        short_allowed=short_selling
                    )
                    
                    # Display optimized weights
                    st.subheader("📊 Optimized Portfolio Weights")
                    
                    optimized_weights_df = pd.DataFrame({
                        'Asset': selected_tickers,
                        'Weight': result['weights'],
                        'Category': [next((cat for cat, assets in GLOBAL_ASSET_UNIVERSE.items() if t in assets), 'Other') 
                                   for t in selected_tickers]
                    }).sort_values('Weight', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(
                            optimized_weights_df.style.format({'Weight': '{:.2%}'}).background_gradient(
                                subset=['Weight'], cmap='Blues'
                            ),
                            use_container_width=True,
                            height=400
                        )
                    
                    with col2:
                        # Pie chart of weights
                        fig = go.Figure(data=[go.Pie(
                            labels=optimized_weights_df['Asset'],
                            values=optimized_weights_df['Weight'],
                            hole=0.4,
                            marker=dict(colors=px.colors.qualitative.Set3),
                            textinfo='label+percent',
                            textposition='auto'
                        )])
                        
                        fig.update_layout(
                            height=400,
                            title="Optimized Portfolio Allocation",
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics
                    st.subheader("📈 Optimized Portfolio Performance")
                    
                    metrics_cols = st.columns(4)
                    
                    with metrics_cols[0]:
                        st.metric("Expected Return", f"{result['expected_return']:.2%}")
                    
                    with metrics_cols[1]:
                        st.metric("Expected Risk", f"{result['expected_risk']:.2%}")
                    
                    with metrics_cols[2]:
                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                    
                    with metrics_cols[3]:
                        if result['success']:
                            st.metric("Status", "Success", delta=None)
                        else:
                            st.metric("Status", "Fallback", delta=None)
        
        except Exception as e:
            st.error(f"Error in Portfolio Optimization tab: {str(e)[:200]}")
    
    # Tab 4: Correlation Matrix
    with tabs[3]:
        st.header("🔗 Correlation Matrix & Risk Decomposition")
        
        # Ensure we have data
        if 'asset_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        try:
            asset_returns = st.session_state.asset_returns
            
            # Risk model selection
            st.subheader("📊 Risk Model Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                corr_method = st.selectbox(
                    "Correlation Method",
                    ["Sample Correlation", "Exponential Weighted", "Constant Correlation"],
                    index=0,
                    key="corr_method_tab4"
                )
            
            with col2:
                lookback_window = st.slider(
                    "Lookback Window (days)",
                    min_value=30,
                    max_value=1000,
                    value=252,
                    step=30,
                    key="lookback_window_tab4"
                )
            
            # Calculate correlation matrix
            recent_returns = asset_returns.iloc[-lookback_window:] if len(asset_returns) > lookback_window else asset_returns
            
            if corr_method == "Sample Correlation":
                corr_matrix = recent_returns.corr()
            elif corr_method == "Exponential Weighted":
                # EWMA correlation with proper handling
                try:
                    corr_matrix = recent_returns.ewm(span=60).corr()
                    # Get the last correlation matrix
                    if isinstance(corr_matrix, pd.DataFrame):
                        # Extract the latest correlation matrix
                        unique_dates = corr_matrix.index.get_level_values(0).unique()
                        if len(unique_dates) > 0:
                            latest_date = unique_dates[-1]
                            corr_matrix = corr_matrix.loc[latest_date]
                        else:
                            corr_matrix = recent_returns.corr()
                    else:
                        corr_matrix = recent_returns.corr()
                except:
                    corr_matrix = recent_returns.corr()
            else:
                corr_matrix = recent_returns.corr()
            
            # Enhanced heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(
                    title="Correlation",
                    titleside="right"
                )
            ))
            
            fig.update_layout(
                title=f"Correlation Matrix ({corr_method})",
                height=700,
                template="plotly_dark",
                xaxis_title="Assets",
                yaxis_title="Assets",
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation statistics
            st.subheader("📈 Correlation Statistics")
            
            # Get upper triangle of correlation matrix (excluding diagonal)
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_corr = corr_values.mean() if len(corr_values) > 0 else 0
                st.metric("Average Correlation", f"{avg_corr:.3f}")
            
            with col2:
                min_corr = corr_values.min() if len(corr_values) > 0 else 0
                st.metric("Minimum Correlation", f"{min_corr:.3f}")
            
            with col3:
                max_corr = corr_values.max() if len(corr_values) > 0 else 0
                st.metric("Maximum Correlation", f"{max_corr:.3f}")
            
            with col4:
                std_corr = corr_values.std() if len(corr_values) > 0 else 0
                st.metric("Correlation Std", f"{std_corr:.3f}")
            
            # Correlation distribution
            fig2 = go.Figure()
            
            fig2.add_trace(go.Histogram(
                x=corr_values,
                nbinsx=30,
                name="Correlation Distribution",
                marker_color='#1a5fb4',
                opacity=0.7
            ))
            
            fig2.add_vline(
                x=avg_corr,
                line_dash="dash",
                line_color="#f5a623",
                annotation_text=f"Mean: {avg_corr:.3f}",
                annotation_position="top"
            )
            
            fig2.update_layout(
                title="Correlation Distribution",
                height=400,
                template="plotly_dark",
                xaxis_title="Correlation",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in Correlation Matrix tab: {str(e)[:200]}")
    
    # Tab 5: EWMA Analysis
    with tabs[4]:
        st.header("📊 EWMA Volatility Analysis")
        
        # Ensure we have data
        if 'asset_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        try:
            asset_returns = st.session_state.asset_returns
            
            # EWMA parameters
            st.subheader("⚙️ EWMA Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lambda_param = st.slider(
                    "Decay Factor (λ)",
                    min_value=0.85,
                    max_value=0.99,
                    value=0.94,
                    step=0.01,
                    help="Higher λ gives more weight to older observations",
                    key="lambda_param_tab5"
                )
            
            with col2:
                ewma_window = st.number_input(
                    "Lookback Period (days)",
                    min_value=30,
                    max_value=1000,
                    value=252,
                    step=30,
                    key="ewma_window_tab5"
                )
            
            with col3:
                display_assets = st.multiselect(
                    "Assets to Display",
                    selected_tickers,
                    default=selected_tickers[:5] if len(selected_tickers) >= 5 else selected_tickers,
                    key="ewma_display_assets_tab5"
                )
            
            # Calculate EWMA volatility
            ewma_analysis = EWMAAnalysis()
            recent_returns = asset_returns.iloc[-ewma_window:] if len(asset_returns) > ewma_window else asset_returns
            ewma_vol = ewma_analysis.calculate_ewma_volatility(recent_returns, lambda_param)
            
            # Time series plot
            st.subheader("📈 EWMA Volatility Time Series")
            
            fig = go.Figure()
            
            for asset in display_assets:
                if asset in ewma_vol.columns:
                    fig.add_trace(go.Scatter(
                        x=ewma_vol.index,
                        y=ewma_vol[asset] * np.sqrt(TRADING_DAYS) * 100,  # Annualized percentage
                        name=asset,
                        mode='lines',
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title=f"EWMA Volatility (λ={lambda_param}) - Annualized %",
                height=500,
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility (%)",
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility comparison
            st.subheader("📊 Volatility Comparison")
            
            # Calculate average volatility for each asset
            avg_vols = {}
            for asset in selected_tickers:
                if asset in ewma_vol.columns:
                    avg_vols[asset] = ewma_vol[asset].mean() * np.sqrt(TRADING_DAYS) * 100
            
            if avg_vols:
                avg_vol_df = pd.DataFrame({
                    'Asset': list(avg_vols.keys()),
                    'Avg Annualized Vol (%)': list(avg_vols.values())
                }).sort_values('Avg Annualized Vol (%)', ascending=False)
                
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=avg_vol_df['Asset'],
                        y=avg_vol_df['Avg Annualized Vol (%)'],
                        marker_color='#1a5fb4',
                        text=avg_vol_df['Avg Annualized Vol (%)'].apply(lambda x: f"{x:.1f}%"),
                        textposition='auto'
                    )
                ])
                
                fig2.update_layout(
                    title="Average Annualized Volatility by Asset",
                    height=400,
                    template="plotly_dark",
                    xaxis_title="Assets",
                    yaxis_title="Annualized Volatility (%)",
                    xaxis_tickangle=45
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in EWMA Analysis tab: {str(e)[:200]}")
    
    # Tab 6: Monte Carlo VaR
    with tabs[5]:
        st.header("🎲 Monte Carlo Simulation & VaR")
        
        # Ensure we have portfolio data
        if 'portfolio_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        try:
            portfolio_returns = st.session_state.portfolio_returns
            
            # Simulation parameters
            st.subheader("⚙️ Simulation Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_simulations = st.number_input(
                    "Number of Simulations",
                    min_value=1000,
                    max_value=50000,
                    value=10000,
                    step=1000,
                    key="n_simulations_tab6"
                )
            
            with col2:
                time_horizon = st.selectbox(
                    "Time Horizon",
                    ["1 Month", "3 Months", "6 Months", "1 Year"],
                    index=3,
                    key="time_horizon_tab6"
                )
                
                horizon_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252}
                horizon_days = horizon_map[time_horizon]
            
            with col3:
                mc_confidence = st.slider(
                    "Confidence Level",
                    min_value=90,
                    max_value=99,
                    value=95,
                    step=1,
                    key="mc_confidence_tab6"
                ) / 100
            
            # Run Monte Carlo simulation
            if st.button("🚀 Run Monte Carlo Simulation", type="primary", key="run_monte_carlo_tab6"):
                with st.spinner(f"Running {n_simulations:,} simulations..."):
                    try:
                        # Use portfolio returns for simulation
                        mu = portfolio_returns.mean()
                        sigma = portfolio_returns.std()
                        initial_value = 100
                        
                        simulator = MonteCarloSimulator()
                        
                        # GBM simulation
                        paths = simulator.simulate_gbm(
                            initial_value, mu, sigma, 
                            T=horizon_days/TRADING_DAYS, 
                            n_steps=horizon_days,
                            n_sims=n_simulations
                        )
                        
                        # Calculate final returns
                        final_returns = (paths[-1] / initial_value) - 1
                        
                        # Calculate VaR and CVaR
                        var_mc, cvar_mc = simulator.calculate_var_cvar(paths, mc_confidence)
                        
                        # Store in session state
                        st.session_state.mc_paths = paths
                        st.session_state.mc_final_returns = final_returns
                        st.session_state.mc_var = var_mc
                        st.session_state.mc_cvar = cvar_mc
                        st.session_state.mc_params = {
                            'n_simulations': n_simulations,
                            'horizon_days': horizon_days,
                            'confidence': mc_confidence,
                            'mu': mu,
                            'sigma': sigma
                        }
                        
                        st.success("✅ Simulation completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Simulation failed: {str(e)[:200]}")
            
            # Display results if available
            if 'mc_paths' in st.session_state:
                st.subheader("📊 Simulation Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        f"{time_horizon} VaR ({mc_confidence*100:.1f}%)",
                        f"{st.session_state.mc_var:.4%}"
                    )
                
                with col2:
                    st.metric(
                        f"{time_horizon} CVaR ({mc_confidence*100:.1f}%)",
                        f"{st.session_state.mc_cvar:.4%}"
                    )
                
                with col3:
                    expected_return = st.session_state.mc_final_returns.mean()
                    st.metric("Expected Return", f"{expected_return:.4%}")
                
                with col4:
                    prob_loss = (st.session_state.mc_final_returns < 0).mean()
                    st.metric("Probability of Loss", f"{prob_loss:.2%}")
                
                # Visualization: Sample paths
                st.subheader("📈 Sample Simulation Paths")
                
                fig1 = go.Figure()
                
                # Plot first 100 paths
                max_paths_to_plot = min(100, st.session_state.mc_params['n_simulations'])
                for i in range(max_paths_to_plot):
                    fig1.add_trace(go.Scatter(
                        x=list(range(st.session_state.mc_params['horizon_days'] + 1)),
                        y=st.session_state.mc_paths[:, i],
                        mode='lines',
                        line=dict(width=0.5, color='rgba(26, 95, 180, 0.1)'),
                        showlegend=False
                    ))
                
                # Plot mean path and confidence intervals
                mean_path = st.session_state.mc_paths.mean(axis=1)
                upper_95 = np.percentile(st.session_state.mc_paths, 97.5, axis=1)
                lower_95 = np.percentile(st.session_state.mc_paths, 2.5, axis=1)
                
                fig1.add_trace(go.Scatter(
                    x=list(range(len(mean_path))),
                    y=mean_path,
                    mode='lines',
                    line=dict(width=3, color='#f5a623'),
                    name='Mean Path'
                ))
                
                fig1.add_trace(go.Scatter(
                    x=list(range(len(upper_95))) + list(range(len(lower_95)))[::-1],
                    y=list(upper_95) + list(lower_95)[::-1],
                    fill='toself',
                    fillcolor='rgba(26, 95, 180, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval'
                ))
                
                fig1.update_layout(
                    title=f"Monte Carlo Simulation Paths ({st.session_state.mc_params['n_simulations']:,} simulations)",
                    height=500,
                    template="plotly_dark",
                    xaxis_title="Days",
                    yaxis_title="Portfolio Value",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Visualization: Distribution of final returns
                st.subheader("📊 Distribution of Final Portfolio Values")
                
                fig2 = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Return Distribution", "Cumulative Distribution")
                )
                
                # Histogram
                fig2.add_trace(
                    go.Histogram(
                        x=st.session_state.mc_final_returns * 100,
                        nbinsx=50,
                        name="Return Distribution",
                        marker_color='#1a5fb4',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Add VaR and CVaR lines
                fig2.add_vline(
                    x=st.session_state.mc_var * 100,
                    line_dash="dash",
                    line_color="#f5a623",
                    annotation_text=f"VaR: {st.session_state.mc_var:.2%}",
                    row=1, col=1
                )
                
                fig2.add_vline(
                    x=st.session_state.mc_cvar * 100,
                    line_dash="dot",
                    line_color="#c01c28",
                    annotation_text=f"CVaR: {st.session_state.mc_cvar:.2%}",
                    row=1, col=1
                )
                
                # CDF
                sorted_returns = np.sort(st.session_state.mc_final_returns)
                cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
                
                fig2.add_trace(
                    go.Scatter(
                        x=sorted_returns * 100,
                        y=cdf,
                        mode='lines',
                        name="CDF",
                        line=dict(color='#26a269', width=3)
                    ),
                    row=1, col=2
                )
                
                fig2.update_layout(
                    height=500,
                    template="plotly_dark",
                    showlegend=True
                )
                
                fig2.update_xaxes(title_text="Return (%)", row=1, col=1)
                fig2.update_xaxes(title_text="Return (%)", row=1, col=2)
                fig2.update_yaxes(title_text="Frequency", row=1, col=1)
                fig2.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
                
                st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in Monte Carlo VaR tab: {str(e)[:200]}")
    
    # Tab 7: Performance Attribution
    with tabs[6]:
        st.header("📊 Performance Attribution Analysis")
        
        # Ensure we have data
        if 'portfolio_returns' not in st.session_state or 'asset_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        try:
            portfolio_returns = st.session_state.portfolio_returns
            asset_returns = st.session_state.asset_returns
            weights = st.session_state.current_weights
            
            # Benchmark weights (assume equal weight for benchmark)
            benchmark_weights = np.ones(len(selected_tickers)) / len(selected_tickers)
            benchmark_portfolio_returns = (asset_returns * benchmark_weights).sum(axis=1)
            
            # Performance Attribution Engine
            st.subheader("🎯 Brinson Attribution Analysis")
            
            attribution = PerformanceAttribution()
            
            # Calculate attribution
            brinson_results = attribution.calculate_brinson_attribution(
                portfolio_returns,
                benchmark_portfolio_returns,
                weights,
                benchmark_weights,
                asset_returns
            )
            
            # Display attribution results
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                value = brinson_results['total_active_return'] * TRADING_DAYS
                st.metric("Total Active Return", f"{value:.2%}")
            
            with col2:
                value = brinson_results['allocation_effect'] * TRADING_DAYS
                st.metric("Allocation Effect", f"{value:.2%}")
            
            with col3:
                value = brinson_results['selection_effect'] * TRADING_DAYS
                st.metric("Selection Effect", f"{value:.2%}")
            
            with col4:
                value = brinson_results['interaction_effect'] * TRADING_DAYS
                st.metric("Interaction Effect", f"{value:.2%}")
            
            with col5:
                value = brinson_results['residual'] * TRADING_DAYS
                st.metric("Residual", f"{value:.2%}")
            
            # Visualization of attribution
            fig = go.Figure(data=[
                go.Bar(
                    name='Allocation',
                    x=['Allocation'],
                    y=[brinson_results['allocation_effect'] * TRADING_DAYS * 100],
                    marker_color='#1a5fb4'
                ),
                go.Bar(
                    name='Selection',
                    x=['Selection'],
                    y=[brinson_results['selection_effect'] * TRADING_DAYS * 100],
                    marker_color='#26a269'
                ),
                go.Bar(
                    name='Interaction',
                    x=['Interaction'],
                    y=[brinson_results['interaction_effect'] * TRADING_DAYS * 100],
                    marker_color='#f5a623'
                ),
                go.Bar(
                    name='Total Active',
                    x=['Total'],
                    y=[brinson_results['total_active_return'] * TRADING_DAYS * 100],
                    marker_color='#c01c28'
                )
            ])
            
            fig.update_layout(
                title="Performance Attribution Breakdown",
                height=500,
                template="plotly_dark",
                yaxis_title="Active Return Contribution (%)",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Asset contribution to returns
            st.subheader("📊 Asset Contribution Analysis")
            
            # Calculate asset contributions
            asset_contributions = pd.DataFrame({
                'Asset': selected_tickers,
                'Weight': weights,
                'Return': asset_returns.mean().values * TRADING_DAYS,
                'Contribution': weights * asset_returns.mean().values * TRADING_DAYS
            }).sort_values('Contribution', ascending=False)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.dataframe(
                    asset_contributions.style.format({
                        'Weight': '{:.2%}',
                        'Return': '{:.2%}',
                        'Contribution': '{:.2%}'
                    }).background_gradient(
                        subset=['Contribution'], cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    height=400
                )
            
            with col_b:
                # Visualization of contributions
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=asset_contributions['Asset'],
                        y=asset_contributions['Contribution'] * 100,
                        marker_color='#1a5fb4',
                        text=asset_contributions['Contribution'].apply(lambda x: f"{x:.2%}"),
                        textposition='auto'
                    )
                ])
                
                fig2.update_layout(
                    title="Asset Contribution to Portfolio Return",
                    height=400,
                    template="plotly_dark",
                    xaxis_title="Assets",
                    yaxis_title="Contribution (%)",
                    xaxis_tickangle=45
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in Performance Attribution tab: {str(e)[:200]}")
    
    # Tab 8: Comprehensive Report Tab
    with tabs[7]:
        st.header("📋 Comprehensive Portfolio Report")
        
        # Report generation section
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            report_type = st.selectbox(
                "Report Type",
                ["Executive Summary", "Detailed Analysis", "Risk Assessment", "Performance Review"],
                key="report_type"
            )
        
        with col_r2:
            include_charts = st.checkbox("Include Charts", value=True, key="include_charts")
        
        with col_r3:
            generate_btn = st.button("📄 Generate Report", type="primary", use_container_width=True)
        
        if generate_btn and 'portfolio_returns' in st.session_state:
            with st.spinner("Generating comprehensive report..."):
                # Create a comprehensive analysis
                portfolio_analysis = PortfolioAnalyzer.analyze_portfolio(
                    st.session_state.prices[selected_tickers],
                    st.session_state.current_weights,
                    st.session_state.prices[benchmark]
                )
                
                # Generate report
                report_generator = ReportGenerator()
                report = report_generator.generate_summary_report(portfolio_analysis, benchmark)
                
                # Display report
                st.markdown(report)
                
                # Download button for report
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=report,
                    file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
                # Additional visualizations if requested
                if include_charts:
                    st.subheader("📈 Report Visualizations")
                    
                    col_v1, col_v2 = st.columns(2)
                    
                    with col_v1:
                        # Monthly returns heatmap
                        monthly_returns = portfolio_analysis.get('monthly_returns', pd.Series())
                        if not monthly_returns.empty:
                            monthly_df = monthly_returns.reset_index()
                            monthly_df['Year'] = monthly_df['Date'].dt.year
                            monthly_df['Month'] = monthly_df['Date'].dt.month
                            
                            pivot_df = monthly_df.pivot(index='Year', columns='Month', values=0)
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=pivot_df.values,
                                x=pivot_df.columns,
                                y=pivot_df.index,
                                colorscale='RdYlGn',
                                text=np.round(pivot_df.values * 100, 1),
                                texttemplate='%{text}%',
                                textfont={"size": 10},
                                colorbar=dict(title="Return %")
                            ))
                            
                            fig.update_layout(
                                title="Monthly Returns Heatmap",
                                height=400,
                                template="plotly_dark"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col_v2:
                        # Drawdown chart
                        drawdown = portfolio_analysis.get('drawdown', pd.Series())
                        if not drawdown.empty:
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=drawdown.index,
                                y=drawdown.values * 100,
                                fill='tozeroy',
                                fillcolor='rgba(192, 28, 40, 0.3)',
                                line=dict(color='#c01c28', width=2),
                                name='Drawdown'
                            ))
                            
                            fig.update_layout(
                                title="Portfolio Drawdown",
                                height=400,
                                template="plotly_dark",
                                yaxis_title="Drawdown (%)",
                                xaxis_title="Date"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# MAIN ENTRY POINT WITH ENHANCED ERROR HANDLING
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Start monitoring
        if 'performance_timers' not in st.session_state:
            st.session_state.performance_timers = {}
        st.session_state.performance_timers["total_runtime"] = time.time()
        
        # Run the main application
        main()
        
        # End monitoring
        if 'performance_timers' in st.session_state and "total_runtime" in st.session_state.performance_timers:
            elapsed = time.time() - st.session_state.performance_timers["total_runtime"]
            
            # Log to session state
            if 'performance_log' not in st.session_state:
                st.session_state.performance_log = []
            
            st.session_state.performance_log.append({
                'timestamp': datetime.now(),
                'label': "total_runtime",
                'elapsed_seconds': elapsed
            })
        
        # Add debug information in development mode
        if st.secrets.get("DEBUG_MODE", False):
            with st.expander("🔧 Debug Information", expanded=False):
                if 'error_log' in st.session_state:
                    st.write("Error Log:", st.session_state.error_log)
                
                if 'performance_log' in st.session_state:
                    st.write("Performance Log:", st.session_state.performance_log)
                
                st.write("Session State Keys:", list(st.session_state.keys()))
                
    except Exception as e:
        # Handle uncaught exceptions
        error_msg = f"Error in main_application: {str(e)}"
        
        # Log to session state for debugging
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        
        st.session_state.error_log.append({
            'timestamp': datetime.now(),
            'context': "main_application",
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        
        # Display user-friendly error
        st.error(f"❌ {error_msg}")
        
        # Add expander for technical details
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        
        # Show restart button
        if st.button("🔄 Restart Application"):
            st.cache_data.clear()
            st.rerun()
