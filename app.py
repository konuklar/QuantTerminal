# =============================================================
# 🏛️ APOLLO/ENIGMA QUANT TERMINAL v7.0 - INSTITUTIONAL EDITION
# Professional Global Multi-Asset Portfolio Management System
# Enhanced with Quantitative Analysis & Institutional Reporting
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
import traceback
import base64
from io import BytesIO

# Import PyPortfolioOpt
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

# Import Quantstats for advanced performance analytics
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False

# Check for scikit-learn
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# ENHANCED LIGHT THEME WITH WHITE BACKGROUND
# -------------------------------------------------------------
st.set_page_config(
    page_title="APOLLO/ENIGMA - Institutional Portfolio Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏛️"
)

# Enhanced light theme with white background
LIGHT_THEME_CSS = """
<style>
:root {
    --primary: #0a3d62;
    --primary-light: #1a5fb4;
    --primary-dark: #082c47;
    --secondary: #26a269;
    --secondary-light: #2ecc71;
    --secondary-dark: #1e864d;
    --accent: #f39c12;
    --accent-light: #f1c40f;
    --accent-dark: #d68910;
    --danger: #e74c3c;
    --danger-light: #ff6b6b;
    --danger-dark: #c0392b;
    --warning: #f1c40f;
    --warning-light: #f7dc6f;
    --warning-dark: #f39c12;
    --success: #27ae60;
    --success-light: #58d68d;
    --success-dark: #229954;
    --info: #3498db;
    --info-light: #5dade2;
    --info-dark: #2980b9;
    
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fa;
    --bg-card: #ffffff;
    --bg-sidebar: #f1f3f5;
    
    --text-primary: #212529;
    --text-secondary: #495057;
    --text-muted: #6c757d;
    --text-light: #adb5bd;
    
    --border: #dee2e6;
    --border-light: #e9ecef;
    --border-dark: #ced4da;
    
    --shadow: rgba(0, 0, 0, 0.1);
    --shadow-light: rgba(0, 0, 0, 0.05);
    --shadow-dark: rgba(0, 0, 0, 0.15);
}

/* Main background */
.main {
    background-color: var(--bg-primary);
}

/* Professional Header */
.institutional-header {
    background: linear-gradient(90deg, var(--primary), var(--primary-dark));
    padding: 1.5rem;
    border-radius: 0 0 15px 15px;
    margin: -1rem -1rem 2rem -1rem;
    box-shadow: 0 4px 20px var(--shadow);
    border-bottom: 3px solid var(--accent);
}

/* Superior Cards */
.institutional-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 10px var(--shadow-light);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.institutional-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px var(--shadow);
    border-color: var(--primary-light);
}

.institutional-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
}

/* Professional Metrics */
.institutional-metric {
    background: linear-gradient(135deg, rgba(10, 61, 98, 0.05), rgba(26, 95, 180, 0.02));
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.institutional-metric:hover {
    background: linear-gradient(135deg, rgba(10, 61, 98, 0.1), rgba(26, 95, 180, 0.05));
    border-color: var(--primary-light);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* Enhanced Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-secondary);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    color: var(--text-muted);
    border: 1px solid transparent;
    transition: all 0.3s ease;
    font-size: 0.9rem;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(10, 61, 98, 0.1);
    color: var(--text-primary);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    border-color: var(--accent) !important;
    box-shadow: 0 4px 15px rgba(10, 61, 98, 0.2);
}

/* Professional Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s;
    box-shadow: 0 4px 12px rgba(10, 61, 98, 0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(10, 61, 98, 0.3);
}

/* Status Indicators */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-success {
    background: linear-gradient(135deg, var(--success), var(--success-dark));
    color: white;
}

.badge-warning {
    background: linear-gradient(135deg, var(--warning), var(--warning-dark));
    color: #212529;
}

.badge-danger {
    background: linear-gradient(135deg, var(--danger), var(--danger-dark));
    color: white;
}

.badge-info {
    background: linear-gradient(135deg, var(--info), var(--info-dark));
    color: white;
}

/* Enhanced Tables */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.stDataFrame table {
    background: var(--bg-card) !important;
}

.stDataFrame th {
    background: var(--primary) !important;
    color: white !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent) !important;
}

.stDataFrame td {
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border-light) !important;
}

.stDataFrame tr:hover {
    background: rgba(10, 61, 98, 0.05) !important;
}

/* Professional Charts */
.js-plotly-plot {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--bg-card);
}

/* Sidebar Enhancement */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
}

/* Input Enhancements */
.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stSelectbox > div > div:hover, .stTextInput > div > div:hover {
    border-color: var(--primary-light) !important;
    box-shadow: 0 0 0 2px rgba(26, 95, 180, 0.1) !important;
}

/* Progress Bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 4px;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary-light);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* Professional Loading */
.stSpinner > div {
    border: 4px solid var(--border);
    border-top: 4px solid var(--primary);
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Grid Layout */
.institutional-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

/* Expander Headers */
.streamlit-expanderHeader {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-weight: 600;
    padding: 1rem;
}

.streamlit-expanderHeader:hover {
    background: var(--bg-sidebar);
    border-color: var(--primary-light);
}

/* Alert Enhancements */
.stAlert {
    border-radius: 10px !important;
    border: 1px solid !important;
}

/* Footer */
.institutional-footer {
    background: var(--bg-secondary);
    padding: 1rem;
    margin-top: 3rem;
    border-radius: 10px;
    border-top: 2px solid var(--accent);
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8rem;
}

/* KPI Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px var(--shadow-light);
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px var(--shadow);
    border-color: var(--primary-light);
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
    margin: 0.5rem 0;
}

.kpi-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Report Cards */
.report-card {
    background: linear-gradient(135deg, rgba(26, 95, 180, 0.05), rgba(38, 162, 105, 0.05));
    border: 2px solid var(--primary);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Section Headers */
.section-header {
    background: linear-gradient(90deg, rgba(10, 61, 98, 0.1), transparent);
    border-left: 4px solid var(--accent);
    padding: 1rem 1.5rem;
    margin: 2rem 0 1rem 0;
    border-radius: 0 8px 8px 0;
}

/* Download Buttons */
.download-button {
    background: linear-gradient(135deg, var(--secondary), var(--secondary-dark)) !important;
}

.download-button:hover {
    background: linear-gradient(135deg, var(--secondary-dark), #1e864d) !important;
}

/* Tab Content */
.tab-content {
    padding: 1rem 0;
}

/* Data Table Styles */
.data-table {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}

/* Chart Container */
.chart-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1.5rem;
}

/* Tooltip */
.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted var(--text-muted);
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: var(--primary-dark);
    color: white;
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.8rem;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
"""

st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------
# ENHANCED GLOBAL ASSET UNIVERSE (WITH SPY AS DEFAULT BENCHMARK)
# -------------------------------------------------------------
GLOBAL_ASSET_UNIVERSE = {
    "Core Equities": {
        "description": "Major equity indices and broad market ETFs",
        "assets": ["SPY", "QQQ", "IWM", "VTI", "VOO", "IVV", "VEA", "VWO"]
    },
    "Fixed Income": {
        "description": "Government and corporate bonds across durations",
        "assets": ["TLT", "IEF", "SHY", "BND", "AGG", "LQD", "TIP", "MUB"]
    },
    "Commodities": {
        "description": "Precious metals, energy, and agriculture",
        "assets": ["GLD", "SLV", "USO", "DBA", "PDBC", "GSG"]
    },
    "Alternatives": {
        "description": "Cryptocurrencies, volatility, real estate",
        "assets": ["BTC-USD", "ETH-USD", "VNQ", "IYR", "REM"]
    },
    "Sector ETFs": {
        "description": "Sector-specific equity exposure",
        "assets": ["XLK", "XLV", "XLF", "XLE", "XLI", "XLP", "XLY", "XLU"]
    },
    "Global Markets": {
        "description": "International and emerging markets",
        "assets": ["EEM", "EWJ", "FEZ", "EWU", "EWG", "EWC", "EWA", "EWZ"]
    },
    "Factor Investing": {
        "description": "Smart beta and factor-based strategies",
        "assets": ["MTUM", "QUAL", "VLUE", "SIZE", "USMV", "LRGF"]
    },
    "Sustainability": {
        "description": "ESG and clean energy focused",
        "assets": ["ESGU", "ICLN", "TAN", "PBW", "QCLN", "PBD"]
    }
}

# Flatten for selection
ALL_TICKERS = []
for category in GLOBAL_ASSET_UNIVERSE.values():
    ALL_TICKERS.extend(category["assets"])
ALL_TICKERS = sorted(list(set(ALL_TICKERS)))

# Default benchmark
DEFAULT_BENCHMARK = "SPY"

# -------------------------------------------------------------
# ENHANCED PORTFOLIO STRATEGIES
# -------------------------------------------------------------
PORTFOLIO_STRATEGIES = {
    "Institutional Balanced": "40% Equities, 40% Bonds, 20% Alternatives",
    "Risk Parity": "Equal risk contribution across asset classes",
    "Minimum Volatility": "Optimized for lowest portfolio volatility",
    "Maximum Sharpe": "Optimal risk-adjusted returns",
    "Equal Weight": "Equal allocation across all selected assets",
    "Maximum Diversification": "Maximizes diversification ratio",
    "Mean-Variance Optimal": "Classical Markowitz optimization"
}

# -------------------------------------------------------------
# ENHANCED DATA LOADER WITH ERROR HANDLING AND BENCHMARK SUPPORT
# -------------------------------------------------------------
class EnhancedDataLoader:
    """Enhanced data loader with robust error handling and benchmark support"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_price_data(tickers: List[str], start_date: str, end_date: str, benchmark: str = DEFAULT_BENCHMARK) -> Dict:
        """Load price data with enhanced error handling and benchmark"""
        
        if not tickers:
            return {'prices': pd.DataFrame(), 'benchmark': pd.Series()}
        
        st.info(f"📊 Loading data for {len(tickers)} assets and benchmark {benchmark}...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        prices_dict = {}
        successful_tickers = []
        failed_tickers = []
        
        # Load all tickers including benchmark
        all_tickers = tickers + [benchmark] if benchmark not in tickers else tickers
        
        for i, ticker in enumerate(all_tickers):
            try:
                # Download data
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True
                )
                
                if not hist.empty and len(hist) > 20:
                    # Use adjusted close if available
                    if 'Close' in hist.columns:
                        price_series = hist['Close']
                    elif 'Adj Close' in hist.columns:
                        price_series = hist['Adj Close']
                    else:
                        continue
                    
                    # Handle extreme outliers
                    median_price = price_series.median()
                    std_price = price_series.std()
                    if std_price > 0:
                        lower_bound = median_price - 10 * std_price
                        upper_bound = median_price + 10 * std_price
                        price_series = price_series.clip(lower_bound, upper_bound)
                    
                    prices_dict[ticker] = price_series
                    successful_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                failed_tickers.append(ticker)
            
            # Update progress
            progress = (i + 1) / len(all_tickers)
            progress_bar.progress(progress)
            status_text.text(f"Loaded {i + 1}/{len(all_tickers)} assets...")
        
        progress_bar.empty()
        status_text.empty()
        
        if prices_dict:
            prices_df = pd.DataFrame(prices_dict)
            
            # Fill missing values (limited forward fill)
            prices_df = prices_df.ffill(limit=5).bfill(limit=5)
            
            # Remove assets with too much missing data
            missing_pct = prices_df.isna().mean()
            valid_columns = missing_pct[missing_pct < 0.3].index.tolist()
            prices_df = prices_df[valid_columns]
            
            # Extract benchmark
            benchmark_series = pd.Series()
            if benchmark in prices_df.columns:
                benchmark_series = prices_df[benchmark]
                # Remove benchmark from portfolio assets if it's not in the original tickers
                if benchmark not in tickers:
                    prices_df = prices_df.drop(columns=[benchmark])
            
            if len(prices_df.columns) >= 2:
                st.success(f"✅ Successfully loaded {len(prices_df.columns)} assets")
                if failed_tickers:
                    with st.expander(f"⚠️ Failed to load {len(failed_tickers)} assets", expanded=False):
                        st.write(", ".join(failed_tickers[:10]))
                return {
                    'prices': prices_df,
                    'benchmark': benchmark_series
                }
            else:
                st.error("❌ Insufficient data for analysis")
                return {'prices': pd.DataFrame(), 'benchmark': pd.Series()}
        
        else:
            st.error("❌ Could not load any data")
            return {'prices': pd.DataFrame(), 'benchmark': pd.Series()}

# -------------------------------------------------------------
# RISK METRICS CALCULATOR (INCLUDING VaR)
# -------------------------------------------------------------
class RiskMetricsCalculator:
    """Calculate various risk metrics including VaR"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95, method: str = 'historical') -> float:
        """Calculate Value at Risk (VaR)"""
        
        if returns.empty:
            return 0.0
        
        try:
            if method == 'historical':
                # Historical VaR
                var = np.percentile(returns, (1 - confidence_level) * 100)
            elif method == 'parametric':
                # Parametric VaR (assuming normal distribution)
                mean = returns.mean()
                std = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                var = mean + z_score * std
            elif method == 'cornish_fisher':
                # Cornish-Fisher expansion (adjusted for skewness and kurtosis)
                mean = returns.mean()
                std = returns.std()
                skew = returns.skew()
                kurt = returns.kurtosis()
                
                z = stats.norm.ppf(1 - confidence_level)
                z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * (kurt - 3) / 24 - (2*z**3 - 5*z) * skew**2 / 36
                var = mean + z_cf * std
            else:
                var = np.percentile(returns, (1 - confidence_level) * 100)
            
            return float(var)
            
        except Exception:
            return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)"""
        
        if returns.empty:
            return 0.0
        
        try:
            var = RiskMetricsCalculator.calculate_var(returns, confidence_level, 'historical')
            # CVaR is the average of returns worse than VaR
            losses_below_var = returns[returns <= var]
            if len(losses_below_var) > 0:
                cvar = losses_below_var.mean()
            else:
                cvar = var
            
            return float(cvar)
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        
        if returns.empty:
            return 0.0
        
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            return float(max_dd)
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_all_risk_metrics(returns: pd.Series, window: int = 252) -> Dict:
        """Calculate comprehensive risk metrics"""
        
        if returns.empty:
            return {}
        
        try:
            metrics = {}
            
            # Basic statistics
            metrics['mean'] = returns.mean() * window
            metrics['std'] = returns.std() * np.sqrt(window)
            metrics['skew'] = returns.skew()
            metrics['kurtosis'] = returns.kurtosis()
            
            # VaR metrics
            metrics['var_95_historical'] = RiskMetricsCalculator.calculate_var(returns, 0.95, 'historical')
            metrics['var_95_parametric'] = RiskMetricsCalculator.calculate_var(returns, 0.95, 'parametric')
            metrics['var_95_cornish_fisher'] = RiskMetricsCalculator.calculate_var(returns, 0.95, 'cornish_fisher')
            
            # CVaR metrics
            metrics['cvar_95'] = RiskMetricsCalculator.calculate_cvar(returns, 0.95)
            
            # Drawdown metrics
            metrics['max_drawdown'] = RiskMetricsCalculator.calculate_max_drawdown(returns)
            
            # Volatility metrics
            metrics['annual_volatility'] = returns.std() * np.sqrt(window)
            metrics['downside_deviation'] = returns[returns < 0].std() * np.sqrt(window) if len(returns[returns < 0]) > 0 else 0
            
            # Ratio metrics
            sharpe = (metrics['mean'] - 0.03) / (metrics['std'] + 1e-10)
            metrics['sharpe'] = sharpe
            
            if metrics['downside_deviation'] > 0:
                sortino = (metrics['mean'] - 0.03) / (metrics['downside_deviation'] + 1e-10)
            else:
                sortino = 0
            metrics['sortino'] = sortino
            
            if abs(metrics['max_drawdown']) > 0:
                calmar = metrics['mean'] / abs(metrics['max_drawdown'])
            else:
                calmar = 0
            metrics['calmar'] = calmar
            
            # Win rate
            metrics['win_rate'] = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            
            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            return metrics
            
        except Exception as e:
            st.warning(f"Risk metrics calculation error: {str(e)[:100]}")
            return {}

# -------------------------------------------------------------
# QUANTSTATS INTEGRATION WITH ENHANCED RISK METRICS
# -------------------------------------------------------------
class QuantStatsAnalytics:
    """QuantStats integration for advanced performance analytics"""
    
    @staticmethod
    def generate_performance_report(returns: pd.Series, benchmark: pd.Series = None, 
                                   rf_rate: float = 0.03) -> Dict:
        """Generate comprehensive performance report"""
        
        if returns.empty:
            return RiskMetricsCalculator.calculate_all_risk_metrics(returns)
        
        try:
            # Always use our own calculations for reliability
            metrics = RiskMetricsCalculator.calculate_all_risk_metrics(returns)
            
            # Add CAGR calculation
            if len(returns) > 0:
                total_return = (1 + returns).prod() - 1
                years = len(returns) / 252
                metrics['cagr'] = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            
            # Add benchmark metrics if available
            if benchmark is not None and not benchmark.empty:
                aligned_returns = returns.reindex(benchmark.index).dropna()
                aligned_benchmark = benchmark.reindex(aligned_returns.index)
                
                if len(aligned_returns) > 10:
                    # Calculate beta
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = aligned_benchmark.var()
                    metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    
                    # Calculate alpha
                    portfolio_return = aligned_returns.mean() * 252
                    benchmark_return = aligned_benchmark.mean() * 252
                    metrics['alpha'] = portfolio_return - rf_rate - metrics.get('beta', 0) * (benchmark_return - rf_rate)
                    
                    # Calculate information ratio
                    excess_returns = aligned_returns - aligned_benchmark
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    metrics['information_ratio'] = (excess_returns.mean() * 252) / (tracking_error + 1e-10)
            
            return metrics
            
        except Exception as e:
            st.warning(f"Performance calculation error: {str(e)[:100]}")
            return RiskMetricsCalculator.calculate_all_risk_metrics(returns)
    
    @staticmethod
    def create_performance_charts(returns: pd.Series, benchmark: pd.Series = None) -> List[go.Figure]:
        """Create performance charts"""
        
        charts = []
        
        if returns.empty:
            return charts
        
        try:
            # 1. Cumulative Returns Chart with Benchmark
            cumulative_returns = (1 + returns).cumprod()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                name="Portfolio",
                line=dict(color='#1a5fb4', width=3),
                fill='tozeroy',
                fillcolor='rgba(26, 95, 180, 0.1)'
            ))
            
            if benchmark is not None and not benchmark.empty:
                benchmark_cumulative = (1 + benchmark).cumprod()
                fig1.add_trace(go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    name=f"Benchmark ({DEFAULT_BENCHMARK})",
                    line=dict(color='#26a269', width=2, dash='dash')
                ))
            
            fig1.update_layout(
                title="Cumulative Returns",
                height=400,
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            charts.append(fig1)
            
            # 2. Drawdown Chart
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                name="Drawdown",
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='#e74c3c', width=2)
            ))
            
            fig2.update_layout(
                title="Portfolio Drawdown",
                height=400,
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                yaxis=dict(ticksuffix="%")
            )
            charts.append(fig2)
            
            # 3. Return Distribution with VaR
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=returns.values * 100,
                nbinsx=50,
                name="Returns",
                marker_color='#1a5fb4',
                opacity=0.7,
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ))
            
            # Add mean line
            mean_return = returns.mean() * 100
            fig3.add_vline(
                x=mean_return,
                line_dash="dash",
                line_color="#27ae60",
                annotation_text=f"Mean: {mean_return:.2f}%",
                annotation_position="top right"
            )
            
            # Add VaR lines
            var_95 = RiskMetricsCalculator.calculate_var(returns, 0.95, 'historical') * 100
            fig3.add_vline(
                x=var_95,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"VaR 95%: {var_95:.2f}%",
                annotation_position="top left"
            )
            
            fig3.update_layout(
                title="Return Distribution with VaR",
                height=400,
                template="plotly_white",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                xaxis=dict(ticksuffix="%")
            )
            charts.append(fig3)
            
            # 4. Rolling Sharpe Ratio
            if len(returns) > 60:
                window = min(60, len(returns) // 4)
                rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
                
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    name="Rolling Sharpe",
                    line=dict(color='#9b59b6', width=2)
                ))
                
                fig4.update_layout(
                    title=f"{window}-Day Rolling Sharpe Ratio",
                    height=400,
                    template="plotly_white",
                    xaxis_title="Date",
                    yaxis_title="Sharpe Ratio",
                    hovermode='x unified'
                )
                charts.append(fig4)
            
        except Exception as e:
            st.warning(f"Chart generation error: {str(e)[:100]}")
        
        return charts

# -------------------------------------------------------------
# ENHANCED PORTFOLIO OPTIMIZER
# -------------------------------------------------------------
class PortfolioOptimizer:
    """Enhanced portfolio optimizer"""
    
    @staticmethod
    def optimize_portfolio(returns: pd.DataFrame, strategy: str, 
                          risk_free_rate: float = 0.03, 
                          constraints: Dict = None,
                          benchmark_returns: pd.Series = None) -> Dict:
        """Optimize portfolio using specified strategy"""
        
        if returns.empty or len(returns.columns) < 2:
            return PortfolioOptimizer._equal_weight_fallback(returns, benchmark_returns)
        
        # Default constraints
        if constraints is None:
            constraints = {
                'max_weight': 0.20,
                'min_weight': 0.01,
                'short_selling': False
            }
        
        try:
            # Calculate expected returns and covariance
            mu = returns.mean() * 252  # Annualized returns
            S = returns.cov() * 252    # Annualized covariance
            
            # Strategy-specific optimization
            weights = None
            
            if strategy == "Minimum Volatility":
                weights = PortfolioOptimizer._min_volatility_optimization(S)
            elif strategy == "Maximum Sharpe":
                weights = PortfolioOptimizer._max_sharpe_optimization(mu, S, risk_free_rate)
            elif strategy == "Risk Parity":
                weights = PortfolioOptimizer._risk_parity_optimization(returns)
            elif strategy == "Equal Weight":
                weights = PortfolioOptimizer._equal_weight_optimization(returns)
            elif strategy == "Institutional Balanced":
                weights = PortfolioOptimizer._institutional_balanced(returns.columns)
            elif strategy == "Maximum Diversification":
                weights = PortfolioOptimizer._max_diversification_optimization(S)
            else:
                # Fallback to equal weights
                weights = PortfolioOptimizer._equal_weight_optimization(returns)
            
            # Apply constraints
            weights = PortfolioOptimizer._apply_constraints(weights, constraints)
            
            # Calculate portfolio metrics
            return PortfolioOptimizer._calculate_portfolio_metrics(
                weights, returns, mu, S, risk_free_rate, benchmark_returns
            )
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)[:200]}")
            return PortfolioOptimizer._equal_weight_fallback(returns, benchmark_returns)
    
    @staticmethod
    def _min_volatility_optimization(S: pd.DataFrame) -> Dict:
        """Minimum volatility optimization"""
        try:
            n = len(S)
            
            # Objective function: w'Σw
            def objective(w):
                return w @ S.values @ w
            
            # Constraints: sum(w) = 1
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # Bounds: no short selling
            bounds = [(0, 1) for _ in range(n)]
            
            # Initial guess (equal weights)
            w0 = np.ones(n) / n
            
            # Optimize
            result = optimize.minimize(
                objective, w0, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                weights = {asset: result.x[i] for i, asset in enumerate(S.index)}
                return weights
            else:
                raise Exception("Optimization failed")
                
        except Exception:
            # Fallback to inverse volatility
            return PortfolioOptimizer._inverse_volatility_weights(S)
    
    @staticmethod
    def _max_sharpe_optimization(mu: pd.Series, S: pd.DataFrame, risk_free_rate: float) -> Dict:
        """Maximum Sharpe ratio optimization"""
        try:
            n = len(S)
            
            # Objective: minimize -Sharpe ratio = -(μ'w - rf)/sqrt(w'Σw)
            def objective(w):
                portfolio_return = mu.values @ w
                portfolio_risk = np.sqrt(w @ S.values @ w)
                sharpe = (portfolio_return - risk_free_rate) / (portfolio_risk + 1e-10)
                return -sharpe  # Minimize negative Sharpe = maximize Sharpe
            
            # Constraints: sum(w) = 1
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # Bounds: no short selling
            bounds = [(0, 1) for _ in range(n)]
            
            # Initial guess (equal weights)
            w0 = np.ones(n) / n
            
            # Optimize
            result = optimize.minimize(
                objective, w0, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                weights = {asset: result.x[i] for i, asset in enumerate(S.index)}
                return weights
            else:
                raise Exception("Optimization failed")
                
        except Exception:
            # Fallback to mean-variance with regularization
            return PortfolioOptimizer._regularized_mean_variance(mu, S)
    
    @staticmethod
    def _risk_parity_optimization(returns: pd.DataFrame) -> Dict:
        """Risk parity optimization"""
        try:
            # Simple risk parity: weights inversely proportional to volatility
            volatilities = returns.std() * np.sqrt(252)
            inv_vol = 1 / (volatilities + 1e-10)
            weights_raw = inv_vol / inv_vol.sum()
            weights = {asset: weights_raw[asset] for asset in returns.columns}
            return weights
        except Exception:
            return PortfolioOptimizer._equal_weight_optimization(returns)
    
    @staticmethod
    def _max_diversification_optimization(S: pd.DataFrame) -> Dict:
        """Maximum diversification optimization"""
        try:
            n = len(S)
            volatilities = np.sqrt(np.diag(S.values))
            
            # Objective: maximize diversification ratio
            def objective(w):
                weighted_vol = volatilities @ w
                portfolio_risk = np.sqrt(w @ S.values @ w)
                diversification = weighted_vol / (portfolio_risk + 1e-10)
                return -diversification  # Minimize negative diversification
            
            # Constraints: sum(w) = 1
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # Bounds: no short selling
            bounds = [(0, 1) for _ in range(n)]
            
            # Initial guess (equal weights)
            w0 = np.ones(n) / n
            
            # Optimize
            result = optimize.minimize(
                objective, w0, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                weights = {asset: result.x[i] for i, asset in enumerate(S.index)}
                return weights
            else:
                raise Exception("Optimization failed")
                
        except Exception:
            return PortfolioOptimizer._inverse_volatility_weights(S)
    
    @staticmethod
    def _institutional_balanced(assets: List[str]) -> Dict:
        """Institutional balanced allocation (40/40/20 rule)"""
        weights = {}
        
        # Categorize assets
        equity_etfs = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'IVV', 'XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU']
        bond_etfs = ['TLT', 'IEF', 'SHY', 'BND', 'AGG', 'LQD', 'TIP', 'MUB']
        alternative_etfs = ['GLD', 'SLV', 'USO', 'DBA', 'PDBC', 'GSG', 'VNQ', 'IYR', 'REM']
        
        equity_count = sum(1 for asset in assets if asset in equity_etfs)
        bond_count = sum(1 for asset in assets if asset in bond_etfs)
        alt_count = sum(1 for asset in assets if asset in alternative_etfs)
        
        total_categorized = equity_count + bond_count + alt_count
        
        if total_categorized > 0:
            for asset in assets:
                if asset in equity_etfs:
                    weights[asset] = 0.4 / equity_count if equity_count > 0 else 0
                elif asset in bond_etfs:
                    weights[asset] = 0.4 / bond_count if bond_count > 0 else 0
                elif asset in alternative_etfs:
                    weights[asset] = 0.2 / alt_count if alt_count > 0 else 0
                else:
                    weights[asset] = 0
        else:
            # Fallback to equal weights
            n = len(assets)
            weights = {asset: 1.0/n for asset in assets}
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    @staticmethod
    def _inverse_volatility_weights(S: pd.DataFrame) -> Dict:
        """Inverse volatility weighting"""
        volatilities = np.sqrt(np.diag(S.values))
        inv_vol = 1 / (volatilities + 1e-10)
        weights_raw = inv_vol / inv_vol.sum()
        weights = {asset: weights_raw[i] for i, asset in enumerate(S.index)}
        return weights
    
    @staticmethod
    def _regularized_mean_variance(mu: pd.Series, S: pd.DataFrame) -> Dict:
        """Regularized mean-variance optimization"""
        n = len(S)
        gamma = 0.5  # Regularization parameter
        
        # Objective: maximize μ'w - γ * w'Σw
        def objective(w):
            return -(mu.values @ w - gamma * (w @ S.values @ w))
        
        # Constraints: sum(w) = 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds: no short selling
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess (equal weights)
        w0 = np.ones(n) / n
        
        # Optimize
        result = optimize.minimize(
            objective, w0, method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        if result.success:
            weights = {asset: result.x[i] for i, asset in enumerate(S.index)}
            return weights
        else:
            return PortfolioOptimizer._equal_weight_optimization(pd.DataFrame(columns=S.index))
    
    @staticmethod
    def _equal_weight_optimization(returns: pd.DataFrame) -> Dict:
        """Equal weight optimization"""
        n = len(returns.columns)
        weights = {asset: 1.0/n for asset in returns.columns}
        return weights
    
    @staticmethod
    def _apply_constraints(weights: Dict, constraints: Dict) -> Dict:
        """Apply weight constraints"""
        max_weight = constraints.get('max_weight', 1.0)
        min_weight = constraints.get('min_weight', 0.0)
        short_selling = constraints.get('short_selling', False)
        
        # Normalize weights
        weight_sum = sum(weights.values())
        if weight_sum == 0:
            return weights
        
        normalized_weights = {}
        for asset, weight in weights.items():
            # Apply min/max bounds
            if not short_selling:
                weight = max(min_weight, min(max_weight, weight))
            normalized_weights[asset] = weight
        
        # Renormalize
        total_weight = sum(normalized_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in normalized_weights.items()}
        
        return normalized_weights
    
    @staticmethod
    def _calculate_portfolio_metrics(weights: Dict, returns: pd.DataFrame, 
                                    mu: pd.Series, S: pd.DataFrame, 
                                    risk_free_rate: float,
                                    benchmark_returns: pd.Series = None) -> Dict:
        """Calculate portfolio performance metrics"""
        
        # Convert weights to array
        assets = list(weights.keys())
        weights_array = np.array([weights[asset] for asset in assets])
        
        # Calculate portfolio returns
        portfolio_returns = (returns[assets] * weights_array).sum(axis=1)
        
        # Calculate performance metrics
        expected_return = mu.values @ weights_array
        expected_risk = np.sqrt(weights_array @ S.values @ weights_array)
        sharpe_ratio = (expected_return - risk_free_rate) / (expected_risk + 1e-10)
        
        # Calculate additional metrics
        quantstats_metrics = QuantStatsAnalytics.generate_performance_report(
            portfolio_returns, benchmark_returns, risk_free_rate
        )
        
        # Calculate diversification metrics
        corr_matrix = returns[assets].corr()
        portfolio_variance = weights_array @ S.values @ weights_array
        weighted_vol = np.sqrt(np.diag(S.values)) @ weights_array
        diversification_ratio = weighted_vol / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1
        
        # Calculate benchmark metrics if available
        benchmark_metrics = {}
        if benchmark_returns is not None and not benchmark_returns.empty:
            # Align returns with benchmark
            aligned_portfolio = portfolio_returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_portfolio.index)
            
            if len(aligned_portfolio) > 10:
                # Calculate beta
                covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                benchmark_variance = aligned_benchmark.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Calculate alpha
                benchmark_return = aligned_benchmark.mean() * 252
                alpha = expected_return - risk_free_rate - beta * (benchmark_return - risk_free_rate)
                
                # Calculate tracking error
                tracking_error = (aligned_portfolio - aligned_benchmark).std() * np.sqrt(252)
                
                # Calculate information ratio
                excess_return = (aligned_portfolio - aligned_benchmark).mean() * 252
                information_ratio = excess_return / (tracking_error + 1e-10)
                
                benchmark_metrics = {
                    'beta': beta,
                    'alpha': alpha,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'benchmark_return': benchmark_return,
                    'excess_return': excess_return
                }
        
        return {
            'weights': weights,
            'weights_array': weights_array,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_returns': portfolio_returns,
            'quantstats_metrics': quantstats_metrics,
            'benchmark_metrics': benchmark_metrics,
            'diversification_ratio': diversification_ratio,
            'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
            'success': True
        }
    
    @staticmethod
    def _equal_weight_fallback(returns: pd.DataFrame, benchmark_returns: pd.Series = None) -> Dict:
        """Fallback to equal weights"""
        if returns.empty:
            return {
                'weights': {},
                'weights_array': np.array([]),
                'expected_return': 0,
                'expected_risk': 0,
                'sharpe_ratio': 0,
                'portfolio_returns': pd.Series(),
                'quantstats_metrics': {},
                'benchmark_metrics': {},
                'success': False
            }
        
        n_assets = len(returns.columns)
        equal_weights = {asset: 1.0/n_assets for asset in returns.columns}
        weights_array = np.ones(n_assets) / n_assets
        
        portfolio_returns = (returns * weights_array).sum(axis=1)
        mu = returns.mean() * 252
        S = returns.cov() * 252
        expected_return = mu.values @ weights_array
        expected_risk = np.sqrt(weights_array @ S.values @ weights_array)
        sharpe_ratio = (expected_return - 0.03) / (expected_risk + 1e-10)
        
        quantstats_metrics = QuantStatsAnalytics.generate_performance_report(
            portfolio_returns, benchmark_returns, 0.03
        )
        
        return {
            'weights': equal_weights,
            'weights_array': weights_array,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_returns': portfolio_returns,
            'quantstats_metrics': quantstats_metrics,
            'benchmark_metrics': {},
            'diversification_ratio': 0.5,
            'avg_correlation': 0.5,
            'success': False
        }

# -------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------
def main():
    """Main application with enhanced features"""
    
    # Check dependencies
    if 'deps_checked' not in st.session_state:
        check_dependencies()
        st.session_state.deps_checked = True
    
    # Custom header
    st.markdown("""
    <div class="institutional-header">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🏛️ APOLLO/ENIGMA</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 1.2rem;">
        Quantitative Portfolio Analysis Terminal v7.0
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'selected_assets' not in st.session_state:
        st.session_state.selected_assets = ["SPY", "TLT", "GLD", "IEF"]
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 1.5rem; font-weight: bold; color: var(--primary);">
                ⚙️ CONFIGURATION
            </div>
            <div style="font-size: 0.9rem; color: var(--text-muted);">
                Portfolio Settings
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Portfolio strategy
        st.markdown("### 🎯 Portfolio Strategy")
        strategy = st.selectbox(
            "Select Strategy",
            list(PORTFOLIO_STRATEGIES.keys()),
            index=0,
            help=PORTFOLIO_STRATEGIES.get(list(PORTFOLIO_STRATEGIES.keys())[0], "")
        )
        
        if strategy in PORTFOLIO_STRATEGIES:
            st.caption(f"📝 {PORTFOLIO_STRATEGIES[strategy]}")
        
        # Benchmark selection
        st.markdown("### 📈 Benchmark")
        benchmark = st.selectbox(
            "Select Benchmark",
            ["SPY", "QQQ", "VTI", "IWM", "AGG"],
            index=0,
            help="Select benchmark for performance comparison"
        )
        
        # Asset selection
        st.markdown("### 🌍 Asset Selection")
        
        # Category selection
        categories = st.multiselect(
            "Asset Categories",
            list(GLOBAL_ASSET_UNIVERSE.keys()),
            default=["Core Equities", "Fixed Income", "Commodities"],
            help="Select asset categories"
        )
        
        # Get assets from selected categories
        available_assets = []
        for category in categories:
            available_assets.extend(GLOBAL_ASSET_UNIVERSE[category]["assets"])
        available_assets = sorted(list(set(available_assets)))
        
        # Asset search
        asset_search = st.text_input("🔍 Search Assets", "", help="Search for specific assets")
        
        if asset_search:
            filtered_assets = [a for a in available_assets if asset_search.upper() in a.upper()]
        else:
            filtered_assets = available_assets
        
        # Select assets
        selected_assets = st.multiselect(
            "Select Assets (3-10 recommended)",
            filtered_assets,
            default=st.session_state.selected_assets,
            help="Select 3-10 assets for optimal diversification"
        )
        
        st.session_state.selected_assets = selected_assets
        
        # Date range
        st.markdown("### 📅 Analysis Period")
        date_period = st.selectbox(
            "Time Horizon",
            ["1 Year", "3 Years", "5 Years", "10 Years", "Max Available"],
            index=2
        )
        
        # Map to years
        years_map = {"1 Year": 1, "3 Years": 3, "5 Years": 5, "10 Years": 10, "Max Available": 20}
        years = years_map[date_period]
        
        # Risk parameters
        st.markdown("### ⚖️ Risk Parameters")
        rf_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1
        ) / 100
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                max_weight = st.slider("Max Weight (%)", 5, 100, 20, 5) / 100
            with col2:
                min_weight = st.slider("Min Weight (%)", 0, 10, 1, 1) / 100
            
            short_selling = st.checkbox("Allow Short Selling", value=False)
            
            # VaR confidence level
            var_confidence = st.slider("VaR Confidence Level (%)", 90, 99, 95, 1) / 100
        
        constraints = {
            'max_weight': max_weight,
            'min_weight': min_weight,
            'short_selling': short_selling
        }
        
        # Action buttons
        st.markdown("---")
        col_run, col_reset = st.columns(2)
        with col_run:
            run_analysis = st.button(
                "🚀 Run Analysis",
                type="primary",
                use_container_width=True,
                disabled=len(selected_assets) < 3
            )
        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        # Status indicators
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Assets", len(selected_assets))
        with col2:
            status = "✅" if PYPFOPT_AVAILABLE else "⚠️"
            st.metric("Optimizer", status)
        with col3:
            status = "✅" if QUANTSTATS_AVAILABLE else "⚠️"
            st.metric("Analytics", status)
    
    # Main content
    if run_analysis and len(selected_assets) >= 3:
        with st.spinner("🔍 Conducting quantitative analysis..."):
            try:
                # Set date range
                end_date = pd.Timestamp.today()
                start_date = end_date - pd.DateOffset(years=years)
                
                # Load data
                data_loader = EnhancedDataLoader()
                data = data_loader.load_price_data(selected_assets, start_date, end_date, benchmark)
                prices = data['prices']
                benchmark_series = data['benchmark']
                
                if prices.empty or len(prices) < 60:
                    st.error("❌ Insufficient data for analysis.")
                    return
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                # Calculate benchmark returns if available
                benchmark_returns = pd.Series()
                if not benchmark_series.empty:
                    benchmark_returns = benchmark_series.pct_change().dropna()
                
                # Run portfolio optimization
                optimizer = PortfolioOptimizer()
                results = optimizer.optimize_portfolio(
                    returns, strategy, rf_rate, constraints, benchmark_returns
                )
                
                if results['success']:
                    # Store results
                    st.session_state.analysis_results = results
                    st.session_state.portfolio_data = {
                        'prices': prices,
                        'returns': returns,
                        'portfolio_returns': results['portfolio_returns'],
                        'benchmark_returns': benchmark_returns,
                        'benchmark': benchmark,
                        'start_date': start_date,
                        'end_date': end_date,
                        'strategy': strategy,
                        'rf_rate': rf_rate,
                        'var_confidence': var_confidence
                    }
                    
                    st.success("✅ Analysis completed successfully!")
                    
                    # Create enhanced tabs
                    tab_names = [
                        "📊 Overview", 
                        "⚖️ Risk Analytics", 
                        "📈 Performance", 
                        "🔍 QuantStats", 
                        "📋 Reports"
                    ]
                    
                    tabs = st.tabs(tab_names)
                    
                    with tabs[0]:
                        display_overview_tab(results, prices, returns, benchmark_returns, benchmark)
                    
                    with tabs[1]:
                        display_risk_tab(results, returns, var_confidence)
                    
                    with tabs[2]:
                        display_performance_tab(results, prices, benchmark_returns, benchmark)
                    
                    with tabs[3]:
                        display_quantstats_tab(results, benchmark_returns, benchmark)
                    
                    with tabs[4]:
                        display_reports_tab(results, prices, benchmark_returns, benchmark)
                
                else:
                    st.error("❌ Portfolio optimization failed.")
                    
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)[:200]}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    else:
        display_welcome_screen()

def display_overview_tab(results: Dict, prices: pd.DataFrame, returns: pd.DataFrame, 
                        benchmark_returns: pd.Series, benchmark: str):
    """Display overview tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, var(--primary), var(--primary-dark)); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📊 PORTFOLIO OVERVIEW</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Portfolio Summary & Allocation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Expected Return", 
            f"{results['expected_return']:.2%}",
            help="Annualized expected return"
        )
    
    with col2:
        st.metric(
            "Expected Risk", 
            f"{results['expected_risk']:.2%}",
            help="Annualized volatility"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio", 
            f"{results['sharpe_ratio']:.2f}",
            help="Risk-adjusted return"
        )
    
    with col4:
        if 'quantstats_metrics' in results and 'max_drawdown' in results['quantstats_metrics']:
            max_dd = results['quantstats_metrics']['max_drawdown']
            st.metric(
                "Max Drawdown", 
                f"{max_dd:.2%}",
                help="Maximum peak-to-trough decline"
            )
        else:
            st.metric("Max Drawdown", "N/A")
    
    # Benchmark comparison
    if benchmark_returns is not None and not benchmark_returns.empty:
        st.markdown("### 📈 Benchmark Comparison")
        
        benchmark_cols = st.columns(4)
        
        with benchmark_cols[0]:
            benchmark_return = benchmark_returns.mean() * 252
            st.metric(
                f"{benchmark} Return", 
                f"{benchmark_return:.2%}",
                delta=f"{(results['expected_return'] - benchmark_return):.2%}",
                delta_color="normal"
            )
        
        with benchmark_cols[1]:
            benchmark_risk = benchmark_returns.std() * np.sqrt(252)
            st.metric(
                f"{benchmark} Risk", 
                f"{benchmark_risk:.2%}",
                delta=f"{(results['expected_risk'] - benchmark_risk):.2%}",
                delta_color="normal"
            )
        
        with benchmark_cols[2]:
            if 'benchmark_metrics' in results and 'beta' in results['benchmark_metrics']:
                beta = results['benchmark_metrics']['beta']
                st.metric("Beta", f"{beta:.2f}")
            else:
                st.metric("Beta", "N/A")
        
        with benchmark_cols[3]:
            if 'benchmark_metrics' in results and 'alpha' in results['benchmark_metrics']:
                alpha = results['benchmark_metrics']['alpha']
                st.metric("Alpha", f"{alpha:.2%}")
            else:
                st.metric("Alpha", "N/A")
    
    # Portfolio allocation
    st.markdown("### 🎯 Portfolio Allocation")
    
    weights = results['weights']
    weights_df = pd.DataFrame({
        'Asset': list(weights.keys()),
        'Weight': list(weights.values()),
        'Category': [get_asset_category(asset) for asset in weights.keys()]
    }).sort_values('Weight', ascending=False)
    
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=weights_df['Asset'],
            values=weights_df['Weight'],
            hole=0.4,
            marker_colors=px.colors.sequential.Blues,
            textinfo='label+percent',
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>Weight: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        # Category allocation bar chart
        category_weights = weights_df.groupby('Category')['Weight'].sum().reset_index()
        
        fig = go.Figure(data=[go.Bar(
            x=category_weights['Weight'],
            y=category_weights['Category'],
            orientation='h',
            marker_color='var(--primary)',
            text=category_weights['Weight'].apply(lambda x: f"{x:.1%}"),
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Category Allocation",
            height=400,
            xaxis=dict(tickformat=".0%"),
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.markdown("### 📊 Detailed Performance Metrics")
    
    if 'quantstats_metrics' in results:
        metrics = results['quantstats_metrics']
        metrics_df = pd.DataFrame([
            ("Sharpe Ratio", metrics.get('sharpe', 'N/A'), "Risk-adjusted return"),
            ("Sortino Ratio", metrics.get('sortino', 'N/A'), "Downside risk-adjusted return"),
            ("Calmar Ratio", metrics.get('calmar', 'N/A'), "Return vs max drawdown"),
            ("Omega Ratio", metrics.get('omega', 'N/A'), "Gain vs loss probability ratio"),
            ("CAGR", f"{metrics.get('cagr', 0):.2%}" if metrics.get('cagr') else 'N/A', "Compound annual growth rate"),
            ("Volatility", f"{metrics.get('annual_volatility', 0):.2%}" if metrics.get('annual_volatility') else 'N/A', "Annualized volatility"),
            ("Value at Risk (95%)", f"{metrics.get('var_95_historical', 0):.2%}" if metrics.get('var_95_historical') else 'N/A', "95% confidence worst loss"),
            ("Conditional VaR (95%)", f"{metrics.get('cvar_95', 0):.2%}" if metrics.get('cvar_95') else 'N/A', "Expected shortfall"),
            ("Win Rate", f"{metrics.get('win_rate', 0):.1%}" if metrics.get('win_rate') else 'N/A', "Percentage of positive periods"),
            ("Profit Factor", f"{metrics.get('profit_factor', 'N/A')}", "Gross profit / gross loss"),
            ("Skewness", f"{metrics.get('skew', 'N/A')}", "Return distribution asymmetry"),
            ("Kurtosis", f"{metrics.get('kurtosis', 'N/A')}", "Tail risk measure"),
        ], columns=['Metric', 'Value', 'Description'])
        
        st.dataframe(
            metrics_df,
            use_container_width=True,
            height=400
        )

def display_risk_tab(results: Dict, returns: pd.DataFrame, var_confidence: float):
    """Display risk analytics tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #c0392b, #e74c3c); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">⚖️ RISK ANALYTICS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Comprehensive Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk metrics
    st.markdown("### 📊 Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Volatility", f"{results['expected_risk']:.2%}")
    
    with col2:
        if 'quantstats_metrics' in results and 'var_95_historical' in results['quantstats_metrics']:
            var = results['quantstats_metrics']['var_95_historical']
            st.metric("VaR (95%)", f"{var:.2%}")
        else:
            # Calculate VaR directly
            portfolio_returns = results['portfolio_returns']
            var = RiskMetricsCalculator.calculate_var(portfolio_returns, 0.95, 'historical')
            st.metric("VaR (95%)", f"{var:.2%}")
    
    with col3:
        if 'quantstats_metrics' in results and 'cvar_95' in results['quantstats_metrics']:
            cvar = results['quantstats_metrics']['cvar_95']
            st.metric("CVaR (95%)", f"{cvar:.2%}")
        else:
            # Calculate CVaR directly
            portfolio_returns = results['portfolio_returns']
            cvar = RiskMetricsCalculator.calculate_cvar(portfolio_returns, 0.95)
            st.metric("CVaR (95%)", f"{cvar:.2%}")
    
    with col4:
        if 'quantstats_metrics' in results and 'max_drawdown' in results['quantstats_metrics']:
            max_dd = results['quantstats_metrics']['max_drawdown']
            st.metric("Max Drawdown", f"{max_dd:.2%}")
        else:
            st.metric("Max Drawdown", "N/A")
    
    # VaR Analysis Section
    st.markdown("### 📉 Value at Risk (VaR) Analysis")
    
    portfolio_returns = results['portfolio_returns']
    
    if not portfolio_returns.empty:
        # Calculate different VaR methods
        var_methods = ['historical', 'parametric', 'cornish_fisher']
        var_results = {}
        
        for method in var_methods:
            var_results[method] = RiskMetricsCalculator.calculate_var(
                portfolio_returns, var_confidence, method
            )
        
        # Display VaR comparison
        var_cols = st.columns(3)
        method_names = {
            'historical': 'Historical',
            'parametric': 'Parametric (Normal)',
            'cornish_fisher': 'Cornish-Fisher'
        }
        
        for idx, (method, var_value) in enumerate(var_results.items()):
            with var_cols[idx]:
                st.metric(
                    f"VaR ({int(var_confidence*100)}%) - {method_names[method]}",
                    f"{var_value:.2%}",
                    help=f"{method_names[method]} VaR calculation"
                )
        
        # VaR chart
        st.markdown("#### VaR Distribution")
        
        fig = go.Figure()
        
        # Histogram of returns
        fig.add_trace(go.Histogram(
            x=portfolio_returns.values * 100,
            nbinsx=50,
            name="Returns",
            marker_color='#1a5fb4',
            opacity=0.7,
            hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
        ))
        
        # Add VaR lines
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        for idx, (method, var_value) in enumerate(var_results.items()):
            fig.add_vline(
                x=var_value * 100,
                line_dash="dash",
                line_color=colors[idx],
                annotation_text=f"{method_names[method]}: {var_value:.2%}",
                annotation_position="top left"
            )
        
        # Add mean line
        mean_return = portfolio_returns.mean() * 100
        fig.add_vline(
            x=mean_return,
            line_dash="solid",
            line_color="#34495e",
            annotation_text=f"Mean: {mean_return:.2f}%",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=f"Return Distribution with VaR ({int(var_confidence*100)}% Confidence)",
            height=400,
            template="plotly_white",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            xaxis=dict(ticksuffix="%")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk decomposition
    st.markdown("### 📊 Risk Decomposition")
    
    weights_array = results['weights_array']
    cov_matrix = returns.cov() * 252
    
    if len(weights_array) > 0:
        # Calculate risk contributions
        portfolio_variance = weights_array.T @ cov_matrix.values @ weights_array
        
        if portfolio_variance > 0:
            marginal_risk = cov_matrix.values @ weights_array
            risk_contributions = weights_array * marginal_risk / portfolio_variance
            
            risk_df = pd.DataFrame({
                'Asset': list(results['weights'].keys()),
                'Weight': weights_array,
                'Risk Contribution': risk_contributions,
                '% of Total Risk': risk_contributions * 100
            }).sort_values('Risk Contribution', ascending=False)
            
            col_risk1, col_risk2 = st.columns(2)
            
            with col_risk1:
                # Risk contribution bar chart
                fig = go.Figure(data=[go.Bar(
                    x=risk_df['Asset'],
                    y=risk_df['% of Total Risk'],
                    marker_color='#e74c3c',
                    text=risk_df['% of Total Risk'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto'
                )])
                
                fig.update_layout(
                    title="Risk Contribution by Asset",
                    height=400,
                    yaxis_title="% of Total Risk",
                    template="plotly_white",
                    xaxis_tickangle=45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col_risk2:
                # Risk vs Return scatter
                asset_returns = returns.mean() * 252
                asset_vols = returns.std() * np.sqrt(252)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=asset_vols,
                    y=asset_returns,
                    mode='markers',
                    marker=dict(
                        size=weights_array * 100,
                        color=risk_contributions,
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Risk Contribution")
                    ),
                    text=asset_returns.index,
                    hovertemplate='<b>%{text}</b><br>Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>'
                ))
                
                # Add portfolio point
                fig.add_trace(go.Scatter(
                    x=[results['expected_risk']],
                    y=[results['expected_return']],
                    mode='markers',
                    marker=dict(size=20, color='#1a5fb4', symbol='star'),
                    name='Portfolio',
                    hovertemplate='<b>Portfolio</b><br>Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Risk-Return Scatter",
                    height=400,
                    xaxis_title="Annual Volatility",
                    yaxis_title="Annual Return",
                    template="plotly_white",
                    yaxis=dict(tickformat=".1%"),
                    xaxis=dict(tickformat=".1%")
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.markdown("### 🔗 Correlation Matrix")
    
    corr_matrix = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Asset Correlation Matrix",
        height=500,
        template="plotly_white",
        xaxis_title="Assets",
        yaxis_title="Assets",
        xaxis_tickangle=45
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_performance_tab(results: Dict, prices: pd.DataFrame, 
                          benchmark_returns: pd.Series, benchmark: str):
    """Display performance analysis tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #27ae60, #2ecc71); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📈 PERFORMANCE ANALYSIS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Performance Metrics & Charts</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'portfolio_returns' not in results or results['portfolio_returns'].empty:
        st.warning("No performance data available.")
        return
    
    portfolio_returns = results['portfolio_returns']
    
    # Create performance charts
    charts = QuantStatsAnalytics.create_performance_charts(portfolio_returns, benchmark_returns)
    
    for fig in charts:
        st.plotly_chart(fig, use_container_width=True)
    
    # Rolling performance metrics
    st.markdown("### 📈 Rolling Performance Metrics")
    
    window = st.slider("Rolling Window (days)", 30, 252, 63, key="rolling_perf_window")
    
    if len(portfolio_returns) > window:
        rolling_return = portfolio_returns.rolling(window).mean() * 252
        rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_return - 0.03) / (rolling_vol + 1e-10)
        
        # Benchmark rolling metrics if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_rolling_return = benchmark_returns.rolling(window).mean() * 252
            benchmark_rolling_vol = benchmark_returns.rolling(window).std() * np.sqrt(252)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Rolling Annual Return', 'Rolling Annual Volatility', 'Rolling Sharpe Ratio'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Rolling returns
        fig.add_trace(
            go.Scatter(x=rolling_return.index, y=rolling_return.values,
                      name="Portfolio Return", line=dict(color='#1a5fb4')),
            row=1, col=1
        )
        
        if benchmark_returns is not None and not benchmark_returns.empty:
            fig.add_trace(
                go.Scatter(x=benchmark_rolling_return.index, y=benchmark_rolling_return.values,
                          name=f"{benchmark} Return", line=dict(color='#26a269', dash='dash')),
                row=1, col=1
            )
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                      name="Portfolio Vol", line=dict(color='#e74c3c')),
            row=2, col=1
        )
        
        if benchmark_returns is not None and not benchmark_returns.empty:
            fig.add_trace(
                go.Scatter(x=benchmark_rolling_vol.index, y=benchmark_rolling_vol.values,
                          name=f"{benchmark} Vol", line=dict(color='#f39c12', dash='dash')),
                row=2, col=1
            )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name="Portfolio Sharpe", line=dict(color='#27ae60')),
            row=3, col=1
        )
        
        fig.update_layout(
            height=600,
            template="plotly_white",
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Return", row=1, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Volatility", row=2, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

def display_quantstats_tab(results: Dict, benchmark_returns: pd.Series, benchmark: str):
    """Display QuantStats analytics tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #8e44ad, #9b59b6); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">🔍 QUANTSTATS ANALYTICS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Advanced Performance Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not QUANTSTATS_AVAILABLE:
        st.warning("""
        ⚠️ QuantStats package is not installed. 
        
        **To install QuantStats, run:**
        ```bash
        pip install quantstats
        ```
        
        The application is using built-in analytics instead.
        """)
    
    if 'portfolio_returns' not in results or results['portfolio_returns'].empty:
        st.warning("No performance data available.")
        return
    
    portfolio_returns = results['portfolio_returns']
    
    # Display advanced metrics
    st.markdown("### 📊 Advanced Performance Metrics")
    
    if 'quantstats_metrics' in results and results['quantstats_metrics']:
        metrics = results['quantstats_metrics']
        
        # Create a grid of metrics
        metric_cols = st.columns(4)
        
        advanced_metrics = [
            ("Sharpe Ratio", metrics.get('sharpe', 'N/A'), "Risk-adjusted return"),
            ("Sortino Ratio", metrics.get('sortino', 'N/A'), "Downside risk-adjusted return"),
            ("Calmar Ratio", metrics.get('calmar', 'N/A'), "Return vs max drawdown"),
            ("Win Rate", f"{metrics.get('win_rate', 0):.1%}" if metrics.get('win_rate') else 'N/A', "Percentage of positive periods"),
            ("CAGR", f"{metrics.get('cagr', 0):.2%}" if metrics.get('cagr') else 'N/A', "Compound annual growth rate"),
            ("Volatility", f"{metrics.get('annual_volatility', 0):.2%}" if metrics.get('annual_volatility') else 'N/A', "Annualized volatility"),
            ("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}" if metrics.get('max_drawdown') else 'N/A', "Maximum peak-to-trough decline"),
            ("VaR (95%)", f"{metrics.get('var_95_historical', 0):.2%}" if metrics.get('var_95_historical') else 'N/A', "Value at Risk"),
        ]
        
        for i, (name, value, desc) in enumerate(advanced_metrics):
            with metric_cols[i % 4]:
                st.metric(name, value, help=desc)
        
        # Performance comparison gauges
        st.markdown("### 📈 Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'win_rate' in metrics:
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number",
                    value=metrics['win_rate'] * 100,
                    title={'text': "Win Rate"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ecc71"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                )])
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'sharpe' in metrics:
                sharpe = float(metrics['sharpe']) if metrics['sharpe'] != 'N/A' else 0
                
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number",
                    value=sharpe,
                    title={'text': "Sharpe Ratio"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, max(3, sharpe * 1.5)]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 1], 'color': "lightgray"},
                            {'range': [1, 2], 'color': "lightgreen"},
                            {'range': [2, 3], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 1.0
                        }
                    }
                )])
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Benchmark metrics if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            st.markdown("### 📊 Benchmark Metrics")
            
            benchmark_cols = st.columns(4)
            
            with benchmark_cols[0]:
                if 'benchmark_metrics' in results and 'alpha' in results['benchmark_metrics']:
                    alpha = results['benchmark_metrics']['alpha']
                    st.metric("Alpha", f"{alpha:.2%}", help="Excess return over benchmark")
            
            with benchmark_cols[1]:
                if 'benchmark_metrics' in results and 'beta' in results['benchmark_metrics']:
                    beta = results['benchmark_metrics']['beta']
                    st.metric("Beta", f"{beta:.2f}", help="Market sensitivity")
            
            with benchmark_cols[2]:
                if 'benchmark_metrics' in results and 'information_ratio' in results['benchmark_metrics']:
                    ir = results['benchmark_metrics']['information_ratio']
                    st.metric("Info Ratio", f"{ir:.2f}", help="Active return vs tracking error")
            
            with benchmark_cols[3]:
                if 'benchmark_metrics' in results and 'tracking_error' in results['benchmark_metrics']:
                    te = results['benchmark_metrics']['tracking_error']
                    st.metric("Tracking Error", f"{te:.2%}", help="Volatility of excess returns")

def display_reports_tab(results: Dict, prices: pd.DataFrame, 
                       benchmark_returns: pd.Series, benchmark: str):
    """Display reports tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #34495e, #2c3e50); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📋 INSTITUTIONAL REPORTS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Professional Reports & Downloads</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Report generation section
    st.markdown("### 📊 Generate Reports")
    
    report_cols = st.columns(2)
    
    with report_cols[0]:
        if st.button("📈 Generate Performance Report", use_container_width=True):
            generate_performance_report(results, prices, benchmark_returns, benchmark)
    
    with report_cols[1]:
        if st.button("⚖️ Generate Risk Report", use_container_width=True):
            generate_risk_report(results)
    
    # Download options
    st.markdown("### 📥 Download Data")
    
    if results and 'portfolio_returns' in results:
        # Portfolio returns data
        returns_df = results['portfolio_returns'].to_frame(name='Portfolio_Returns')
        
        # Remove timezone info for Excel compatibility
        if returns_df.index.tz is not None:
            returns_df.index = returns_df.index.tz_localize(None)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv = returns_df.to_csv()
            st.download_button(
                label="📊 Download Portfolio Returns (CSV)",
                data=csv,
                file_name=f"portfolio_returns_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel download with proper datetime handling
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Portfolio returns (without timezone)
                returns_df.to_excel(writer, sheet_name='Portfolio Returns')
                
                # Weights
                if 'weights' in results:
                    weights_df = pd.DataFrame({
                        'Asset': list(results['weights'].keys()),
                        'Weight': list(results['weights'].values())
                    })
                    weights_df.to_excel(writer, sheet_name='Allocation', index=False)
                
                # Risk metrics
                if 'quantstats_metrics' in results:
                    risk_df = pd.DataFrame.from_dict(results['quantstats_metrics'], orient='index', columns=['Value'])
                    risk_df.to_excel(writer, sheet_name='Risk Metrics')
                
                # Benchmark comparison
                if benchmark_returns is not None and not benchmark_returns.empty:
                    benchmark_df = benchmark_returns.to_frame(name=f'{benchmark}_Returns')
                    if benchmark_df.index.tz is not None:
                        benchmark_df.index = benchmark_df.index.tz_localize(None)
                    benchmark_df.to_excel(writer, sheet_name='Benchmark Returns')
            
            st.download_button(
                label="📊 Download Full Report (Excel)",
                data=buffer.getvalue(),
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Summary statistics
    st.markdown("### 📊 Portfolio Summary")
    
    if results and 'quantstats_metrics' in results:
        metrics = results['quantstats_metrics']
        
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            st.markdown("#### Returns")
            
            return_metrics = [
                ("Annual Return", f"{results.get('expected_return', 0):.2%}"),
                ("CAGR", f"{metrics.get('cagr', 0):.2%}" if metrics.get('cagr') else 'N/A'),
                ("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}"),
                ("Alpha", f"{results.get('benchmark_metrics', {}).get('alpha', 0):.2%}" if 'benchmark_metrics' in results else 'N/A'),
            ]
            
            for name, value in return_metrics:
                st.markdown(f"**{name}:** {value}")
        
        with summary_cols[1]:
            st.markdown("#### Risk")
            
            risk_metrics = [
                ("Annual Volatility", f"{results.get('expected_risk', 0):.2%}"),
                ("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}" if metrics.get('max_drawdown') else 'N/A'),
                ("VaR (95%)", f"{metrics.get('var_95_historical', 0):.2%}" if metrics.get('var_95_historical') else 'N/A'),
                ("CVaR (95%)", f"{metrics.get('cvar_95', 0):.2%}" if metrics.get('cvar_95') else 'N/A'),
            ]
            
            for name, value in risk_metrics:
                st.markdown(f"**{name}:** {value}")
        
        with summary_cols[2]:
            st.markdown("#### Ratios")
            
            ratio_metrics = [
                ("Sortino Ratio", f"{metrics.get('sortino', 0):.2f}" if metrics.get('sortino') else 'N/A'),
                ("Calmar Ratio", f"{metrics.get('calmar', 0):.2f}" if metrics.get('calmar') else 'N/A'),
                ("Win Rate", f"{metrics.get('win_rate', 0):.1%}" if metrics.get('win_rate') else 'N/A'),
                ("Profit Factor", f"{metrics.get('profit_factor', 'N/A')}"),
            ]
            
            for name, value in ratio_metrics:
                st.markdown(f"**{name}:** {value}")
    
    # Allocation details
    st.markdown("### 🎯 Portfolio Allocation Details")
    
    if results and 'weights' in results:
        weights = results['weights']
        weights_df = pd.DataFrame({
            'Asset': list(weights.keys()),
            'Weight': list(weights.values()),
            'Category': [get_asset_category(asset) for asset in weights.keys()]
        }).sort_values('Weight', ascending=False)
        
        # Format weights as percentages
        weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(
            weights_df,
            use_container_width=True,
            hide_index=True
        )

def generate_performance_report(results: Dict, prices: pd.DataFrame, 
                              benchmark_returns: pd.Series, benchmark: str):
    """Generate comprehensive performance report"""
    
    with st.spinner("Generating performance report..."):
        try:
            # Create a comprehensive report
            report_content = f"""
            # PORTFOLIO PERFORMANCE REPORT
            ## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ### Executive Summary
            - **Strategy**: {st.session_state.get('portfolio_data', {}).get('strategy', 'N/A')}
            - **Benchmark**: {benchmark}
            - **Analysis Period**: {st.session_state.get('portfolio_data', {}).get('start_date', 'N/A')} to {st.session_state.get('portfolio_data', {}).get('end_date', 'N/A')}
            - **Number of Assets**: {len(results.get('weights', {}))}
            
            ### Key Performance Indicators
            - **Expected Annual Return**: {results.get('expected_return', 0):.2%}
            - **Annual Volatility**: {results.get('expected_risk', 0):.2%}
            - **Sharpe Ratio**: {results.get('sharpe_ratio', 0):.2f}
            - **Maximum Drawdown**: {results.get('quantstats_metrics', {}).get('max_drawdown', 0):.2%}
            
            ### Benchmark Comparison
            """
            
            if benchmark_returns is not None and not benchmark_returns.empty:
                benchmark_return = benchmark_returns.mean() * 252
                report_content += f"- **Benchmark Return ({benchmark})**: {benchmark_return:.2%}\n"
            
            if 'benchmark_metrics' in results:
                bm_metrics = results['benchmark_metrics']
                report_content += f"- **Alpha**: {bm_metrics.get('alpha', 0):.2%}\n"
                report_content += f"- **Beta**: {bm_metrics.get('beta', 0):.2f}\n"
                report_content += f"- **Information Ratio**: {bm_metrics.get('information_ratio', 0):.2f}\n"
            
            report_content += "\n### Portfolio Allocation\n"
            
            if 'weights' in results:
                weights = results['weights']
                for asset, weight in weights.items():
                    report_content += f"- **{asset}**: {weight:.2%}\n"
            
            # Create a downloadable report
            st.download_button(
                label="📥 Download Performance Report (TXT)",
                data=report_content,
                file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            st.success("Performance report generated successfully!")
            
        except Exception as e:
            st.error(f"Failed to generate report: {str(e)}")

def generate_risk_report(results: Dict):
    """Generate comprehensive risk report"""
    
    with st.spinner("Generating risk report..."):
        try:
            # Create risk report
            report_content = f"""
            # PORTFOLIO RISK REPORT
            ## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ### Risk Metrics Summary
            
            #### Volatility Metrics
            - **Annual Volatility**: {results.get('expected_risk', 0):.2%}
            - **Daily Volatility**: {results.get('expected_risk', 0) / np.sqrt(252):.2%}
            
            #### Downside Risk
            - **Maximum Drawdown**: {results.get('quantstats_metrics', {}).get('max_drawdown', 0):.2%}
            - **Value at Risk (95%)**: {results.get('quantstats_metrics', {}).get('var_95_historical', 0):.2%}
            - **Conditional VaR (95%)**: {results.get('quantstats_metrics', {}).get('cvar_95', 0):.2%}
            
            #### VaR Analysis (95% Confidence)
            - **Historical VaR**: {results.get('quantstats_metrics', {}).get('var_95_historical', 0):.2%}
            - **Parametric VaR**: {results.get('quantstats_metrics', {}).get('var_95_parametric', 0):.2%}
            - **Cornish-Fisher VaR**: {results.get('quantstats_metrics', {}).get('var_95_cornish_fisher', 0):.2%}
            
            #### Diversification Metrics
            - **Diversification Ratio**: {results.get('diversification_ratio', 0):.2f}
            - **Average Correlation**: {results.get('avg_correlation', 0):.2f}
            
            #### Statistical Metrics
            - **Skewness**: {results.get('quantstats_metrics', {}).get('skew', 0):.3f}
            - **Kurtosis**: {results.get('quantstats_metrics', {}).get('kurtosis', 0):.3f}
            
            ### Risk Management Recommendations
            1. Monitor correlation levels between assets
            2. Review position sizing based on volatility
            3. Implement stop-loss strategies based on VaR
            4. Regularly rebalance to maintain target allocations
            5. Consider stress testing with different market scenarios
            """
            
            st.download_button(
                label="📥 Download Risk Report (TXT)",
                data=report_content,
                file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            st.success("Risk report generated successfully!")
            
        except Exception as e:
            st.error(f"Failed to generate risk report: {str(e)}")

def get_asset_category(ticker: str) -> str:
    """Get asset category for a ticker"""
    
    for category, data in GLOBAL_ASSET_UNIVERSE.items():
        if ticker in data["assets"]:
            return category
    return "Unknown"

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    missing_deps = []
    
    if not SKLEARN_AVAILABLE:
        missing_deps.append("scikit-learn")
    
    if not PYPFOPT_AVAILABLE:
        missing_deps.append("PyPortfolioOpt")
    
    if not QUANTSTATS_AVAILABLE:
        missing_deps.append("QuantStats")
    
    if missing_deps:
        st.warning(f"""
        ⚠️ **Missing Dependencies Detected**
        
        The following packages are not installed:
        - {', '.join(missing_deps)}
        
        **To install all dependencies, run:**
        ```bash
        pip install scikit-learn pypfopt quantstats
        ```
        
        The application will run with built-in optimization algorithms.
        """)
    else:
        st.success("✅ All dependencies are installed!")

def display_welcome_screen():
    """Display welcome screen when no analysis is running"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 4rem; margin-bottom: 2rem;">🏛️</div>
            <h1 style="color: var(--primary); margin-bottom: 1rem;">
                APOLLO/ENIGMA v7.0
            </h1>
            <p style="color: var(--text-muted); font-size: 1.2rem; margin-bottom: 2rem;">
                Professional Portfolio Management & Quantitative Analysis Platform
            </p>
            <div style="background: linear-gradient(135deg, rgba(10, 61, 98, 0.1), rgba(38, 162, 105, 0.1)); 
                        padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
                <h3 style="color: var(--primary); margin-bottom: 1rem;">🚀 Quick Start</h3>
                <ol style="text-align: left; color: var(--text-secondary);">
                    <li>Select assets from sidebar (minimum 3)</li>
                    <li>Choose portfolio strategy</li>
                    <li>Select benchmark (default: SPY)</li>
                    <li>Configure risk parameters</li>
                    <li>Click "Run Analysis"</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Features grid
    st.markdown("### ✨ Key Features")
    
    features_cols = st.columns(3)
    
    features = [
        ("📊 Portfolio Optimization", "Advanced optimization algorithms for optimal asset allocation"),
        ("⚖️ Risk Analytics", "Comprehensive risk metrics including VaR, CVaR, and drawdown analysis"),
        ("📈 Performance Analysis", "Detailed performance attribution and benchmarking"),
        ("🔍 Quantitative Analytics", "Professional-grade analytics and metrics"),
        ("📉 Risk Modeling", "Advanced VaR calculations and risk decomposition"),
        ("📋 Institutional Reports", "Professional reporting and data export capabilities")
    ]
    
    for i, (title, description) in enumerate(features):
        with features_cols[i % 3]:
            st.markdown(f"""
            <div class="institutional-card" style="height: 180px;">
                <h4>{title}</h4>
                <p style="color: var(--text-muted); font-size: 0.9rem;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 1rem;">
        <p>APOLLO/ENIGMA v7.0 | Professional Portfolio Management System</p>
        <p>For institutional use only. Past performance is not indicative of future results.</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)[:200]}")
        st.info("Please refresh the page and try again.")
        
        # Display error details for debugging
        with st.expander("Error Details", expanded=False):
            st.code(traceback.format_exc())
