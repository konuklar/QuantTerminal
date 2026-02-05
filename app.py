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
import concurrent.futures
from functools import lru_cache
import traceback
import time
import hashlib
import inspect
from pathlib import Path
import pickle
import tempfile
import base64
from io import BytesIO
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Import PyPortfolioOpt
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.risk_models import CovarianceShrinkage
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

# Import Quantstats for advanced performance analytics
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False

# Import for GARCH models
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

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
# ENHANCED GLOBAL ASSET UNIVERSE
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

# -------------------------------------------------------------
# ENHANCED PORTFOLIO STRATEGIES
# -------------------------------------------------------------
PORTFOLIO_STRATEGIES = {
    "Institutional Balanced": "40% Equities, 40% Bonds, 20% Alternatives",
    "Risk Parity": "Equal risk contribution across asset classes",
    "Minimum Volatility": "Optimized for lowest portfolio volatility",
    "Maximum Sharpe": "Optimal risk-adjusted returns",
    "Equal Weight": "Equal allocation across all selected assets",
    "Hierarchical Risk Parity": "Cluster-based diversification",
    "Maximum Diversification": "Maximizes diversification ratio",
    "Black-Litterman": "Market equilibrium with investor views",
    "Mean-Variance Optimal": "Classical Markowitz optimization",
    "Efficient Risk": "Optimizes for maximum return at given risk level",
    "Efficient Return": "Optimizes for minimum risk at given return target"
}

# -------------------------------------------------------------
# ENHANCED DATA LOADER WITH ERROR HANDLING
# -------------------------------------------------------------
class EnhancedDataLoader:
    """Enhanced data loader with robust error handling"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_price_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load price data with enhanced error handling"""
        
        if not tickers:
            return pd.DataFrame()
        
        st.info(f"📊 Loading data for {len(tickers)} assets...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        prices_dict = {}
        successful_tickers = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
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
            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            status_text.text(f"Loaded {i + 1}/{len(tickers)} assets...")
        
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
            
            if len(prices_df.columns) >= 2:
                st.success(f"✅ Successfully loaded {len(prices_df.columns)} assets")
                if failed_tickers:
                    with st.expander(f"⚠️ Failed to load {len(failed_tickers)} assets", expanded=False):
                        st.write(", ".join(failed_tickers[:10]))
                return prices_df
            else:
                st.error("❌ Insufficient data for analysis")
                return pd.DataFrame()
        
        else:
            st.error("❌ Could not load any data")
            return pd.DataFrame()

# -------------------------------------------------------------
# QUANTSTATS INTEGRATION
# -------------------------------------------------------------
class QuantStatsAnalytics:
    """QuantStats integration for advanced performance analytics"""
    
    @staticmethod
    def generate_performance_report(returns: pd.Series, benchmark: pd.Series = None, 
                                   rf_rate: float = 0.03) -> Dict:
        """Generate comprehensive performance report using QuantStats"""
        
        if not QUANTSTATS_AVAILABLE or returns.empty:
            return {}
        
        try:
            # Calculate QuantStats metrics
            metrics = {}
            
            # Basic metrics
            metrics['sharpe'] = qs.stats.sharpe(returns, rf_rate)
            metrics['sortino'] = qs.stats.sortino(returns, rf_rate)
            metrics['calmar'] = qs.stats.calmar(returns)
            metrics['omega'] = qs.stats.omega(returns, rf_rate)
            metrics['cagr'] = qs.stats.cagr(returns)
            
            # Risk metrics
            metrics['volatility'] = qs.stats.volatility(returns)
            metrics['value_at_risk'] = qs.stats.value_at_risk(returns)
            metrics['conditional_var'] = qs.stats.conditional_value_at_risk(returns)
            metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
            metrics['ulcer_index'] = qs.stats.ulcer_index(returns)
            
            # Statistical metrics
            metrics['skew'] = qs.stats.skew(returns)
            metrics['kurtosis'] = qs.stats.kurtosis(returns)
            metrics['tail_ratio'] = qs.stats.tail_ratio(returns)
            
            # Benchmark-relative metrics
            if benchmark is not None and not benchmark.empty:
                aligned_returns = returns.reindex(benchmark.index).dropna()
                aligned_benchmark = benchmark.reindex(aligned_returns.index)
                
                if len(aligned_returns) > 10:
                    metrics['alpha'] = qs.stats.alpha(aligned_returns, aligned_benchmark, rf_rate)
                    metrics['beta'] = qs.stats.beta(aligned_returns, aligned_benchmark)
                    metrics['information_ratio'] = qs.stats.information_ratio(aligned_returns, aligned_benchmark)
                    metrics['r_squared'] = qs.stats.r_squared(aligned_returns, aligned_benchmark)
            
            # Win/Loss metrics
            metrics['win_rate'] = qs.stats.win_rate(returns)
            metrics['profit_factor'] = qs.stats.profit_factor(returns)
            metrics['gain_to_pain_ratio'] = qs.stats.gain_to_pain_ratio(returns)
            
            return metrics
            
        except Exception as e:
            st.warning(f"QuantStats calculation error: {str(e)[:100]}")
            return {}
    
    @staticmethod
    def create_performance_charts(returns: pd.Series, benchmark: pd.Series = None) -> List[go.Figure]:
        """Create QuantStats-style performance charts"""
        
        charts = []
        
        if returns.empty:
            return charts
        
        try:
            # 1. Cumulative Returns Chart
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
            
            if benchmark is not None:
                benchmark_cumulative = (1 + benchmark).cumprod()
                fig1.add_trace(go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    name="Benchmark",
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
            
            # 3. Monthly Returns Heatmap
            if len(returns) > 60:
                monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns_df = monthly_returns.reset_index()
                monthly_returns_df['Year'] = monthly_returns_df['index'].dt.year
                monthly_returns_df['Month'] = monthly_returns_df['index'].dt.strftime('%b')
                
                heatmap_data = monthly_returns_df.pivot(index='Year', columns='Month', values=0)
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                heatmap_data = heatmap_data.reindex(columns=months_order)
                
                fig3 = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values * 100,
                    x=heatmap_data.columns,
                    y=[str(y) for y in heatmap_data.index],
                    colorscale='RdYlGn',
                    zmid=0,
                    text=heatmap_data.values,
                    texttemplate='%{z:.1f}%',
                    textfont={"size": 10},
                    hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
                ))
                
                fig3.update_layout(
                    title="Monthly Returns Heatmap",
                    height=400,
                    template="plotly_white",
                    xaxis_title="Month",
                    yaxis_title="Year",
                    yaxis=dict(autorange="reversed")
                )
                charts.append(fig3)
            
            # 4. Return Distribution
            fig4 = go.Figure()
            fig4.add_trace(go.Histogram(
                x=returns.values * 100,
                nbinsx=50,
                name="Returns",
                marker_color='#1a5fb4',
                opacity=0.7,
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ))
            
            # Add mean line
            mean_return = returns.mean() * 100
            fig4.add_vline(
                x=mean_return,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"Mean: {mean_return:.2f}%",
                annotation_position="top right"
            )
            
            fig4.update_layout(
                title="Return Distribution",
                height=400,
                template="plotly_white",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                xaxis=dict(ticksuffix="%")
            )
            charts.append(fig4)
            
        except Exception as e:
            st.warning(f"Chart generation error: {str(e)[:100]}")
        
        return charts
    
    @staticmethod
    def generate_tearsheet_html(returns: pd.Series, benchmark: pd.Series = None) -> str:
        """Generate HTML tearsheet using QuantStats"""
        
        if not QUANTSTATS_AVAILABLE or returns.empty:
            return ""
        
        try:
            # Create temporary file for tearsheet
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                temp_file = f.name
            
            # Generate tearsheet
            if benchmark is not None:
                qs.reports.html(returns, benchmark, output=temp_file)
            else:
                qs.reports.html(returns, output=temp_file)
            
            # Read the generated HTML
            with open(temp_file, 'r') as f:
                html_content = f.read()
            
            # Clean up
            os.unlink(temp_file)
            
            return html_content
            
        except Exception as e:
            st.warning(f"Tearsheet generation error: {str(e)[:100]}")
            return ""

# -------------------------------------------------------------
# EWMA VOLATILITY ANALYSIS
# -------------------------------------------------------------
class EWMAnalysis:
    """Exponentially Weighted Moving Average volatility analysis"""
    
    @staticmethod
    def calculate_ewma_volatility(returns: pd.DataFrame, lambda_param: float = 0.94) -> pd.DataFrame:
        """Calculate EWMA volatility for multiple assets"""
        
        if returns.empty:
            return pd.DataFrame()
        
        ewma_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for asset in returns.columns:
            r = returns[asset].dropna()
            if len(r) < 20:
                ewma_vol[asset] = np.nan
                continue
            
            try:
                # Initialize with simple variance
                init_window = min(30, len(r))
                if init_window < 10:
                    ewma_vol[asset] = r.rolling(20).std() * np.sqrt(252)
                    continue
                
                init_var = r.iloc[:init_window].var()
                
                if pd.isna(init_var) or init_var <= 0:
                    ewma_vol[asset] = r.rolling(20).std() * np.sqrt(252)
                    continue
                
                # Calculate squared returns
                r_squared = r ** 2
                
                # Recursive EWMA calculation
                ewma_var = pd.Series(index=r.index, dtype=float)
                ewma_var.iloc[:init_window] = init_var
                
                for i in range(init_window, len(r)):
                    prev_var = ewma_var.iloc[i-1]
                    ewma_var.iloc[i] = lambda_param * prev_var + (1 - lambda_param) * r_squared.iloc[i-1]
                
                # Calculate annualized volatility
                ewma_vol[asset] = np.sqrt(ewma_var) * np.sqrt(252)
                
            except Exception:
                ewma_vol[asset] = r.rolling(20).std() * np.sqrt(252)
        
        return ewma_vol
    
    @staticmethod
    def create_ewma_charts(returns: pd.DataFrame, lambda_params: List[float] = None) -> go.Figure:
        """Create EWMA volatility comparison charts"""
        
        if returns.empty or len(returns.columns) == 0:
            return go.Figure()
        
        if lambda_params is None:
            lambda_params = [0.94, 0.97, 0.99]
        
        # Select first asset for visualization
        asset = returns.columns[0]
        asset_returns = returns[asset].dropna()
        
        fig = go.Figure()
        
        # Add different lambda EWMA volatilities
        colors = ['#1a5fb4', '#26a269', '#f39c12', '#e74c3c']
        
        for idx, lambda_param in enumerate(lambda_params[:4]):
            ewma_vol = EWMAnalysis.calculate_ewma_volatility(pd.DataFrame({asset: asset_returns}), lambda_param)
            
            fig.add_trace(go.Scatter(
                x=ewma_vol.index,
                y=ewma_vol[asset],
                name=f'EWMA λ={lambda_param}',
                line=dict(color=colors[idx], width=2),
                hovertemplate='Date: %{x}<br>Volatility: %{y:.2%}<extra></extra>'
            ))
        
        # Add simple rolling volatility for comparison
        rolling_vol = asset_returns.rolling(20).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            name='20-Day Rolling',
            line=dict(color='#8e44ad', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Volatility: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"EWMA Volatility Comparison - {asset}",
            height=400,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility",
            hovermode='x unified',
            yaxis=dict(tickformat=".0%"),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig

# -------------------------------------------------------------
# GARCH VOLATILITY ANALYSIS
# -------------------------------------------------------------
class GARCHAnalysis:
    """GARCH model volatility analysis"""
    
    @staticmethod
    def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1) -> Dict:
        """Fit GARCH(p,q) model to returns"""
        
        if not ARCH_AVAILABLE or returns.empty or len(returns) < 100:
            return {}
        
        try:
            # Fit GARCH model
            model = arch_model(returns * 100, vol='Garch', p=p, q=q, rescale=False)
            result = model.fit(disp='off')
            
            # Extract model parameters
            params = {
                'omega': result.params['omega'],
                'alpha': result.params.get('alpha[1]', 0),
                'beta': result.params.get('beta[1]', 0),
                'log_likelihood': result.loglikelihood,
                'aic': result.aic,
                'bic': result.bic,
                'residuals': result.resid / 100,  # Convert back from percentage
                'conditional_volatility': result.conditional_volatility / 100  # Convert back
            }
            
            # Forecast volatility
            forecast = result.forecast(horizon=5)
            params['volatility_forecast'] = np.sqrt(forecast.variance.values[-1, :]) / 100
            
            return params
            
        except Exception as e:
            st.warning(f"GARCH model fitting error: {str(e)[:100]}")
            return {}
    
    @staticmethod
    def create_garch_charts(returns: pd.Series, garch_params: Dict) -> go.Figure:
        """Create GARCH model diagnostic charts"""
        
        if not garch_params:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Conditional Volatility',
                'Standardized Residuals',
                'Residual Distribution',
                'Volatility Forecast'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Conditional Volatility
        if 'conditional_volatility' in garch_params:
            conditional_vol = garch_params['conditional_volatility'] * np.sqrt(252) * 100  # Annualized percentage
            
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=conditional_vol,
                    name='Conditional Vol',
                    line=dict(color='#1a5fb4', width=2)
                ),
                row=1, col=1
            )
            
            fig.update_yaxes(title_text="Volatility (%)", row=1, col=1, ticksuffix="%")
        
        # 2. Standardized Residuals
        if 'residuals' in garch_params and 'conditional_volatility' in garch_params:
            residuals = garch_params['residuals']
            conditional_vol = garch_params['conditional_volatility']
            standardized_residuals = residuals / (conditional_vol + 1e-10)
            
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=standardized_residuals,
                    name='Std Residuals',
                    mode='markers',
                    marker=dict(size=3, color='#26a269'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Residual Distribution
        if 'residuals' in garch_params:
            residuals = garch_params['residuals']
            
            fig.add_trace(
                go.Histogram(
                    x=residuals * 100,
                    nbinsx=50,
                    name='Residuals',
                    marker_color='#f39c12',
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Residual (%)", row=2, col=1, ticksuffix="%")
        
        # 4. Volatility Forecast
        if 'volatility_forecast' in garch_params:
            forecast = garch_params['volatility_forecast'] * np.sqrt(252) * 100  # Annualized percentage
            forecast_days = np.arange(1, len(forecast) + 1)
            
            fig.add_trace(
                go.Bar(
                    x=forecast_days,
                    y=forecast,
                    name='Vol Forecast',
                    marker_color='#e74c3c',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Days Ahead", row=2, col=2)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=2, ticksuffix="%")
        
        fig.update_layout(
            height=600,
            template="plotly_white",
            showlegend=True,
            title_text="GARCH Model Diagnostics"
        )
        
        return fig

# -------------------------------------------------------------
# ENHANCED PORTFOLIO OPTIMIZER
# -------------------------------------------------------------
class PortfolioOptimizer:
    """Enhanced portfolio optimizer with PyPortfolioOpt"""
    
    @staticmethod
    def optimize_portfolio(returns: pd.DataFrame, strategy: str, 
                          risk_free_rate: float = 0.03, 
                          constraints: Dict = None) -> Dict:
        """Optimize portfolio using specified strategy"""
        
        if returns.empty or len(returns.columns) < 2:
            return PortfolioOptimizer._equal_weight_fallback(returns)
        
        if not PYPFOPT_AVAILABLE:
            return PortfolioOptimizer._equal_weight_fallback(returns)
        
        # Default constraints
        if constraints is None:
            constraints = {
                'max_weight': 0.20,
                'min_weight': 0.01,
                'short_selling': False
            }
        
        try:
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(returns)
            S = CovarianceShrinkage(returns).ledoit_wolf()
            
            # Create efficient frontier
            weight_bounds = (0, 1) if not constraints.get('short_selling', False) else (-1, 1)
            ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
            
            # Apply weight constraints
            if constraints.get('max_weight') is not None:
                ef.add_constraint(lambda w: w <= constraints['max_weight'])
            if constraints.get('min_weight') is not None:
                ef.add_constraint(lambda w: w >= constraints['min_weight'])
            
            # Strategy-specific optimization
            weights = None
            
            if strategy == "Minimum Volatility":
                weights = ef.min_volatility()
            elif strategy == "Maximum Sharpe":
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate/252)
            elif strategy == "Risk Parity":
                # Simple risk parity implementation
                volatilities = returns.std() * np.sqrt(252)
                inv_vol = 1 / (volatilities + 1e-10)
                weights_raw = inv_vol / inv_vol.sum()
                weights = {asset: weights_raw[asset] for asset in returns.columns}
            elif strategy == "Hierarchical Risk Parity":
                hrp = HRPOpt(returns)
                weights = hrp.optimize()
            elif strategy == "Maximum Diversification":
                # Max diversification approximation
                volatilities = returns.std() * np.sqrt(252)
                weights_raw = 1 / (volatilities + 1e-10)
                weights_raw = weights_raw / weights_raw.sum()
                weights = {asset: weights_raw[asset] for asset in returns.columns}
            else:  # Equal Weight or fallback
                n_assets = len(returns.columns)
                weights = {asset: 1.0/n_assets for asset in returns.columns}
            
            # Clean weights
            if isinstance(weights, dict):
                cleaned_weights = ef.clean_weights() if hasattr(ef, 'clean_weights') else weights
            else:
                cleaned_weights = ef.clean_weights()
            
            # Calculate performance
            if hasattr(ef, 'portfolio_performance'):
                expected_return, expected_risk, sharpe_ratio = ef.portfolio_performance(
                    risk_free_rate=risk_free_rate/252
                )
            else:
                # Fallback calculation
                weights_array = np.array([cleaned_weights.get(asset, 0) for asset in returns.columns])
                portfolio_returns = (returns * weights_array).sum(axis=1)
                expected_return = portfolio_returns.mean() * 252
                expected_risk = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = (expected_return - risk_free_rate) / (expected_risk + 1e-10)
            
            # Calculate portfolio returns
            weights_array = np.array([cleaned_weights.get(asset, 0) for asset in returns.columns])
            portfolio_returns = (returns * weights_array).sum(axis=1)
            
            # Calculate additional metrics using QuantStats
            quantstats_metrics = QuantStatsAnalytics.generate_performance_report(portfolio_returns)
            
            # Calculate diversification metrics
            corr_matrix = returns.corr()
            portfolio_variance = weights_array.T @ S.values @ weights_array
            weighted_vol = np.sqrt(np.diag(S.values)) @ weights_array
            diversification_ratio = weighted_vol / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1
            
            return {
                'weights': cleaned_weights,
                'weights_array': weights_array,
                'expected_return': expected_return,
                'expected_risk': expected_risk,
                'sharpe_ratio': sharpe_ratio,
                'portfolio_returns': portfolio_returns,
                'quantstats_metrics': quantstats_metrics,
                'diversification_ratio': diversification_ratio,
                'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                'success': True
            }
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)[:200]}")
            return PortfolioOptimizer._equal_weight_fallback(returns)
    
    @staticmethod
    def _equal_weight_fallback(returns: pd.DataFrame) -> Dict:
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
                'success': False
            }
        
        n_assets = len(returns.columns)
        equal_weights = {asset: 1.0/n_assets for asset in returns.columns}
        weights_array = np.ones(n_assets) / n_assets
        
        portfolio_returns = (returns * weights_array).sum(axis=1)
        expected_return = portfolio_returns.mean() * 252
        expected_risk = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (expected_return - 0.03) / (expected_risk + 1e-10)
        
        quantstats_metrics = QuantStatsAnalytics.generate_performance_report(portfolio_returns)
        
        return {
            'weights': equal_weights,
            'weights_array': weights_array,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_returns': portfolio_returns,
            'quantstats_metrics': quantstats_metrics,
            'diversification_ratio': 0.5,
            'avg_correlation': 0.5,
            'success': False
        }

# -------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------
def main():
    """Main application with enhanced features"""
    
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
            
            # EWMA lambda
            ewma_lambda = st.slider("EWMA Lambda", 0.80, 0.99, 0.94, 0.01)
        
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
            st.metric("PyPortfolioOpt", "✅" if PYPFOPT_AVAILABLE else "❌")
        with col3:
            st.metric("QuantStats", "✅" if QUANTSTATS_AVAILABLE else "❌")
    
    # Main content
    if run_analysis and len(selected_assets) >= 3:
        with st.spinner("🔍 Conducting quantitative analysis..."):
            try:
                # Set date range
                end_date = pd.Timestamp.today()
                start_date = end_date - pd.DateOffset(years=years)
                
                # Load data
                data_loader = EnhancedDataLoader()
                prices = data_loader.load_price_data(selected_assets, start_date, end_date)
                
                if prices.empty or len(prices) < 60:
                    st.error("❌ Insufficient data for analysis.")
                    return
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                # Run portfolio optimization
                optimizer = PortfolioOptimizer()
                results = optimizer.optimize_portfolio(returns, strategy, rf_rate, constraints)
                
                if results['success']:
                    # Store results
                    st.session_state.analysis_results = results
                    st.session_state.portfolio_data = {
                        'prices': prices,
                        'returns': returns,
                        'portfolio_returns': results['portfolio_returns'],
                        'start_date': start_date,
                        'end_date': end_date,
                        'strategy': strategy,
                        'rf_rate': rf_rate
                    }
                    
                    st.success("✅ Analysis completed successfully!")
                    
                    # Create enhanced tabs
                    tab_names = [
                        "📊 Overview", 
                        "⚖️ Risk Analytics", 
                        "📈 Performance", 
                        "🔍 QuantStats", 
                        "📉 EWMA/GARCH", 
                        "📋 Reports"
                    ]
                    
                    tabs = st.tabs(tab_names)
                    
                    with tabs[0]:
                        display_overview_tab(results, prices, returns)
                    
                    with tabs[1]:
                        display_risk_tab(results, returns)
                    
                    with tabs[2]:
                        display_performance_tab(results, prices)
                    
                    with tabs[3]:
                        display_quantstats_tab(results)
                    
                    with tabs[4]:
                        display_volatility_tab(returns, ewma_lambda)
                    
                    with tabs[5]:
                        display_reports_tab(results, prices)
                
                else:
                    st.error("❌ Portfolio optimization failed.")
                    
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)[:200]}")
    
    else:
        display_welcome_screen()

def display_overview_tab(results: Dict, prices: pd.DataFrame, returns: pd.DataFrame):
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
            ("Volatility", f"{metrics.get('volatility', 0):.2%}" if metrics.get('volatility') else 'N/A', "Annualized volatility"),
            ("Value at Risk", f"{metrics.get('value_at_risk', 0):.2%}" if metrics.get('value_at_risk') else 'N/A', "95% confidence worst loss"),
            ("Conditional VaR", f"{metrics.get('conditional_var', 0):.2%}" if metrics.get('conditional_var') else 'N/A', "Expected shortfall"),
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

def display_risk_tab(results: Dict, returns: pd.DataFrame):
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
        if 'quantstats_metrics' in results and 'value_at_risk' in results['quantstats_metrics']:
            var = results['quantstats_metrics']['value_at_risk']
            st.metric("VaR (95%)", f"{var:.2%}")
        else:
            st.metric("VaR (95%)", "N/A")
    
    with col3:
        if 'quantstats_metrics' in results and 'conditional_var' in results['quantstats_metrics']:
            cvar = results['quantstats_metrics']['conditional_var']
            st.metric("CVaR (95%)", f"{cvar:.2%}")
        else:
            st.metric("CVaR (95%)", "N/A")
    
    with col4:
        if 'quantstats_metrics' in results and 'ulcer_index' in results['quantstats_metrics']:
            ulcer = results['quantstats_metrics']['ulcer_index']
            st.metric("Ulcer Index", f"{ulcer:.3f}")
        else:
            st.metric("Ulcer Index", "N/A")
    
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

def display_performance_tab(results: Dict, prices: pd.DataFrame):
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
    
    # Create performance charts using QuantStats
    charts = QuantStatsAnalytics.create_performance_charts(portfolio_returns)
    
    for fig in charts:
        st.plotly_chart(fig, use_container_width=True)
    
    # Rolling performance metrics
    st.markdown("### 📈 Rolling Performance Metrics")
    
    window = st.slider("Rolling Window (days)", 30, 252, 63, key="rolling_perf_window")
    
    if len(portfolio_returns) > window:
        rolling_return = portfolio_returns.rolling(window).mean() * 252
        rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_return - 0.03) / (rolling_vol + 1e-10)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Rolling Annual Return', 'Rolling Annual Volatility', 'Rolling Sharpe Ratio'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_return.index, y=rolling_return.values,
                      name="Return", line=dict(color='#1a5fb4')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                      name="Volatility", line=dict(color='#e74c3c')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name="Sharpe", line=dict(color='#27ae60')),
            row=3, col=1
        )
        
        fig.update_layout(
            height=600,
            template="plotly_white",
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Return", row=1, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Volatility", row=2, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

def display_quantstats_tab(results: Dict):
    """Display QuantStats analytics tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #8e44ad, #9b59b6); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">🔍 QUANTSTATS ANALYTICS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Advanced Performance Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not QUANTSTATS_AVAILABLE:
        st.warning("QuantStats package is not installed. Install with: pip install quantstats")
        return
    
    if 'portfolio_returns' not in results or results['portfolio_returns'].empty:
        st.warning("No performance data available.")
        return
    
    portfolio_returns = results['portfolio_returns']
    
    # Generate QuantStats tearsheet
    st.markdown("### 📊 QuantStats Tearsheet")
    
    if st.button("Generate QuantStats Tearsheet", key="gen_tearsheet"):
        with st.spinner("Generating QuantStats tearsheet..."):
            tearsheet_html = QuantStatsAnalytics.generate_tearsheet_html(portfolio_returns)
            
            if tearsheet_html:
                # Display the tearsheet
                st.components.v1.html(tearsheet_html, height=800, scrolling=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Tearsheet (HTML)",
                    data=tearsheet_html,
                    file_name=f"quantstats_tearsheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            else:
                st.error("Failed to generate tearsheet.")
    
    # Display advanced metrics
        # Display advanced metrics
    st.markdown("### 📊 Advanced Performance Metrics")
    
    if 'quantstats_metrics' in results and results['quantstats_metrics']:
        metrics = results['quantstats_metrics']
        
        # Create a grid of metrics
        metric_cols = st.columns(4)
        
        advanced_metrics = [
            ("Omega Ratio", metrics.get('omega', 'N/A'), "Probability-weighted return"),
            ("Tail Ratio", metrics.get('tail_ratio', 'N/A'), "Right vs left tail ratio"),
            ("Skewness", f"{metrics.get('skew', 0):.3f}" if metrics.get('skew') else 'N/A', "Return distribution asymmetry"),
            ("Kurtosis", f"{metrics.get('kurtosis', 0):.3f}" if metrics.get('kurtosis') else 'N/A', "Fat-tailedness"),
            ("Gain to Pain", f"{metrics.get('gain_to_pain_ratio', 0):.3f}" if metrics.get('gain_to_pain_ratio') else 'N/A', "Return vs drawdown"),
            ("Information Ratio", f"{metrics.get('information_ratio', 0):.3f}" if metrics.get('information_ratio') else 'N/A', "Active return vs tracking error"),
            ("Alpha", f"{metrics.get('alpha', 0):.3f}" if metrics.get('alpha') else 'N/A', "Excess return"),
            ("Beta", f"{metrics.get('beta', 0):.3f}" if metrics.get('beta') else 'N/A', "Market sensitivity"),
        ]
        
        for i, (name, value, desc) in enumerate(advanced_metrics):
            with metric_cols[i % 4]:
                st.metric(name, value, help=desc)
        
        # Performance comparison
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
            if 'profit_factor' in metrics:
                profit_factor = float(metrics['profit_factor']) if metrics['profit_factor'] != 'N/A' else 1.0
                
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number",
                    value=profit_factor,
                    title={'text': "Profit Factor"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, max(5, profit_factor * 1.5)]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 1], 'color': "lightgray"},
                            {'range': [1, 2], 'color': "lightgreen"},
                            {'range': [2, 5], 'color': "lightblue"}
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

def display_volatility_tab(returns: pd.DataFrame, ewma_lambda: float):
    """Display volatility analysis tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #f39c12, #f1c40f); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📉 VOLATILITY ANALYTICS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">EWMA & GARCH Volatility Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    if returns.empty:
        st.warning("No returns data available.")
        return
    
    # EWMA Analysis
    st.markdown("### 📊 EWMA Volatility Analysis")
    
    # Select asset for volatility analysis
    selected_asset = st.selectbox(
        "Select Asset for Volatility Analysis",
        returns.columns,
        key="vol_asset_select"
    )
    
    if selected_asset:
        asset_returns = returns[selected_asset].dropna()
        
        # EWMA chart
        st.markdown(f"#### EWMA Volatility - {selected_asset}")
        
        ewma_fig = EWMAnalysis.create_ewma_charts(
            pd.DataFrame({selected_asset: asset_returns}), 
            lambda_params=[ewma_lambda, 0.97, 0.99]
        )
        st.plotly_chart(ewma_fig, use_container_width=True)
        
        # Volatility clustering visualization
        st.markdown("#### Volatility Clustering")
        
        fig = go.Figure()
        
        # Add returns
        fig.add_trace(go.Scatter(
            x=asset_returns.index,
            y=asset_returns.values * 100,
            name="Returns",
            line=dict(color='#1a5fb4', width=1),
            opacity=0.7,
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add EWMA volatility bands
        ewma_vol = EWMAnalysis.calculate_ewma_volatility(
            pd.DataFrame({selected_asset: asset_returns}), 
            lambda_param=ewma_lambda
        )
        
        if not ewma_vol.empty:
            vol_values = ewma_vol[selected_asset].dropna() / np.sqrt(252) * 100  # Daily percentage
            
            # Upper band (mean + 2*std)
            fig.add_trace(go.Scatter(
                x=vol_values.index,
                y=vol_values.values * 2,
                name="+2σ Band",
                line=dict(color='#e74c3c', width=1, dash='dot'),
                opacity=0.5,
                fill=None
            ))
            
            # Lower band (mean - 2*std)
            fig.add_trace(go.Scatter(
                x=vol_values.index,
                y=vol_values.values * -2,
                name="-2σ Band",
                line=dict(color='#2ecc71', width=1, dash='dot'),
                opacity=0.5,
                fill='tonexty',
                fillcolor='rgba(231, 76, 60, 0.1)'
            ))
        
        fig.update_layout(
            title="Returns with Volatility Bands",
            height=400,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Daily Return (%)",
            hovermode='x unified',
            yaxis=dict(ticksuffix="%")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # GARCH Analysis
    if ARCH_AVAILABLE and len(returns) > 100:
        st.markdown("### 📊 GARCH Model Analysis")
        
        if st.button("Fit GARCH Model", key="fit_garch"):
            with st.spinner("Fitting GARCH model..."):
                # Select asset with sufficient data
                garch_asset = selected_asset if selected_asset else returns.columns[0]
                asset_returns = returns[garch_asset].dropna()
                
                if len(asset_returns) >= 100:
                    # Fit GARCH model
                    garch_params = GARCHAnalysis.fit_garch_model(asset_returns)
                    
                    if garch_params:
                        # Display model parameters
                        st.markdown("#### GARCH Model Parameters")
                        
                        param_cols = st.columns(4)
                        with param_cols[0]:
                            st.metric("ω (Constant)", f"{garch_params.get('omega', 0):.6f}")
                        with param_cols[1]:
                            st.metric("α (ARCH)", f"{garch_params.get('alpha', 0):.3f}")
                        with param_cols[2]:
                            st.metric("β (GARCH)", f"{garch_params.get('beta', 0):.3f}")
                        with param_cols[3]:
                            st.metric("Persistence", f"{garch_params.get('alpha', 0) + garch_params.get('beta', 0):.3f}")
                        
                        # Display diagnostics chart
                        garch_fig = GARCHAnalysis.create_garch_charts(asset_returns, garch_params)
                        st.plotly_chart(garch_fig, use_container_width=True)
                        
                        # Model comparison
                        st.markdown("#### Model Comparison")
                        
                        try:
                            # Fit different GARCH models
                            models = {
                                'GARCH(1,1)': GARCHAnalysis.fit_garch_model(asset_returns, 1, 1),
                                'GARCH(1,2)': GARCHAnalysis.fit_garch_model(asset_returns, 1, 2),
                                'GARCH(2,1)': GARCHAnalysis.fit_garch_model(asset_returns, 2, 1)
                            }
                            
                            comparison_data = []
                            for name, params in models.items():
                                if params:
                                    comparison_data.append({
                                        'Model': name,
                                        'Log-Likelihood': params.get('log_likelihood', 0),
                                        'AIC': params.get('aic', 0),
                                        'BIC': params.get('bic', 0),
                                        'Persistence': params.get('alpha', 0) + params.get('beta', 0)
                                    })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(
                                    comparison_df.style.format({
                                        'Log-Likelihood': '{:.2f}',
                                        'AIC': '{:.2f}',
                                        'BIC': '{:.2f}',
                                        'Persistence': '{:.3f}'
                                    }),
                                    use_container_width=True
                                )
                                
                        except Exception as e:
                            st.warning(f"Model comparison failed: {str(e)[:100]}")
                    else:
                        st.error("Failed to fit GARCH model.")
                else:
                    st.warning("Insufficient data for GARCH model. Need at least 100 observations.")
    elif not ARCH_AVAILABLE:
        st.info("ARCH package not installed. Install with: pip install arch")
    
    # Rolling volatility comparison
    st.markdown("### 📈 Rolling Volatility Comparison")
    
    if len(returns.columns) >= 2:
        # Calculate rolling volatility for all assets
        window = st.slider("Rolling Window (days)", 20, 252, 63, key="vol_window")
        
        rolling_vols = pd.DataFrame()
        for asset in returns.columns[:5]:  # Limit to first 5 assets
            rolling_vols[asset] = returns[asset].rolling(window).std() * np.sqrt(252)
        
        fig = go.Figure()
        
        for asset in rolling_vols.columns:
            fig.add_trace(go.Scatter(
                x=rolling_vols.index,
                y=rolling_vols[asset],
                name=asset,
                mode='lines',
                hovertemplate=f'{asset}<br>Date: %{{x}}<br>Vol: %{{y:.2%}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"{window}-Day Rolling Volatility",
            height=400,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility",
            hovermode='x unified',
            yaxis=dict(tickformat=".1%")
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_reports_tab(results: Dict, prices: pd.DataFrame):
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
            generate_performance_report(results, prices)
    
    with report_cols[1]:
        if st.button("⚖️ Generate Risk Report", use_container_width=True):
            generate_risk_report(results)
    
    # Download options
    st.markdown("### 📥 Download Data")
    
    if results and 'portfolio_returns' in results:
        # Portfolio returns data
        returns_df = results['portfolio_returns'].to_frame(name='Portfolio_Returns')
        
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
            # Excel download
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                returns_df.to_excel(writer, sheet_name='Returns')
                if 'weights' in results:
                    weights_df = pd.DataFrame({
                        'Asset': list(results['weights'].keys()),
                        'Weight': list(results['weights'].values())
                    })
                    weights_df.to_excel(writer, sheet_name='Allocation', index=False)
            
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
                ("Best Month", f"{metrics.get('best_month', 0):.2%}" if metrics.get('best_month') else 'N/A'),
                ("Worst Month", f"{metrics.get('worst_month', 0):.2%}" if metrics.get('worst_month') else 'N/A'),
            ]
            
            for name, value in return_metrics:
                st.markdown(f"**{name}:** {value}")
        
        with summary_cols[1]:
            st.markdown("#### Risk")
            
            risk_metrics = [
                ("Annual Volatility", f"{results.get('expected_risk', 0):.2%}"),
                ("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}" if metrics.get('max_drawdown') else 'N/A'),
                ("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}"),
                ("Sortino Ratio", f"{metrics.get('sortino', 0):.2f}" if metrics.get('sortino') else 'N/A'),
            ]
            
            for name, value in risk_metrics:
                st.markdown(f"**{name}:** {value}")
        
        with summary_cols[2]:
            st.markdown("#### Ratios")
            
            ratio_metrics = [
                ("Calmar Ratio", f"{metrics.get('calmar', 0):.2f}" if metrics.get('calmar') else 'N/A'),
                ("Omega Ratio", f"{metrics.get('omega', 0):.2f}" if metrics.get('omega') else 'N/A'),
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

def generate_performance_report(results: Dict, prices: pd.DataFrame):
    """Generate comprehensive performance report"""
    
    with st.spinner("Generating performance report..."):
        try:
            # Create a comprehensive report
            report_content = f"""
            # PORTFOLIO PERFORMANCE REPORT
            ## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ### Executive Summary
            - **Strategy**: {st.session_state.get('portfolio_data', {}).get('strategy', 'N/A')}
            - **Analysis Period**: {st.session_state.get('portfolio_data', {}).get('start_date', 'N/A')} to {st.session_state.get('portfolio_data', {}).get('end_date', 'N/A')}
            - **Number of Assets**: {len(results.get('weights', {}))}
            
            ### Key Performance Indicators
            - **Expected Annual Return**: {results.get('expected_return', 0):.2%}
            - **Annual Volatility**: {results.get('expected_risk', 0):.2%}
            - **Sharpe Ratio**: {results.get('sharpe_ratio', 0):.2f}
            - **Maximum Drawdown**: {results.get('quantstats_metrics', {}).get('max_drawdown', 0):.2%}
            
            ### Portfolio Allocation
            """
            
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
            - **Rolling Volatility (63-day)**: Calculated
            
            #### Downside Risk
            - **Maximum Drawdown**: {results.get('quantstats_metrics', {}).get('max_drawdown', 0):.2%}
            - **Value at Risk (95%)**: {results.get('quantstats_metrics', {}).get('value_at_risk', 0):.2%}
            - **Conditional VaR (95%)**: {results.get('quantstats_metrics', {}).get('conditional_var', 0):.2%}
            - **Ulcer Index**: {results.get('quantstats_metrics', {}).get('ulcer_index', 0):.3f}
            
            #### Diversification Metrics
            - **Diversification Ratio**: {results.get('diversification_ratio', 0):.2f}
            - **Average Correlation**: {results.get('avg_correlation', 0):.2f}
            
            #### Statistical Metrics
            - **Skewness**: {results.get('quantstats_metrics', {}).get('skew', 0):.3f}
            - **Kurtosis**: {results.get('quantstats_metrics', {}).get('kurtosis', 0):.3f}
            - **Tail Ratio**: {results.get('quantstats_metrics', {}).get('tail_ratio', 0):.3f}
            
            ### Risk Management Recommendations
            1. Monitor correlation levels between assets
            2. Review position sizing based on volatility
            3. Implement stop-loss strategies based on VaR
            4. Regularly rebalance to maintain target allocations
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
        ("🔍 Quantitative Analytics", "QuantStats integration for professional-grade analytics"),
        ("📉 Volatility Modeling", "EWMA and GARCH models for volatility forecasting"),
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
    
    # Recent performance (placeholder)
    st.markdown("### 📈 Recent Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("S&P 500", "4,567.89", "+1.23%")
    
    with col2:
        st.metric("NASDAQ", "14,234.56", "+2.34%")
    
    with col3:
        st.metric("10Y Treasury", "4.12%", "-0.05%")
    
    with col4:
        st.metric("Gold", "$1,978.45", "+0.78%")
    
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
