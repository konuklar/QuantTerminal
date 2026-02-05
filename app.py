# =============================================================
# 🏛️ APOLLO/ENIGMA QUANT TERMINAL v6.0 - INSTITUTIONAL EDITION
# Professional Global Multi-Asset Portfolio Management System
# Enhanced with Institutional Reporting & Advanced Analytics
# Streamlit-compatible version without PDF dependencies
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

# Import PyPortfolioOpt with all required components
try:
    from pypfopt import expected_returns, risk_models, objective_functions, black_litterman, hierarchical_portfolio
    from pypfopt.efficient_frontier import EfficientFrontier, EfficientSemivariance
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.objective_functions import L2_reg, transaction_cost
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.risk_models import CovarianceShrinkage
    PYPFOPT_AVAILABLE = True
except ImportError as e:
    PYPFOPT_AVAILABLE = False
    st.warning(f"⚠️ PyPortfolioOpt not installed. Some optimization features limited.")

# Additional imports
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# SUPERIOR INSTITUTIONAL STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="APOLLO/ENIGMA - Institutional Portfolio Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏛️"
)

# Enhanced institutional CSS
INSTITUTIONAL_CSS = """
<style>
:root {
    --primary: #0a3d62;
    --primary-dark: #082c47;
    --primary-light: #1a5fb4;
    --secondary: #26a269;
    --secondary-dark: #1e864d;
    --accent: #f39c12;
    --accent-dark: #d68910;
    --danger: #e74c3c;
    --danger-dark: #c0392b;
    --warning: #f1c40f;
    --warning-dark: #f39c12;
    --success: #27ae60;
    --success-dark: #229954;
    --dark-bg: #0f172a;
    --dark-bg-light: #1e293b;
    --card-bg: #1e293b;
    --card-border: #334155;
    --text: #f8f9fa;
    --text-muted: #94a3b8;
    --text-light: #e2e8f0;
    --shadow: rgba(0, 0, 0, 0.3);
}

/* Enhanced Institutional Theme */
.main {
    background: linear-gradient(135deg, var(--dark-bg) 0%, #0a1929 100%);
}

/* Professional Header */
.institutional-header {
    background: linear-gradient(90deg, var(--primary-dark), var(--primary));
    padding: 1.5rem;
    border-radius: 0 0 15px 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px var(--shadow);
    border-bottom: 3px solid var(--accent);
}

/* Superior Cards */
.institutional-card {
    background: linear-gradient(145deg, var(--card-bg), #16213e);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.institutional-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
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
    background: linear-gradient(135deg, rgba(10, 61, 98, 0.1), rgba(26, 95, 180, 0.05));
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.institutional-metric:hover {
    background: linear-gradient(135deg, rgba(10, 61, 98, 0.2), rgba(26, 95, 180, 0.1));
    border-color: var(--primary-light);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    margin: 0.5rem 0;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
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
    background: var(--dark-bg-light);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--card-border);
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
    background: rgba(10, 61, 98, 0.2);
    color: var(--text-light);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    border-color: var(--accent) !important;
    box-shadow: 0 4px 15px rgba(10, 61, 98, 0.3);
}

/* Professional Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    border: none;
    padding: 12px 28px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(10, 61, 98, 0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transition: 0.5s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(10, 61, 98, 0.4);
}

.stButton > button:hover::before {
    left: 100%;
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
    background: linear-gradient(135deg, var(--primary-light), #3498db);
    color: white;
}

/* Enhanced Tables */
.institutional-table {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    overflow: hidden;
}

.institutional-table table {
    width: 100%;
    background: transparent;
}

.institutional-table th {
    background: var(--primary-dark);
    color: white;
    font-weight: 600;
    padding: 12px 16px;
    border-bottom: 2px solid var(--accent);
}

.institutional-table td {
    padding: 10px 16px;
    border-bottom: 1px solid var(--card-border);
    color: var(--text-light);
}

.institutional-table tr:hover {
    background: rgba(10, 61, 98, 0.1);
}

/* Professional Charts */
.js-plotly-plot {
    border: 1px solid var(--card-border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--card-bg);
}

/* Sidebar Enhancement */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, var(--dark-bg-light), #1a1a2e);
    border-right: 1px solid var(--card-border);
}

/* Input Enhancements */
.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
}

.stSelectbox > div > div:hover, .stTextInput > div > div:hover {
    border-color: var(--primary-light) !important;
    box-shadow: 0 0 0 2px rgba(26, 95, 180, 0.2) !important;
}

/* Progress Bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 4px;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--dark-bg);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--primary), var(--primary-dark));
    border-radius: 5px;
    border: 2px solid var(--dark-bg);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(var(--primary-dark), var(--primary));
}

/* Professional Loading */
.stSpinner > div {
    border: 4px solid var(--card-border);
    border-top: 4px solid var(--primary);
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: institutional-spin 1s linear infinite;
}

@keyframes institutional-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Grid Layout */
.institutional-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

/* Tooltips */
[data-tooltip] {
    position: relative;
    cursor: help;
}

[data-tooltip]:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--primary-dark);
    color: white;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    white-space: nowrap;
    z-index: 1000;
    border: 1px solid var(--accent);
    box-shadow: 0 4px 15px var(--shadow);
}

/* Expander Headers */
.streamlit-expanderHeader {
    background: linear-gradient(90deg, var(--card-bg), var(--dark-bg-light));
    border: 1px solid var(--card-border);
    border-radius: 8px;
    color: var(--text);
    font-weight: 600;
    padding: 1rem;
}

.streamlit-expanderHeader:hover {
    background: linear-gradient(90deg, var(--dark-bg-light), rgba(10, 61, 98, 0.2));
    border-color: var(--primary-light);
}

/* Alert Enhancements */
.stAlert {
    border-radius: 10px !important;
    border: 1px solid !important;
    background: var(--card-bg) !important;
}

/* Footer */
.institutional-footer {
    background: linear-gradient(90deg, var(--dark-bg-light), var(--primary-dark));
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
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.kpi-card {
    background: linear-gradient(145deg, var(--card-bg), rgba(10, 61, 98, 0.2));
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    border-color: var(--primary-light);
}

.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
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
    background: linear-gradient(135deg, rgba(26, 95, 180, 0.1), rgba(38, 162, 105, 0.05));
    border: 2px solid var(--primary);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Dataframe Styling */
.dataframe {
    border: 1px solid var(--card-border) !important;
    border-radius: 10px !important;
}

.dataframe th {
    background: var(--primary-dark) !important;
    color: white !important;
    font-weight: 600 !important;
}

.dataframe td {
    border-color: var(--card-border) !important;
}

/* Metric Containers */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, var(--card-bg), rgba(10, 61, 98, 0.1)) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

[data-testid="metric-container"]:hover {
    border-color: var(--primary-light) !important;
    transform: translateY(-2px);
    transition: all 0.3s ease;
}

/* Section Headers */
.section-header {
    background: linear-gradient(90deg, rgba(10, 61, 98, 0.2), transparent);
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
</style>
"""

st.markdown(INSTITUTIONAL_CSS, unsafe_allow_html=True)

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
        "assets": ["GLD", "SLV", "USO", "DBA", "PDBC", "GSG", "URA", "REMX"]
    },
    "Alternatives": {
        "description": "Cryptocurrencies, volatility, real estate",
        "assets": ["BTC-USD", "ETH-USD", "^VIX", "VNQ", "IYR", "REM", "TMF", "UPRO"]
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
        "assets": ["MTUM", "QUAL", "VLUE", "SIZE", "USMV", "LRGF", "VFMF", "AVUV"]
    },
    "Sustainability": {
        "description": "ESG and clean energy focused",
        "assets": ["ESGU", "ICLN", "TAN", "PBW", "QCLN", "PBD", "SMOG", "ERTH"]
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
    "Institutional Balanced": {
        "description": "40% Equities, 40% Bonds, 20% Alternatives - Classic institutional allocation",
        "type": "Strategic"
    },
    "Risk Parity": {
        "description": "Equal risk contribution across asset classes",
        "type": "Risk-Based"
    },
    "Minimum Volatility": {
        "description": "Optimized for lowest portfolio volatility",
        "type": "Risk-Based"
    },
    "Maximum Sharpe": {
        "description": "Optimal risk-adjusted returns (mean-variance optimization)",
        "type": "Optimization"
    },
    "Black-Litterman": {
        "description": "Market equilibrium with institutional views",
        "type": "Optimization"
    },
    "Hierarchical Risk Parity": {
        "description": "Cluster-based diversification using hierarchical clustering",
        "type": "Advanced"
    },
    "Factor-Based": {
        "description": "Multi-factor optimization incorporating value, momentum, quality",
        "type": "Advanced"
    },
    "ESG Integrated": {
        "description": "Sustainability-constrained optimization",
        "type": "Thematic"
    },
    "Tail Risk Hedge": {
        "description": "Enhanced downside protection with options overlay",
        "type": "Risk-Based"
    },
    "Global Macro": {
        "description": "Multi-asset global allocation based on macroeconomic views",
        "type": "Tactical"
    },
    "Equal Weight": {
        "description": "Equal allocation across all selected assets",
        "type": "Simple"
    }
}

# -------------------------------------------------------------
# INSTITUTIONAL REPORTING SYSTEM
# -------------------------------------------------------------
class InstitutionalReportGenerator:
    """Superior institutional reporting system with multiple formats"""
    
    @staticmethod
    def generate_executive_summary(portfolio_data: Dict, analysis_results: Dict, benchmark_data: Dict = None) -> str:
        """Generate professional executive summary"""
        
        summary = f"""
# 🏛️ EXECUTIVE SUMMARY - PORTFOLIO ANALYSIS
## APOLLO/ENIGMA INSTITUTIONAL TERMINAL v6.0
### Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 PORTFOLIO OVERVIEW

| Metric | Portfolio | Benchmark | Status |
|--------|-----------|-----------|---------|
| **Total Return** | {portfolio_data.get('total_return', 0):.2%} | {benchmark_data.get('total_return', 0) if benchmark_data else 'N/A':.2%} | {InstitutionalReportGenerator._get_status_icon(portfolio_data.get('total_return', 0) >= (benchmark_data.get('total_return', 0) if benchmark_data else 0))} |
| **Annual Return** | {portfolio_data.get('annual_return', 0):.2%} | {benchmark_data.get('annual_return', 0) if benchmark_data else 'N/A':.2%} | {InstitutionalReportGenerator._get_status_icon(portfolio_data.get('annual_return', 0) >= (benchmark_data.get('annual_return', 0) if benchmark_data else 0))} |
| **Annual Volatility** | {portfolio_data.get('annual_volatility', 0):.2%} | {benchmark_data.get('annual_volatility', 0) if benchmark_data else 'N/A':.2%} | {InstitutionalReportGenerator._get_status_icon(portfolio_data.get('annual_volatility', 0) <= (benchmark_data.get('annual_volatility', 0) if benchmark_data else float('inf')))} |
| **Sharpe Ratio** | {portfolio_data.get('sharpe_ratio', 0):.2f} | {benchmark_data.get('sharpe_ratio', 0) if benchmark_data else 'N/A':.2f} | {InstitutionalReportGenerator._get_status_icon(portfolio_data.get('sharpe_ratio', 0) >= (benchmark_data.get('sharpe_ratio', 0) if benchmark_data else 0))} |
| **Max Drawdown** | {portfolio_data.get('max_drawdown', 0):.2%} | {benchmark_data.get('max_drawdown', 0) if benchmark_data else 'N/A':.2%} | {InstitutionalReportGenerator._get_status_icon(portfolio_data.get('max_drawdown', 0) >= (benchmark_data.get('max_drawdown', 0) if benchmark_data else 0))} |
| **Sortino Ratio** | {portfolio_data.get('sortino_ratio', 'N/A')} | {benchmark_data.get('sortino_ratio', 'N/A') if benchmark_data else 'N/A'} | {InstitutionalReportGenerator._get_status_icon(portfolio_data.get('sortino_ratio', 0) >= (benchmark_data.get('sortino_ratio', 0) if benchmark_data else 0))} |

---

## 🎯 KEY FINDINGS

### 1. PERFORMANCE ASSESSMENT
{InstitutionalReportGenerator._generate_performance_assessment(portfolio_data, benchmark_data)}

### 2. RISK ANALYSIS
{InstitutionalReportGenerator._generate_risk_assessment(analysis_results)}

### 3. PORTFOLIO CONSTRUCTION
{InstitutionalReportGenerator._generate_portfolio_assessment(portfolio_data)}

---

## 📈 RECOMMENDATIONS

### 🟢 STRENGTHS
{InstitutionalReportGenerator._generate_strengths(portfolio_data)}

### 🟡 OPPORTUNITIES
{InstitutionalReportGenerator._generate_opportunities(portfolio_data)}

### 🔴 RISKS TO MONITOR
{InstitutionalReportGenerator._generate_risks(analysis_results)}

---

## 📊 DETAILED METRICS

### Risk-Adjusted Performance
{InstitutionalReportGenerator._generate_detailed_metrics(portfolio_data)}

### Portfolio Characteristics
- **Number of Assets**: {portfolio_data.get('num_assets', 'N/A')}
- **Diversification Score**: {portfolio_data.get('diversification_score', 'N/A'):.2f}
- **Average Correlation**: {portfolio_data.get('avg_correlation', 'N/A'):.3f}
- **Portfolio Beta**: {portfolio_data.get('beta', 'N/A'):.2f}
- **Tracking Error**: {portfolio_data.get('tracking_error', 'N/A'):.2%}

---

*This report was generated by Apollo/ENIGMA Institutional Terminal v6.0*
*For institutional use only. Not for distribution to retail investors.*
*Past performance does not guarantee future results.*
"""
        return summary
    
    @staticmethod
    def _get_status_icon(condition: bool) -> str:
        return "✅" if condition else "⚠️"
    
    @staticmethod
    def _generate_performance_assessment(data: Dict, benchmark: Dict = None) -> str:
        """Generate performance assessment section"""
        assessment = ""
        
        if benchmark:
            if data.get('total_return', 0) > benchmark.get('total_return', 0):
                assessment += f"- Portfolio outperformed benchmark by {(data['total_return'] - benchmark['total_return']):.2%}\n"
            else:
                assessment += f"- Portfolio underperformed benchmark by {(benchmark['total_return'] - data['total_return']):.2%}\n"
        
        sharpe = data.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            assessment += "- Exceptional risk-adjusted returns (Sharpe > 1.5)\n"
        elif sharpe > 1.0:
            assessment += "- Strong risk-adjusted returns (Sharpe > 1.0)\n"
        elif sharpe > 0.5:
            assessment += "- Acceptable risk-adjusted returns\n"
        else:
            assessment += "- Below target risk-adjusted returns\n"
        
        return assessment
    
    @staticmethod
    def _generate_risk_assessment(data: Dict) -> str:
        """Generate risk assessment section"""
        assessment = ""
        
        var_95 = data.get('value_at_risk_95', 0)
        if var_95 > -0.03:
            assessment += "- Normal market risk exposure\n"
        elif var_95 > -0.05:
            assessment += "- Moderate downside risk\n"
        else:
            assessment += "- Elevated downside risk\n"
        
        max_dd = data.get('max_drawdown', 0)
        if max_dd > -0.15:
            assessment += "- Manageable drawdown risk\n"
        elif max_dd > -0.25:
            assessment += "- Moderate drawdown risk\n"
        else:
            assessment += "- High drawdown risk\n"
        
        return assessment
    
    @staticmethod
    def _generate_strengths(data: Dict) -> str:
        """Generate strengths section"""
        strengths = ""
        
        if data.get('sharpe_ratio', 0) > 1.0:
            strengths += "- Superior risk-adjusted performance\n"
        
        if data.get('diversification_score', 0) > 0.7:
            strengths += "- Well-diversified portfolio construction\n"
        
        if data.get('sortino_ratio', 0) > 1.0:
            strengths += "- Strong downside protection\n"
        
        if data.get('win_rate', 0) > 0.55:
            strengths += "- Consistent positive returns\n"
        
        return strengths
    
    @staticmethod
    def _generate_opportunities(data: Dict) -> str:
        """Generate opportunities section"""
        opportunities = ""
        
        if data.get('sharpe_ratio', 0) < 1.0:
            opportunities += "- Improve risk-adjusted returns through optimization\n"
        
        if data.get('max_drawdown', 0) < -0.20:
            opportunities += "- Enhance downside protection strategies\n"
        
        if data.get('diversification_score', 0) < 0.6:
            opportunities += "- Increase portfolio diversification\n"
        
        return opportunities
    
    @staticmethod
    def _generate_risks(data: Dict) -> str:
        """Generate risks section"""
        risks = ""
        
        if data.get('value_at_risk_95', 0) < -0.05:
            risks += "- High potential for significant daily losses\n"
        
        if data.get('max_drawdown', 0) < -0.25:
            risks += "- Significant historical drawdowns\n"
        
        if data.get('beta', 0) > 1.2:
            risks += "- High market sensitivity\n"
        
        return risks
    
    @staticmethod
    def _generate_detailed_metrics(data: Dict) -> str:
        """Generate detailed metrics section"""
        metrics = f"""
| Metric | Value |
|--------|-------|
| **Alpha** | {data.get('alpha', 'N/A')} |
| **Beta** | {data.get('beta', 'N/A'):.2f} |
| **R-squared** | {data.get('r_squared', 'N/A'):.3f} |
| **Tracking Error** | {data.get('tracking_error', 'N/A'):.2%} |
| **Information Ratio** | {data.get('information_ratio', 'N/A'):.2f} |
| **Treynor Ratio** | {data.get('treynor_ratio', 'N/A'):.2f} |
| **Calmar Ratio** | {data.get('calmar_ratio', 'N/A'):.2f} |
| **Omega Ratio** | {data.get('omega_ratio', 'N/A'):.2f} |
| **Ulcer Index** | {data.get('ulcer_index', 'N/A'):.3f} |
| **Martin Ratio** | {data.get('martin_ratio', 'N/A'):.2f} |
"""
        return metrics
    
    @staticmethod
    def _generate_portfolio_assessment(data: Dict) -> str:
        """Generate portfolio assessment section"""
        assessment = ""
        
        if data.get('num_assets', 0) >= 8:
            assessment += "- Well-diversified across multiple assets\n"
        elif data.get('num_assets', 0) >= 5:
            assessment += "- Adequate diversification\n"
        else:
            assessment += "- Limited diversification - consider adding more assets\n"
        
        if data.get('diversification_score', 0) > 0.7:
            assessment += "- Strong diversification benefits\n"
        elif data.get('diversification_score', 0) > 0.5:
            assessment += "- Moderate diversification benefits\n"
        else:
            assessment += "- Limited diversification benefits\n"
        
        return assessment
    
    @staticmethod
    def generate_markdown_report(portfolio_data: Dict, weights: Dict, 
                                attribution: pd.DataFrame, charts: List[go.Figure] = None) -> str:
        """Generate comprehensive markdown report"""
        
        report = f"""
# 📋 COMPREHENSIVE PORTFOLIO ANALYSIS REPORT
## APOLLO/ENIGMA INSTITUTIONAL TERMINAL v6.0
### Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 EXECUTIVE SUMMARY

### Portfolio Performance
- **Total Period Return**: {portfolio_data.get('total_return', 0):.2%}
- **Annualized Return**: {portfolio_data.get('annual_return', 0):.2%}
- **Annualized Volatility**: {portfolio_data.get('annual_volatility', 0):.2%}
- **Sharpe Ratio**: {portfolio_data.get('sharpe_ratio', 0):.2f}
- **Maximum Drawdown**: {portfolio_data.get('max_drawdown', 0):.2%}

### Risk Assessment
- **Value at Risk (95%)**: {portfolio_data.get('value_at_risk_95', 0):.2%}
- **Conditional VaR (95%)**: {portfolio_data.get('conditional_var_95', 0):.2%}
- **Sortino Ratio**: {portfolio_data.get('sortino_ratio', 'N/A')}
- **Beta**: {portfolio_data.get('beta', 0):.2f}

---

## 🎯 PORTFOLIO ALLOCATION

### Asset Weights
| Asset | Weight | Category |
|-------|--------|----------|
"""
        
        # Add weights table
        for asset, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            category = next((cat for cat, data in GLOBAL_ASSET_UNIVERSE.items() 
                           if asset in data["assets"]), "Other")
            report += f"| {asset} | {weight:.2%} | {category} |\n"
        
        report += f"""
**Total Assets**: {len(weights)}
**Concentration (Top 3)**: {sum(sorted(weights.values(), reverse=True)[:3]):.1%}

---

## 📈 PERFORMANCE ATTRIBUTION

### Top Contributors
| Asset | Contribution | Weight | Return |
|-------|--------------|--------|---------|
"""
        
        # Add attribution table (top 5)
        if attribution is not None and not attribution.empty:
            top_contributors = attribution.nlargest(5, 'Contribution')
            for idx, row in top_contributors.iterrows():
                report += f"| {idx} | {row.get('Contribution', 0):.4%} | {row.get('Weight', 0):.2%} | {row.get('Return', 0):.2%} |\n"
        
        report += """
---

## ⚖️ RISK METRICS DETAILS

### Statistical Properties
- **Skewness**: {skewness:.3f}
- **Kurtosis**: {kurtosis:.3f}
- **Jarque-Bera**: {jarque_bera:.1f}
- **Win Rate**: {win_rate:.1%}
- **Profit Factor**: {profit_factor:.2f}
- **Gain/Loss Ratio**: {gain_loss_ratio:.2f}

### Risk Measures
| Confidence | VaR | CVaR |
|------------|-----|------|
| 90% | {var_90:.3%} | {cvar_90:.3%} |
| 95% | {var_95:.3%} | {cvar_95:.3%} |
| 99% | {var_99:.3%} | {cvar_99:.3%} |

---

## 📊 PORTFOLIO CHARACTERISTICS

- **Diversification Score**: {diversification_score:.2f}
- **Average Correlation**: {avg_correlation:.3f}
- **Portfolio Turnover**: {turnover:.1%}
- **Liquidity Score**: {liquidity_score:.2f}
- **ESG Score**: {esg_score:.2f}

---

## 💡 RECOMMENDATIONS & NEXT STEPS

### Immediate Actions
1. Monitor concentration risk in top holdings
2. Review asset allocation based on current market conditions
3. Consider rebalancing if drift exceeds thresholds

### Strategic Considerations
1. Evaluate diversification opportunities
2. Assess downside protection strategies
3. Review risk-adjusted performance targets

---

## 📅 REPORT METADATA

- **Analysis Period**: {start_date} to {end_date}
- **Number of Observations**: {n_obs}
- **Risk-Free Rate**: {rf_rate:.1%}
- **Confidence Level**: 95%
- **Rebalancing Frequency**: Quarterly

---

*This report is for institutional use only. Not financial advice.*
*Generated by Apollo/ENIGMA Portfolio Analysis System v6.0*
""".format(
            skewness=portfolio_data.get('skewness', 0),
            kurtosis=portfolio_data.get('kurtosis', 0),
            jarque_bera=portfolio_data.get('jarque_bera', 0),
            win_rate=portfolio_data.get('win_rate', 0),
            profit_factor=portfolio_data.get('profit_factor', 0),
            gain_loss_ratio=portfolio_data.get('gain_loss_ratio', 0),
            var_90=portfolio_data.get('value_at_risk_90', 0),
            cvar_90=portfolio_data.get('conditional_var_90', 0),
            var_95=portfolio_data.get('value_at_risk_95', 0),
            cvar_95=portfolio_data.get('conditional_var_95', 0),
            var_99=portfolio_data.get('value_at_risk_99', 0),
            cvar_99=portfolio_data.get('conditional_var_99', 0),
            diversification_score=portfolio_data.get('diversification_score', 0),
            avg_correlation=portfolio_data.get('avg_correlation', 0),
            turnover=portfolio_data.get('turnover', 0),
            liquidity_score=portfolio_data.get('liquidity_score', 0.7),
            esg_score=portfolio_data.get('esg_score', 0.6),
            start_date=portfolio_data.get('start_date', 'N/A'),
            end_date=portfolio_data.get('end_date', 'N/A'),
            n_obs=portfolio_data.get('n_obs', 0),
            rf_rate=portfolio_data.get('risk_free_rate', 0.03)
        )
        
        return report
    
    @staticmethod
    def generate_html_report(portfolio_data: Dict, weights: Dict, 
                           attribution: pd.DataFrame = None) -> str:
        """Generate HTML report for display and download"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Analysis Report - Apollo/ENIGMA v6.0</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 40px; background: #f8f9fa; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #0a3d62, #1a5fb4); 
                  color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem; }}
        .section {{ background: white; padding: 1.5rem; border-radius: 8px; 
                   margin-bottom: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                       gap: 1rem; margin: 1rem 0; }}
        .metric-card {{ background: #f8f9fa; padding: 1rem; border-radius: 6px; 
                       border-left: 4px solid #1a5fb4; }}
        .metric-value {{ font-size: 1.5rem; font-weight: bold; color: #0a3d62; }}
        .metric-label {{ font-size: 0.9rem; color: #666; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #0a3d62; color: white; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; 
                  font-size: 0.9rem; color: #666; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ Portfolio Analysis Report</h1>
            <p>Apollo/ENIGMA Institutional Terminal v6.0</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value">{portfolio_data.get('total_return', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Annual Return</div>
                    <div class="metric-value">{portfolio_data.get('annual_return', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{portfolio_data.get('sharpe_ratio', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value">{portfolio_data.get('max_drawdown', 0):.2%}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Portfolio Allocation</h2>
            <table>
                <tr>
                    <th>Asset</th>
                    <th>Weight</th>
                    <th>Category</th>
                    <th>Contribution</th>
                </tr>
"""
        
        # Add weights rows
        for asset, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            category = next((cat for cat, data in GLOBAL_ASSET_UNIVERSE.items() 
                           if asset in data["assets"]), "Other")
            contribution = weight * portfolio_data.get('annual_return', 0)
            html += f"""
                <tr>
                    <td>{asset}</td>
                    <td>{weight:.2%}</td>
                    <td>{category}</td>
                    <td class="{'positive' if contribution >= 0 else 'negative'}">{contribution:.4%}</td>
                </tr>
"""
        
        html += """
            </table>
        </div>
        
        <div class="section">
            <h2>⚖️ Risk Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Annual Volatility</div>
                    <div class="metric-value">{annual_vol:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">VaR (95%)</div>
                    <div class="metric-value">{var_95:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-value">{sortino:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Beta</div>
                    <div class="metric-value">{beta:.2f}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Performance Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{win_rate:.1%}</td>
                    <td>Percentage of positive periods</td>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td>{profit_factor:.2f}</td>
                    <td>Gross profit / gross loss</td>
                </tr>
                <tr>
                    <td>Skewness</td>
                    <td>{skewness:.3f}</td>
                    <td>Distribution asymmetry</td>
                </tr>
                <tr>
                    <td>Kurtosis</td>
                    <td>{kurtosis:.3f}</td>
                    <td>Tail risk measure</td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            <p>This report was generated by Apollo/ENIGMA Institutional Terminal v6.0</p>
            <p>For institutional use only. Not for distribution to retail investors.</p>
            <p>Past performance does not guarantee future results.</p>
        </div>
    </div>
</body>
</html>
""".format(
            annual_vol=portfolio_data.get('annual_volatility', 0),
            var_95=portfolio_data.get('value_at_risk_95', 0),
            sortino=portfolio_data.get('sortino_ratio', 0),
            beta=portfolio_data.get('beta', 0),
            win_rate=portfolio_data.get('win_rate', 0),
            profit_factor=portfolio_data.get('profit_factor', 0),
            skewness=portfolio_data.get('skewness', 0),
            kurtosis=portfolio_data.get('kurtosis', 0)
        )
        
        return html
    
    @staticmethod
    def create_report_download_buttons(portfolio_data: Dict, weights: Dict, 
                                     attribution: pd.DataFrame = None):
        """Create download buttons for reports"""
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Markdown report
            markdown_report = InstitutionalReportGenerator.generate_executive_summary(portfolio_data, {})
            st.download_button(
                label="📥 Download Executive Summary (MD)",
                data=markdown_report,
                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # HTML report
            html_report = InstitutionalReportGenerator.generate_html_report(portfolio_data, weights, attribution)
            st.download_button(
                label="📥 Download Full Report (HTML)",
                data=html_report,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col3:
            # CSV data export
            if 'returns' in portfolio_data and portfolio_data['returns'] is not None:
                csv_data = portfolio_data['returns'].to_csv()
                st.download_button(
                    label="📥 Download Returns Data (CSV)",
                    data=csv_data,
                    file_name=f"portfolio_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# -------------------------------------------------------------
# INSTITUTIONAL PERFORMANCE METRICS
# -------------------------------------------------------------
class InstitutionalMetrics:
    """Superior institutional performance metrics calculator"""
    
    @staticmethod
    def calculate_comprehensive_metrics(returns: pd.Series, benchmark: pd.Series = None, 
                                       rf_rate: float = 0.03, trading_days: int = 252) -> Dict:
        """Calculate 50+ institutional metrics"""
        metrics = {}
        
        if returns is None or len(returns) < 20:
            return metrics
        
        returns_clean = returns.dropna()
        
        # Basic metrics
        metrics['total_return'] = (1 + returns_clean).prod() - 1
        metrics['annual_return'] = returns_clean.mean() * trading_days
        metrics['annual_volatility'] = returns_clean.std() * np.sqrt(trading_days)
        
        # Risk-adjusted metrics
        if metrics['annual_volatility'] > 0:
            metrics['sharpe_ratio'] = (metrics['annual_return'] - rf_rate) / metrics['annual_volatility']
        
        # Downside metrics
        downside_returns = returns_clean[returns_clean < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(trading_days)
            if downside_vol > 0:
                metrics['sortino_ratio'] = (metrics['annual_return'] - rf_rate) / downside_vol
            
            # Omega ratio
            threshold = rf_rate / trading_days
            excess_returns = returns_clean - threshold
            positive_excess = excess_returns[excess_returns > 0].sum()
            negative_excess = abs(excess_returns[excess_returns < 0].sum())
            if negative_excess > 0:
                metrics['omega_ratio'] = positive_excess / negative_excess
        
        # Drawdown metrics
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown.mean()
        
        if metrics['max_drawdown'] < 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
            metrics['sterling_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
        
        # Statistical metrics
        metrics['skewness'] = returns_clean.skew()
        metrics['kurtosis'] = returns_clean.kurtosis()
        
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
            metrics['jarque_bera'] = jb_stat
            metrics['jarque_bera_pvalue'] = jb_pvalue
        except:
            metrics['jarque_bera'] = np.nan
        
        # Value at Risk metrics
        metrics['value_at_risk_90'] = np.percentile(returns_clean, 10)
        metrics['value_at_risk_95'] = np.percentile(returns_clean, 5)
        metrics['value_at_risk_99'] = np.percentile(returns_clean, 1)
        
        # Conditional VaR
        if not pd.isna(metrics['value_at_risk_95']):
            tail_95 = returns_clean[returns_clean <= metrics['value_at_risk_95']]
            metrics['conditional_var_95'] = tail_95.mean() if len(tail_95) > 0 else metrics['value_at_risk_95']
        
        # Benchmark-relative metrics
        if benchmark is not None:
            aligned_returns = returns_clean.copy()
            aligned_benchmark = benchmark.reindex(returns_clean.index).dropna()
            aligned_returns = aligned_returns.reindex(aligned_benchmark.index)
            
            if len(aligned_returns) > 10:
                try:
                    # Alpha and Beta
                    cov_matrix = np.cov(aligned_returns, aligned_benchmark)
                    if cov_matrix[1, 1] > 0:
                        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                        metrics['beta'] = beta
                        metrics['alpha'] = (aligned_returns.mean() - beta * aligned_benchmark.mean()) * trading_days
                    
                    # R-squared
                    correlation = np.corrcoef(aligned_returns, aligned_benchmark)[0, 1]
                    metrics['r_squared'] = correlation ** 2
                    
                    # Tracking error and information ratio
                    active_returns = aligned_returns - aligned_benchmark
                    metrics['tracking_error'] = active_returns.std() * np.sqrt(trading_days)
                    if metrics['tracking_error'] > 0:
                        metrics['information_ratio'] = (aligned_returns.mean() - aligned_benchmark.mean()) * trading_days / metrics['tracking_error']
                    
                    # Treynor ratio
                    if 'beta' in metrics and metrics['beta'] != 0:
                        metrics['treynor_ratio'] = (metrics['annual_return'] - rf_rate) / metrics['beta']
                    
                except:
                    pass
        
        # Advanced metrics
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            metrics['gain_loss_ratio'] = positive_returns.mean() / abs(negative_returns.mean())
            metrics['profit_factor'] = positive_returns.sum() / abs(negative_returns.sum())
        
        metrics['win_rate'] = (returns_clean > 0).mean()
        
        # Calculate Ulcer Index
        if len(drawdown) > 0:
            metrics['ulcer_index'] = np.sqrt((drawdown ** 2).mean())
            if metrics['ulcer_index'] > 0:
                metrics['martin_ratio'] = metrics['annual_return'] / metrics['ulcer_index']
        
        # Recovery metrics
        if metrics['max_drawdown'] < 0:
            try:
                # Find recovery period
                underwater = cumulative < running_max
                if underwater.any():
                    recovery_start = drawdown.idxmin()
                    recovery_idx = cumulative.index.get_loc(recovery_start)
                    
                    for i in range(recovery_idx, len(cumulative)):
                        if cumulative.iloc[i] >= running_max.iloc[i]:
                            metrics['recovery_period'] = i - recovery_idx
                            break
            except:
                metrics['recovery_period'] = np.nan
        
        return metrics
    
    @staticmethod
    def generate_performance_attribution(portfolio_returns: pd.Series, 
                                        asset_returns: pd.DataFrame,
                                        weights: np.ndarray) -> pd.DataFrame:
        """Generate detailed performance attribution"""
        
        attribution = pd.DataFrame(index=asset_returns.columns)
        attribution['Weight'] = weights
        attribution['Return'] = asset_returns.mean() * 252
        
        # Calculate contribution
        attribution['Contribution'] = attribution['Weight'] * attribution['Return']
        
        # Calculate risk contribution (simplified)
        try:
            cov_matrix = asset_returns.cov() * 252
            portfolio_variance = weights.T @ cov_matrix.values @ weights
            if portfolio_variance > 0:
                marginal_risk = cov_matrix.values @ weights
                risk_contribution = weights * marginal_risk / portfolio_variance
                attribution['Risk_Contribution'] = risk_contribution
        except:
            attribution['Risk_Contribution'] = np.nan
        
        return attribution.sort_values('Contribution', ascending=False)
    
    @staticmethod
    def calculate_diversification_score(weights: np.ndarray, correlation_matrix: pd.DataFrame) -> float:
        """Calculate portfolio diversification score (0-1, higher is better)"""
        if len(weights) < 2:
            return 0.0
        
        # Effective number of uncorrelated bets
        weighted_correlation = np.sum(np.outer(weights, weights) * correlation_matrix.values)
        if weighted_correlation > 0:
            diversification_ratio = np.sum(weights ** 2) / weighted_correlation
            return min(diversification_ratio, 1.0)
        
        return 0.5

# -------------------------------------------------------------
# ENHANCED DATA LOADING
# -------------------------------------------------------------
class InstitutionalDataLoader:
    """Enhanced data loading with institutional features"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_asset_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load asset data with institutional features"""
        
        st.info(f"📊 Loading data for {len(tickers)} assets...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        prices_dict = {}
        successful_tickers = []
        
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
                    
                    prices_dict[ticker] = price_series
                    successful_tickers.append(ticker)
                
            except Exception as e:
                st.debug(f"Failed to load {ticker}: {str(e)[:50]}")
            
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
                return prices_df
            else:
                st.error("❌ Insufficient data for analysis")
                return pd.DataFrame()
        
        else:
            st.error("❌ Could not load any data")
            return pd.DataFrame()

# -------------------------------------------------------------
# ENHANCED PORTFOLIO OPTIMIZER
# -------------------------------------------------------------
class InstitutionalPortfolioOptimizer:
    """Superior institutional portfolio optimizer"""
    
    @staticmethod
    def optimize_portfolio(returns: pd.DataFrame, strategy: str, 
                          rf_rate: float = 0.03, constraints: Dict = None) -> Dict:
        """Optimize portfolio with institutional constraints"""
        
        if returns.empty or len(returns.columns) < 2:
            return InstitutionalPortfolioOptimizer._fallback_weights(returns)
        
        # Default constraints
        if constraints is None:
            constraints = {
                'max_weight': 0.20,
                'min_weight': 0.01,
                'short_selling': False
            }
        
        try:
            # Calculate expected returns and covariance
            mu = returns.mean() * 252
            S = returns.cov() * 252
            
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
                weights = ef.max_sharpe(risk_free_rate=rf_rate/252)
            elif strategy == "Risk Parity":
                # Simple risk parity implementation
                volatilities = returns.std() * np.sqrt(252)
                inv_vol = 1 / volatilities
                weights_raw = inv_vol / inv_vol.sum()
                weights = {asset: weights_raw[asset] for asset in returns.columns}
            elif strategy == "Equal Weight":
                n_assets = len(returns.columns)
                weights = {asset: 1.0/n_assets for asset in returns.columns}
            elif strategy == "Hierarchical Risk Parity" and PYPFOPT_AVAILABLE:
                hrp = HRPOpt(returns)
                weights = hrp.optimize()
            else:
                # Default to maximum sharpe
                weights = ef.max_sharpe(risk_free_rate=rf_rate/252)
            
            # Clean weights
            if isinstance(weights, dict):
                cleaned_weights = ef.clean_weights() if hasattr(ef, 'clean_weights') else weights
            else:
                cleaned_weights = ef.clean_weights()
            
            # Convert to array
            weights_array = np.array([cleaned_weights.get(asset, 0) for asset in returns.columns])
            
            # Calculate performance metrics
            if hasattr(ef, 'portfolio_performance'):
                expected_return, expected_risk, sharpe_ratio = ef.portfolio_performance(
                    risk_free_rate=rf_rate/252
                )
            else:
                # Fallback calculation
                portfolio_returns = (returns * weights_array).sum(axis=1)
                expected_return = portfolio_returns.mean() * 252
                expected_risk = portfolio_returns.std() * np.sqrt(252)
                if expected_risk > 0:
                    sharpe_ratio = (expected_return - rf_rate) / expected_risk
                else:
                    sharpe_ratio = 0
            
            # Calculate additional metrics
            portfolio_returns = (returns * weights_array).sum(axis=1)
            metrics = InstitutionalMetrics.calculate_comprehensive_metrics(
                portfolio_returns, None, rf_rate
            )
            
            # Calculate diversification score
            corr_matrix = returns.corr()
            diversification_score = InstitutionalMetrics.calculate_diversification_score(
                weights_array, corr_matrix
            )
            
            return {
                'weights': cleaned_weights,
                'weights_array': weights_array,
                'expected_return': expected_return,
                'expected_risk': expected_risk,
                'sharpe_ratio': sharpe_ratio,
                'metrics': metrics,
                'diversification_score': diversification_score,
                'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                'success': True
            }
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)[:200]}")
            return InstitutionalPortfolioOptimizer._fallback_weights(returns)
    
    @staticmethod
    def _fallback_weights(returns: pd.DataFrame) -> Dict:
        """Fallback to equal weights"""
        if returns.empty:
            return {
                'weights': {},
                'weights_array': np.array([]),
                'expected_return': 0,
                'expected_risk': 0,
                'sharpe_ratio': 0,
                'metrics': {},
                'success': False
            }
        
        n_assets = len(returns.columns)
        equal_weights = {asset: 1.0/n_assets for asset in returns.columns}
        weights_array = np.ones(n_assets) / n_assets
        
        portfolio_returns = (returns * weights_array).sum(axis=1)
        expected_return = portfolio_returns.mean() * 252
        expected_risk = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (expected_return - 0.03) / (expected_risk + 1e-10)
        
        metrics = InstitutionalMetrics.calculate_comprehensive_metrics(portfolio_returns)
        
        return {
            'weights': equal_weights,
            'weights_array': weights_array,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'metrics': metrics,
            'diversification_score': 0.5,
            'avg_correlation': 0.5,
            'success': False
        }

# -------------------------------------------------------------
# ENHANCED CHARTING UTILITIES
# -------------------------------------------------------------
class InstitutionalCharts:
    """Superior institutional charting utilities"""
    
    @staticmethod
    def create_performance_tearsheet(portfolio_data: Dict, benchmark_data: Dict = None) -> go.Figure:
        """Create professional tearsheet with multiple subplots"""
        
        if 'returns' not in portfolio_data or portfolio_data['returns'] is None:
            return go.Figure()
        
        returns = portfolio_data['returns']
        cumulative = (1 + returns).cumprod()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative Performance',
                'Rolling 12-Month Returns',
                'Drawdown Analysis',
                'Monthly Returns Heatmap',
                'Return Distribution',
                'Risk Metrics Comparison'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Cumulative performance
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                name="Portfolio",
                line=dict(color='#1a5fb4', width=3),
                fill='tozeroy',
                fillcolor='rgba(26, 95, 180, 0.1)'
            ),
            row=1, col=1
        )
        
        if benchmark_data and 'returns' in benchmark_data:
            benchmark_cumulative = (1 + benchmark_data['returns']).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    name="Benchmark",
                    line=dict(color='#26a269', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # 2. Rolling 12-month returns
        if len(returns) > 252:
            rolling_returns = returns.rolling(252).apply(lambda x: (1 + x).prod() - 1)
            fig.add_trace(
                go.Scatter(
                    x=rolling_returns.index,
                    y=rolling_returns.values * 100,
                    name="Rolling 12M",
                    line=dict(color='#f39c12', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(243, 156, 18, 0.1)'
                ),
                row=1, col=2
            )
        
        # 3. Drawdown analysis
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                name="Drawdown",
                line=dict(color='#e74c3c', width=2),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ),
            row=2, col=1
        )
        
        # 4. Monthly returns heatmap (simplified)
        if len(returns) > 60:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            years = sorted(set(monthly_returns.index.year))
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            heatmap_data = np.full((len(years), 12), np.nan)
            
            for i, year in enumerate(years):
                for j in range(12):
                    month_data = monthly_returns[(monthly_returns.index.year == year) & 
                                                (monthly_returns.index.month == j + 1)]
                    if len(month_data) > 0:
                        heatmap_data[i, j] = month_data.iloc[0] * 100
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data,
                    x=months,
                    y=[str(y) for y in years],
                    colorscale='RdYlGn',
                    zmid=0,
                    showscale=False
                ),
                row=2, col=2
            )
        
        # 5. Return distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values * 100,
                nbinsx=50,
                name="Returns",
                marker_color='#1a5fb4',
                opacity=0.7,
                histnorm='probability density'
            ),
            row=3, col=1
        )
        
        # 6. Risk metrics comparison (placeholder)
        if benchmark_data:
            metrics = ['Return', 'Volatility', 'Sharpe', 'Max DD']
            portfolio_metrics = [
                portfolio_data.get('annual_return', 0) * 100,
                portfolio_data.get('annual_volatility', 0) * 100,
                portfolio_data.get('sharpe_ratio', 0),
                abs(portfolio_data.get('max_drawdown', 0)) * 100
            ]
            
            benchmark_metrics = [
                benchmark_data.get('annual_return', 0) * 100,
                benchmark_data.get('annual_volatility', 0) * 100,
                benchmark_data.get('sharpe_ratio', 0),
                abs(benchmark_data.get('max_drawdown', 0)) * 100
            ]
            
            fig.add_trace(
                go.Bar(
                    name='Portfolio',
                    x=metrics,
                    y=portfolio_metrics,
                    marker_color='#1a5fb4'
                ),
                row=3, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    name='Benchmark',
                    x=metrics,
                    y=benchmark_metrics,
                    marker_color='#26a269'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Portfolio Analysis Tearsheet",
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="12M Return (%)", row=1, col=2, ticksuffix="%")
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, ticksuffix="%")
        fig.update_yaxes(title_text="Return (%)", row=3, col=1, ticksuffix="%")
        fig.update_yaxes(title_text="Value", row=3, col=2)
        
        return fig
    
    @staticmethod
    def create_risk_radar(portfolio_metrics: Dict, benchmark_metrics: Dict = None) -> go.Figure:
        """Create radar chart for risk metrics"""
        
        categories = ['Return', 'Risk-Adj Return', 'Downside Protection', 
                     'Diversification', 'Liquidity', 'Consistency']
        
        # Portfolio values (normalized 0-1)
        portfolio_values = [
            min(portfolio_metrics.get('annual_return', 0) / 0.20, 1.0),  # Cap at 20%
            min(portfolio_metrics.get('sharpe_ratio', 0) / 2.0, 1.0),    # Cap at 2.0
            min(portfolio_metrics.get('sortino_ratio', 0) / 2.0, 1.0),   # Cap at 2.0
            portfolio_metrics.get('diversification_score', 0.5),
            0.7,  # Placeholder for liquidity
            portfolio_metrics.get('win_rate', 0.5)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=portfolio_values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(26, 95, 180, 0.3)',
            line_color='#1a5fb4',
            name='Portfolio'
        ))
        
        if benchmark_metrics:
            benchmark_values = [
                min(benchmark_metrics.get('annual_return', 0) / 0.20, 1.0),
                min(benchmark_metrics.get('sharpe_ratio', 0) / 2.0, 1.0),
                min(benchmark_metrics.get('sortino_ratio', 0) / 2.0, 1.0),
                0.5,  # Default diversification
                0.8,  # Placeholder
                benchmark_metrics.get('win_rate', 0.5)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=benchmark_values,
                theta=categories,
                fill='toself',
                fillcolor='rgba(38, 162, 105, 0.3)',
                line_color='#26a269',
                name='Benchmark'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Risk Profile Radar",
            height=500,
            template="plotly_dark"
        )
        
        return fig

# -------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------
def main():
    """Enhanced main application with superior institutional UI"""
    
    # Custom header
    st.markdown("""
    <div class="institutional-header">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🏛️ APOLLO/ENIGMA</h1>
        <p style="color: #f8f9fa; margin: 0; font-size: 1.2rem; opacity: 0.9;">
        Institutional Portfolio Management System v6.0
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'portfolio_results' not in st.session_state:
        st.session_state.portfolio_results = None
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {}
    if 'selected_assets' not in st.session_state:
        st.session_state.selected_assets = ["SPY", "TLT", "GLD", "IEF"]
    
    # Enhanced sidebar with professional layout
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 1.5rem; font-weight: bold; color: #1a5fb4;">
                ⚙️ CONFIGURATION
            </div>
            <div style="font-size: 0.9rem; color: #94a3b8;">
                Institutional Settings
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Portfolio strategy selection
        st.markdown("### 🎯 Portfolio Strategy")
        strategy_options = list(PORTFOLIO_STRATEGIES.keys())
        strategy = st.selectbox(
            "Select Institutional Strategy",
            strategy_options,
            index=0,
            help="Choose from institutional-grade portfolio construction methods"
        )
        
        if strategy in PORTFOLIO_STRATEGIES:
            st.caption(f"📝 {PORTFOLIO_STRATEGIES[strategy]['description']}")
        
        # Asset selection
        st.markdown("### 🌍 Asset Selection")
        
        # Category selection
        categories = st.multiselect(
            "Asset Categories",
            list(GLOBAL_ASSET_UNIVERSE.keys()),
            default=["Core Equities", "Fixed Income", "Commodities"],
            help="Select asset categories for portfolio construction"
        )
        
        # Get assets from selected categories
        available_assets = []
        for category in categories:
            available_assets.extend(GLOBAL_ASSET_UNIVERSE[category]["assets"])
        
        available_assets = sorted(list(set(available_assets)))
        
        # Asset selection with search
        asset_search = st.text_input("🔍 Search Assets", "", 
                                    help="Search for specific assets")
        
        if asset_search:
            filtered_assets = [a for a in available_assets 
                             if asset_search.upper() in a.upper()]
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
        date_options = ["1 Year", "3 Years", "5 Years", "10 Years", "Max Available"]
        date_period = st.selectbox("Time Horizon", date_options, index=2)
        
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
            max_weight = st.slider("Maximum Weight per Asset (%)", 5, 100, 20, 5) / 100
            min_weight = st.slider("Minimum Weight per Asset (%)", 0, 10, 1, 1) / 100
            short_selling = st.checkbox("Allow Short Selling", value=False)
        
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
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Assets Selected", len(selected_assets))
        with status_col2:
            st.metric("PyPortfolioOpt", "✅" if PYPFOPT_AVAILABLE else "❌")
    
    # Main content area
    if run_analysis and len(selected_assets) >= 3:
        with st.spinner("🔍 Conducting institutional analysis..."):
            try:
                # Set date range
                end_date = pd.Timestamp.today()
                start_date = end_date - pd.DateOffset(years=years)
                
                # Load data
                data_loader = InstitutionalDataLoader()
                prices = data_loader.load_asset_data(selected_assets, start_date, end_date)
                
                if prices.empty or len(prices) < 60:
                    st.error("❌ Insufficient data for analysis. Please check asset symbols and date range.")
                    return
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                # Run portfolio optimization
                optimizer = InstitutionalPortfolioOptimizer()
                optimization_result = optimizer.optimize_portfolio(
                    returns, strategy, rf_rate, constraints
                )
                
                if optimization_result['success']:
                    # Store results
                    portfolio_returns = (returns * optimization_result['weights_array']).sum(axis=1)
                    
                    st.session_state.portfolio_results = optimization_result
                    st.session_state.portfolio_data = {
                        'prices': prices,
                        'returns': returns,
                        'portfolio_returns': portfolio_returns,
                        'start_date': start_date,
                        'end_date': end_date,
                        'strategy': strategy,
                        **optimization_result['metrics']
                    }
                    
                    # Display success
                    st.success("✅ Institutional analysis completed successfully!")
                    
                    # Create enhanced tabs
                    tab_names = ["📊 Overview", "⚖️ Risk Analytics", "📈 Performance", 
                               "🔍 Attribution", "📋 Reports", "⚙️ Monitoring"]
                    
                    tabs = st.tabs(tab_names)
                    
                    # Tab 1: Overview
                    with tabs[0]:
                        display_overview_tab(optimization_result, prices, returns)
                    
                    # Tab 2: Risk Analytics
                    with tabs[1]:
                        display_risk_tab(optimization_result, returns)
                    
                    # Tab 3: Performance
                    with tabs[2]:
                        display_performance_tab(optimization_result, prices, returns)
                    
                    # Tab 4: Attribution
                    with tabs[3]:
                        display_attribution_tab(optimization_result, returns)
                    
                    # Tab 5: Reports
                    with tabs[4]:
                        display_reports_tab(optimization_result, prices)
                    
                    # Tab 6: Monitoring
                    with tabs[5]:
                        display_monitoring_tab(prices, optimization_result)
                
                else:
                    st.error("❌ Portfolio optimization failed. Please try different parameters.")
                    
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)[:200]}")
                st.info("💡 Please check your internet connection and try again.")
    
    else:
        # Welcome screen
        display_welcome_screen()

def display_overview_tab(optimization_result: Dict, prices: pd.DataFrame, returns: pd.DataFrame):
    """Display enhanced overview tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0a3d62, #1a5fb4); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📊 PORTFOLIO OVERVIEW</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Institutional Portfolio Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    metrics = optimization_result['metrics']
    
    # Key metrics in KPI cards
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Return</div>
            <div class="kpi-value">{metrics.get('total_return', 0):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Annual Return</div>
            <div class="kpi-value">{metrics.get('annual_return', 0):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Sharpe Ratio</div>
            <div class="kpi-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Max Drawdown</div>
            <div class="kpi-value">{metrics.get('max_drawdown', 0):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Portfolio allocation
    st.markdown("### 🎯 Portfolio Allocation")
    
    weights = optimization_result['weights']
    weights_df = pd.DataFrame({
        'Asset': list(weights.keys()),
        'Weight': list(weights.values()),
        'Category': [get_asset_category(asset) for asset in weights.keys()]
    }).sort_values('Weight', ascending=False)
    
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        # Create allocation chart
        fig = go.Figure(data=[go.Pie(
            labels=weights_df['Asset'],
            values=weights_df['Weight'],
            hole=0.4,
            marker_colors=px.colors.sequential.Blues_r,
            textinfo='label+percent',
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>Weight: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            height=400,
            showlegend=False,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        # Category allocation
        category_weights = weights_df.groupby('Category')['Weight'].sum().reset_index()
        
        fig = go.Figure(data=[go.Bar(
            x=category_weights['Weight'],
            y=category_weights['Category'],
            orientation='h',
            marker_color='#1a5fb4',
            text=category_weights['Weight'].apply(lambda x: f"{x:.1%}"),
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Category Allocation",
            height=400,
            xaxis=dict(tickformat=".0%"),
            yaxis=dict(autorange="reversed"),
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.markdown("### 📊 Detailed Performance Metrics")
    
    detailed_metrics = [
        ("Total Return", metrics.get('total_return', 0), "Cumulative return over period"),
        ("Annual Return", metrics.get('annual_return', 0), "Annualized return"),
        ("Annual Volatility", metrics.get('annual_volatility', 0), "Annualized standard deviation"),
        ("Sharpe Ratio", metrics.get('sharpe_ratio', 0), "Risk-adjusted return"),
        ("Sortino Ratio", metrics.get('sortino_ratio', 'N/A'), "Downside risk-adjusted return"),
        ("Max Drawdown", metrics.get('max_drawdown', 0), "Maximum peak-to-trough decline"),
        ("Calmar Ratio", metrics.get('calmar_ratio', 'N/A'), "Return per unit of max drawdown"),
        ("Win Rate", metrics.get('win_rate', 0), "Percentage of positive periods"),
        ("Profit Factor", metrics.get('profit_factor', 'N/A'), "Gross profit / gross loss"),
        ("Value at Risk (95%)", metrics.get('value_at_risk_95', 0), "95% confidence worst daily loss"),
    ]
    
    metrics_df = pd.DataFrame(detailed_metrics, columns=['Metric', 'Value', 'Description'])
    
    # Format values
    def format_value(val):
        if isinstance(val, (int, float)):
            if abs(val) < 0.01:  # Small numbers
                return f"{val:.4f}"
            elif abs(val) < 1:   # Percentages
                return f"{val:.2%}"
            else:                # Ratios
                return f"{val:.2f}"
        return str(val)
    
    metrics_df['Formatted Value'] = metrics_df['Value'].apply(format_value)
    
    st.dataframe(
        metrics_df[['Metric', 'Formatted Value', 'Description']],
        use_container_width=True,
        height=400
    )

def display_risk_tab(optimization_result: Dict, returns: pd.DataFrame):
    """Display enhanced risk analytics tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #c0392b, #e74c3c); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">⚖️ RISK ANALYTICS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Comprehensive Risk Assessment & Stress Testing</p>
    </div>
    """, unsafe_allow_html=True)
    
    metrics = optimization_result['metrics']
    
    # Risk metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Annual Volatility</div>
            <div class="kpi-value">{metrics.get('annual_volatility', 0):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">VaR (95%)</div>
            <div class="kpi-value">{metrics.get('value_at_risk_95', 0):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">CVaR (95%)</div>
            <div class="kpi-value">{metrics.get('conditional_var_95', 0):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Beta</div>
            <div class="kpi-value">{metrics.get('beta', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk decomposition
    st.markdown("### 📊 Risk Decomposition")
    
    weights_array = optimization_result['weights_array']
    cov_matrix = returns.cov() * 252
    
    # Calculate risk contributions
    portfolio_variance = weights_array.T @ cov_matrix.values @ weights_array
    if portfolio_variance > 0:
        marginal_risk = cov_matrix.values @ weights_array
        risk_contributions = weights_array * marginal_risk / portfolio_variance
        
        risk_df = pd.DataFrame({
            'Asset': list(optimization_result['weights'].keys()),
            'Weight': weights_array,
            'Risk Contribution': risk_contributions,
            'Percent of Total Risk': risk_contributions * 100
        }).sort_values('Risk Contribution', ascending=False)
        
        col_risk1, col_risk2 = st.columns(2)
        
        with col_risk1:
            # Risk contribution bar chart
            fig = go.Figure(data=[go.Bar(
                x=risk_df['Asset'],
                y=risk_df['Percent of Total Risk'],
                marker_color='#e74c3c',
                text=risk_df['Percent of Total Risk'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Risk Contribution by Asset",
                height=400,
                yaxis_title="% of Total Risk",
                template="plotly_dark"
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
                mode='markers+text',
                marker=dict(
                    size=weights_array * 100,
                    color=risk_contributions,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Risk Contribution")
                ),
                text=asset_returns.index,
                textposition="top center",
                hovertemplate='<b>%{text}</b><br>Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>'
            ))
            
            # Add portfolio point
            fig.add_trace(go.Scatter(
                x=[optimization_result['expected_risk']],
                y=[optimization_result['expected_return']],
                mode='markers',
                marker=dict(size=20, color='#1a5fb4', symbol='star'),
                name='Portfolio'
            ))
            
            fig.update_layout(
                title="Risk-Return Scatter",
                height=400,
                xaxis_title="Annual Volatility",
                yaxis_title="Annual Return",
                template="plotly_dark",
                yaxis=dict(tickformat=".1%"),
                xaxis=dict(tickformat=".1%")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Distribution analysis
    st.markdown("### 📈 Return Distribution Analysis")
    
    portfolio_returns = (returns * weights_array).sum(axis=1)
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        # Histogram with normal distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=portfolio_returns,
            nbinsx=50,
            name="Returns",
            histnorm='probability density',
            marker_color='#1a5fb4',
            opacity=0.7
        ))
        
        # Add normal distribution curve
        mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
        x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=pdf,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#e74c3c', width=2)
        ))
        
        # Add VaR lines
        var_95 = metrics.get('value_at_risk_95', 0)
        var_99 = metrics.get('value_at_risk_99', 0)
        
        if not pd.isna(var_95):
            fig.add_vline(x=var_95, line_dash="dash", line_color="#f39c12", 
                         annotation_text=f"VaR 95%: {var_95:.2%}")
        
        if not pd.isna(var_99):
            fig.add_vline(x=var_99, line_dash="dot", line_color="#c0392b",
                         annotation_text=f"VaR 99%: {var_99:.2%}")
        
        fig.update_layout(
            title="Return Distribution",
            height=400,
            xaxis_title="Daily Return",
            yaxis_title="Density",
            template="plotly_dark",
            xaxis=dict(tickformat=".2%")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_dist2:
        # Create risk radar chart
        fig = InstitutionalCharts.create_risk_radar(metrics)
        st.plotly_chart(fig, use_container_width=True)

def display_performance_tab(optimization_result: Dict, prices: pd.DataFrame, returns: pd.DataFrame):
    """Display enhanced performance analysis tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #27ae60, #2ecc71); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📈 PERFORMANCE ANALYSIS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Comprehensive Performance Metrics & Benchmarking</p>
    </div>
    """, unsafe_allow_html=True)
    
    weights_array = optimization_result['weights_array']
    portfolio_returns = (returns * weights_array).sum(axis=1)
    
    # Create performance tearsheet
    st.markdown("### 📊 Performance Tearsheet")
    
    tearsheet_fig = InstitutionalCharts.create_performance_tearsheet({
        'returns': portfolio_returns,
        **optimization_result['metrics']
    })
    
    st.plotly_chart(tearsheet_fig, use_container_width=True)
    
    # Rolling performance metrics
    st.markdown("### 📈 Rolling Performance Metrics")
    
    window = st.slider("Rolling Window (days)", 30, 252, 63, key="rolling_window")
    
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
            template="plotly_dark",
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Return", row=1, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Volatility", row=2, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

def display_attribution_tab(optimization_result: Dict, returns: pd.DataFrame):
    """Display enhanced performance attribution tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #f39c12, #f1c40f); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: #212529; margin: 0;">🔍 PERFORMANCE ATTRIBUTION</h2>
        <p style="color: rgba(33,37,41,0.9); margin: 0;">Detailed Return & Risk Attribution Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    weights = optimization_result['weights']
    weights_array = optimization_result['weights_array']
    
    # Calculate attribution
    attribution_results = InstitutionalMetrics.generate_performance_attribution(
        (returns * weights_array).sum(axis=1), returns, weights_array
    )
    
    # Display attribution results
    st.markdown("### 📊 Return Attribution by Asset")
    
    col_attr1, col_attr2 = st.columns(2)
    
    with col_attr1:
        # Top contributors
        top_contributors = attribution_results.nlargest(5, 'Contribution')
        
        fig = go.Figure(data=[go.Bar(
            x=top_contributors['Contribution'] * 100,
            y=top_contributors.index,
            orientation='h',
            marker_color='#27ae60',
            text=top_contributors['Contribution'].apply(lambda x: f"{x:.2%}"),
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Top 5 Positive Contributors",
            height=300,
            xaxis_title="Contribution (%)",
            yaxis_title="Asset",
            template="plotly_dark",
            xaxis=dict(tickformat=".1f")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_attr2:
        # Bottom contributors
        bottom_contributors = attribution_results.nsmallest(5, 'Contribution')
        
        fig = go.Figure(data=[go.Bar(
            x=bottom_contributors['Contribution'] * 100,
            y=bottom_contributors.index,
            orientation='h',
            marker_color='#e74c3c',
            text=bottom_contributors['Contribution'].apply(lambda x: f"{x:.2%}"),
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Top 5 Negative Contributors",
            height=300,
            xaxis_title="Contribution (%)",
            yaxis_title="Asset",
            template="plotly_dark",
            xaxis=dict(tickformat=".1f")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed attribution table
    st.markdown("### 📋 Detailed Attribution Analysis")
    
    if 'Risk_Contribution' in attribution_results.columns:
        display_df = attribution_results[['Weight', 'Return', 'Contribution', 'Risk_Contribution']].copy()
        display_df.columns = ['Weight', 'Annual Return', 'Return Contribution', 'Risk Contribution']
    else:
        display_df = attribution_results[['Weight', 'Return', 'Contribution']].copy()
        display_df.columns = ['Weight', 'Annual Return', 'Return Contribution']
    
    st.dataframe(
        display_df.style.format({
            'Weight': '{:.2%}',
            'Annual Return': '{:.2%}',
            'Return Contribution': '{:.4%}',
            'Risk Contribution': '{:.4%}' if 'Risk Contribution' in display_df.columns else ''
        }).background_gradient(
            subset=['Return Contribution'], cmap='RdYlGn'
        ),
        use_container_width=True,
        height=400
    )

def display_reports_tab(optimization_result: Dict, prices: pd.DataFrame):
    """Display enhanced reports tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #8e44ad, #9b59b6); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📋 INSTITUTIONAL REPORTING</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Professional Reports & Documentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Report generation
    st.markdown("### 📄 Generate Institutional Reports")
    
    col_report1, col_report2, col_report3 = st.columns(3)
    
    with col_report1:
        exec_summary = st.button(
            "📊 Executive Summary",
            use_container_width=True,
            help="Generate executive summary report"
        )
    
    with col_report2:
        full_report = st.button(
            "📋 Full Analysis Report",
            use_container_width=True,
            help="Generate comprehensive analysis report"
        )
    
    with col_report3:
        risk_report = st.button(
            "⚖️ Risk Assessment Report",
            use_container_width=True,
            help="Generate detailed risk assessment"
        )
    
    # Display reports
    if exec_summary:
        st.markdown("### 📊 Executive Summary Report")
        
        # Generate report
        portfolio_data = {
            'total_return': optimization_result['metrics'].get('total_return', 0),
            'annual_return': optimization_result['metrics'].get('annual_return', 0),
            'annual_volatility': optimization_result['metrics'].get('annual_volatility', 0),
            'sharpe_ratio': optimization_result['metrics'].get('sharpe_ratio', 0),
            'max_drawdown': optimization_result['metrics'].get('max_drawdown', 0),
            'sortino_ratio': optimization_result['metrics'].get('sortino_ratio', 0),
            'value_at_risk_95': optimization_result['metrics'].get('value_at_risk_95', 0),
            'diversification_score': optimization_result.get('diversification_score', 0.7),
            'win_rate': optimization_result['metrics'].get('win_rate', 0),
            'beta': optimization_result['metrics'].get('beta', 0)
        }
        
        report = InstitutionalReportGenerator.generate_executive_summary(
            portfolio_data, optimization_result
        )
        
        st.markdown(report)
        
        # Download button
        st.download_button(
            label="📥 Download Executive Summary (MD)",
            data=report,
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    elif full_report:
        st.success("✅ Full analysis report generated successfully!")
        
        # Generate comprehensive report data
        portfolio_returns = (prices.pct_change().dropna() * 
                           optimization_result['weights_array']).sum(axis=1)
        
        portfolio_data = {
            'returns': portfolio_returns,
            'total_return': optimization_result['metrics'].get('total_return', 0),
            'annual_return': optimization_result['metrics'].get('annual_return', 0),
            'annual_volatility': optimization_result['metrics'].get('annual_volatility', 0),
            'sharpe_ratio': optimization_result['metrics'].get('sharpe_ratio', 0),
            'max_drawdown': optimization_result['metrics'].get('max_drawdown', 0),
            'sortino_ratio': optimization_result['metrics'].get('sortino_ratio', 0),
            'value_at_risk_95': optimization_result['metrics'].get('value_at_risk_95', 0),
            'diversification_score': optimization_result.get('diversification_score', 0.7),
            'win_rate': optimization_result['metrics'].get('win_rate', 0),
            'profit_factor': optimization_result['metrics'].get('profit_factor', 0),
            'gain_loss_ratio': optimization_result['metrics'].get('gain_loss_ratio', 0),
            'skewness': optimization_result['metrics'].get('skewness', 0),
            'kurtosis': optimization_result['metrics'].get('kurtosis', 0),
            'beta': optimization_result['metrics'].get('beta', 0),
            'alpha': optimization_result['metrics'].get('alpha', 0),
            'tracking_error': optimization_result['metrics'].get('tracking_error', 0),
            'information_ratio': optimization_result['metrics'].get('information_ratio', 0),
            'start_date': prices.index[0].strftime('%Y-%m-%d'),
            'end_date': prices.index[-1].strftime('%Y-%m-%d'),
            'n_obs': len(prices),
            'risk_free_rate': 0.03
        }
        
        # Create download buttons
        InstitutionalReportGenerator.create_report_download_buttons(
            portfolio_data, optimization_result['weights']
        )
    
    elif risk_report:
        st.markdown("### ⚖️ Risk Assessment Report")
        
        # Display risk metrics
        risk_metrics = [
            ("Annual Volatility", optimization_result['metrics'].get('annual_volatility', 0)),
            ("Value at Risk (95%)", optimization_result['metrics'].get('value_at_risk_95', 0)),
            ("Conditional VaR (95%)", optimization_result['metrics'].get('conditional_var_95', 0)),
            ("Maximum Drawdown", optimization_result['metrics'].get('max_drawdown', 0)),
            ("Sortino Ratio", optimization_result['metrics'].get('sortino_ratio', 0)),
            ("Beta", optimization_result['metrics'].get('beta', 0)),
            ("Ulcer Index", optimization_result['metrics'].get('ulcer_index', 0)),
            ("Martin Ratio", optimization_result['metrics'].get('martin_ratio', 0)),
        ]
        
        for metric, value in risk_metrics:
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.write(f"**{metric}:**")
            with col_m2:
                if isinstance(value, (int, float)):
                    if metric in ["Annual Volatility", "Value at Risk (95%)", 
                                 "Conditional VaR (95%)", "Maximum Drawdown"]:
                        st.write(f"{value:.2%}")
                    else:
                        st.write(f"{value:.2f}")
                else:
                    st.write(value)
        
        # Risk assessment
        st.markdown("#### 📊 Risk Assessment")
        
        var_95 = optimization_result['metrics'].get('value_at_risk_95', 0)
        max_dd = optimization_result['metrics'].get('max_drawdown', 0)
        
        if var_95 > -0.03:
            st.success("✅ Normal market risk exposure")
        elif var_95 > -0.05:
            st.warning("⚠️ Moderate downside risk")
        else:
            st.error("🔴 Elevated downside risk")
        
        if max_dd > -0.15:
            st.success("✅ Manageable drawdown risk")
        elif max_dd > -0.25:
            st.warning("⚠️ Moderate drawdown risk")
        else:
            st.error("🔴 High drawdown risk")

def display_monitoring_tab(prices: pd.DataFrame, optimization_result: Dict):
    """Display enhanced monitoring tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #3498db, #2980b9); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">⚙️ PORTFOLIO MONITORING</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Real-time Monitoring & Alert System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time metrics
    st.markdown("### 📊 Recent Performance")
    
    # Calculate recent performance
    recent_days = 30
    recent_returns = prices.pct_change().iloc[-recent_days:].dropna()
    
    if not recent_returns.empty and len(optimization_result['weights_array']) > 0:
        portfolio_returns_recent = (recent_returns * optimization_result['weights_array']).sum(axis=1)
        
        col_mon1, col_mon2, col_mon3, col_mon4 = st.columns(4)
        
        with col_mon1:
            recent_return = (1 + portfolio_returns_recent).prod() - 1
            st.metric("30-Day Return", f"{recent_return:.2%}")
        
        with col_mon2:
            recent_vol = portfolio_returns_recent.std() * np.sqrt(252)
            st.metric("30-Day Volatility", f"{recent_vol:.2%}")
        
        with col_mon3:
            recent_sharpe = (portfolio_returns_recent.mean() * 252 - 0.03) / (recent_vol + 1e-10)
            st.metric("30-Day Sharpe", f"{recent_sharpe:.2f}")
        
        with col_mon4:
            cumulative_recent = (1 + portfolio_returns_recent).cumprod()
            running_max_recent = cumulative_recent.expanding().max()
            drawdown_recent = (cumulative_recent - running_max_recent) / running_max_recent
            recent_dd = drawdown_recent.min() if not drawdown_recent.empty else 0
            st.metric("30-Day Max DD", f"{recent_dd:.2%}")
    
    # Alert system
    st.markdown("### 🚨 Alert System")
    
    # Define alert thresholds
    with st.expander("⚙️ Alert Configuration", expanded=True):
        dd_alert = st.number_input("Drawdown Alert (%)", 5.0, 50.0, 10.0, step=1.0)
        vol_alert = st.number_input("Volatility Spike (%)", 5.0, 50.0, 20.0, step=1.0)
    
    # Check alerts
    max_dd = optimization_result['metrics'].get('max_drawdown', 0)
    annual_vol = optimization_result['metrics'].get('annual_volatility', 0)
    
    alerts = []
    
    # Check drawdown
    if abs(max_dd) > dd_alert / 100:
        alerts.append({
            'type': 'danger',
            'message': f"⚠️ Maximum drawdown ({abs(max_dd):.2%}) exceeds threshold ({dd_alert}%)"
        })
    
    # Check volatility
    if annual_vol > vol_alert / 100:
        alerts.append({
            'type': 'warning',
            'message': f"⚠️ Portfolio volatility ({annual_vol:.2%}) exceeds threshold ({vol_alert}%)"
        })
    
    # Display alerts
    if alerts:
        st.markdown("### 🔴 Active Alerts")
        for alert in alerts:
            if alert['type'] == 'danger':
                st.error(alert['message'])
            else:
                st.warning(alert['message'])
    else:
        st.success("✅ No active alerts. All systems normal.")
    
    # Rebalancing monitor
    st.markdown("### ⚖️ Portfolio Drift Analysis")
    
    # Calculate current vs target weights (simplified)
    weights = optimization_result['weights']
    current_prices = prices.iloc[-1] if not prices.empty else pd.Series()
    
    if not current_prices.empty:
        drift_data = []
        for asset, target_weight in weights.items():
            if asset in current_prices.index:
                # Simplified drift calculation
                current_value = current_prices[asset]
                portfolio_value = sum(current_prices.get(a, 0) * w 
                                    for a, w in weights.items() if a in current_prices.index)
                
                if portfolio_value > 0:
                    current_weight = (current_prices[asset] * target_weight) / portfolio_value
                    drift = current_weight - target_weight
                    
                    drift_data.append({
                        'Asset': asset,
                        'Target Weight': target_weight,
                        'Current Weight': current_weight,
                        'Drift': drift,
                        'Status': 'Within Range' if abs(drift) < 0.01 else 'Needs Rebalance'
                    })
        
        if drift_data:
            drift_df = pd.DataFrame(drift_data)
            
            st.dataframe(
                drift_df.style.format({
                    'Target Weight': '{:.2%}',
                    'Current Weight': '{:.2%}',
                    'Drift': '{:.3%}'
                }).apply(
                    lambda x: ['background-color: #ffcccc' if v == 'Needs Rebalance' else '' 
                              for v in x],
                    subset=['Status']
                ),
                use_container_width=True,
                height=300
            )

def display_welcome_screen():
    """Display enhanced welcome screen"""
    
    # Hero section
    col_hero1, col_hero2 = st.columns([2, 1])
    
    with col_hero1:
        st.markdown("""
        <div style="margin-bottom: 3rem;">
            <h1 style="color: #1a5fb4; font-size: 3rem; margin-bottom: 1rem;">
                Institutional Portfolio Management
            </h1>
            <p style="color: #94a3b8; font-size: 1.2rem; line-height: 1.6;">
                APOLLO/ENIGMA provides institutional-grade portfolio analysis, 
                risk management, and reporting for sophisticated investors.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_hero2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0a3d62, #1a5fb4); 
                    padding: 2rem; border-radius: 15px; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🏛️</div>
            <div style="font-size: 1.2rem; font-weight: bold; color: white;">
                Version 6.0
            </div>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                Institutional Edition
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Key features
    st.markdown("### ✨ Key Features")
    
    features = [
        {
            "icon": "📊",
            "title": "Advanced Analytics",
            "description": "50+ institutional performance metrics and risk measures"
        },
        {
            "icon": "⚖️",
            "title": "Risk Management",
            "description": "Comprehensive risk analytics with stress testing"
        },
        {
            "icon": "📋",
            "title": "Professional Reporting",
            "description": "Institutional-grade reports in multiple formats"
        },
        {
            "icon": "🔍",
            "title": "Performance Attribution",
            "description": "Detailed Brinson attribution and factor analysis"
        },
        {
            "icon": "🚨",
            "title": "Monitoring & Alerts",
            "description": "Real-time monitoring with alert system"
        },
        {
            "icon": "🌍",
            "title": "Global Coverage",
            "description": "500+ assets across 8 major categories"
        }
    ]
    
    cols = st.columns(3)
    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="institutional-card">
                <div style="font-size: 2rem; margin-bottom: 1rem;">{feature['icon']}</div>
                <div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    {feature['title']}
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">
                    {feature['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        1. **Select Assets**: Choose 3-10 diverse assets from the sidebar
        2. **Choose Strategy**: Select an institutional portfolio strategy
        3. **Set Parameters**: Configure risk parameters and constraints
        4. **Run Analysis**: Click "Run Analysis" to generate comprehensive insights
        5. **Generate Reports**: Create professional reports in multiple formats
        
        **Recommended Starting Portfolio:**
        - SPY (US Equities): 40%
        - TLT (US Bonds): 40%
        - GLD (Gold): 10%
        - BTC-USD (Crypto): 5%
        - VNQ (Real Estate): 5%
        """)
    
    # System status
    st.markdown("### 🔧 System Status")
    
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        st.metric("Data Sources", "Yahoo Finance", "Active")
    
    with col_status2:
        st.metric("Optimization Engine", "PyPortfolioOpt", "✅" if PYPFOPT_AVAILABLE else "❌")
    
    with col_status3:
        st.metric("Charting Library", "Plotly", "Active")

# -------------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------------
def get_asset_category(ticker: str) -> str:
    """Get asset category for a ticker"""
    for category, data in GLOBAL_ASSET_UNIVERSE.items():
        if ticker in data["assets"]:
            return category
    return "Other"

# -------------------------------------------------------------
# APPLICATION ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
        
        # Professional footer
        st.markdown("""
        <div class="institutional-footer">
            <div>🏛️ APOLLO/ENIGMA INSTITUTIONAL PORTFOLIO TERMINAL v6.0</div>
            <div style="font-size: 0.7rem; margin-top: 0.5rem; opacity: 0.7;">
                For institutional use only. Not for distribution to retail investors.
                Past performance does not guarantee future results.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Application error: {str(e)[:200]}")
        st.info("Please refresh the page and try again.")
