# Multi-Modal Alternative Data for Short-Term Equity Return Prediction

**Paper:** [Forecasting NVIDIA Residual Returns: A Multi-Modal Deep Learning Approach](https://www.researchgate.net/publication/398676247_Forecasting_NVIDIA_Residual_Returns_A_Multi-Modal_Deep_Learning_Approach?channel=doi&linkId=693f7c969aa6b4649dc09a73&showFulltext=true)

**Authors:** Krishna Tej Bhat, Ashwin Kumar, Abhiraj Ezhil

**Course:** SI 699 – Capstone Project, University of Michigan School of Information

**Term:** Fall 2025

## Abstract

This project investigates whether multi-modal alternative data sources—specifically news sentiment and SEC regulatory filings—can improve short-term stock return predictions for NVIDIA Corporation (NVDA) beyond what traditional time-series models achieve using price history alone. We develop a custom neural network architecture (GatedNewsNet) that fuses market technicals with text embeddings and compare its performance against ARIMA and GARCH baselines.

Over a 12-month test period (October 2024 – October 2025), our full model achieved a 78% realized return versus 31% for a buy-and-hold strategy, with a 52.23% directional accuracy (hit rate). The results suggest that SEC filings function as an "alpha generation" signal while news sentiment serves as a "risk filtering" mechanism for residual returns.

## Methodology

### Research Question

Can unstructured alternative data (news articles and SEC filings) provide incremental predictive power for NVIDIA's residual returns (alpha) after controlling for market exposure?

### Data Sources

- **Market Data:** Daily OHLCV for NVDA, AMD, TSM, and QQQ from Yahoo Finance (July 2021 – October 2025)
- **News Articles:** ~34,000 articles from StockNewsAPI with ticker-based filtering
- **SEC Filings:** 10-K, 10-Q, and 8-K forms from EDGAR for NVIDIA

### Technical Approach

1. **Baseline Models:** Established predictive baselines using ARIMA and GARCH models on residual returns (alpha) after removing market exposure (beta = 1.2 to QQQ).

2. **Text Processing:**
   - FinBERT embeddings for news sentiment analysis
   - Qwen 2.5 LLM for SEC filing categorization into impact categories (Earnings Surprise, Strategic Announcements, Capital Return Policy, Legal/ESG Issues)
   - Multi-head self-attention for daily news aggregation

3. **Model Architecture (GatedNewsNet):**
   - Dual-branch LSTM processing (market technicals + news embeddings)
   - Learnable gating mechanism conditioned on SEC filing impact scores
   - Directional Huber Loss function to penalize directional errors more heavily than magnitude errors
   - Separate Ridge Regression model for SEC filing event windows

4. **Ablation Study:** Evaluated five model configurations to isolate the contribution of each data modality:
   - NVDA-only features
   - Multi-asset market features
   - Market + SEC filings
   - Market + News sentiment
   - Full model (Market + News + SEC)

### Key Results

- **Realized Return:** 78% over 12-month test period (vs. 31% buy-and-hold)
- **Directional Accuracy:** 52.23% hit rate
- **Information Coefficient:** 0.347 (p < 0.001)
- **Key Finding:** SEC filings identified as primary alpha generator; news sentiment acts as risk filter

Ablation analysis revealed that news sentiment alone does not improve returns beyond market-only models, but when combined with SEC signals, it provides defensive value by filtering false positives.

## Repository Structure

```
nvidia-residual-returns/
├── README.md                          
├── notebooks/                         # Analysis notebooks (workflow order)
│   ├── 01_news_data_collection.ipynb     # StockNewsAPI data collection
│   ├── 02_article_embeddings.ipynb       # FinBERT embedding generation
│   ├── 03_sec_edgar_pipeline.ipynb       # SEC filing processing with Qwen LLM
│   ├── 04_time_series_baseline.ipynb    # ARIMA/GARCH baseline models
│   └── 05_deep_learning_models.ipynb    # GatedNewsNet architecture & training
├── data/
│   ├── raw/
│   │   └── articles/                  # Raw news articles from StockNewsAPI
│   ├── processed/
│   │   ├── embeddings/                # FinBERT daily aggregated vectors
│   │   └── *_sentiment.csv            # Article-level sentiment scores
│   └── splits/
│       ├── train/                     # Training data (Jul 2021 – Oct 2024)
│       └── test/                      # Test data (Oct 2024 – Oct 2025)
└── docs/                              # (Reserved for checkpoint documents)
```

### Notebook Descriptions

1. **01_news_data_collection.ipynb:** Implements backward monthly windowing to fetch historical news articles via StockNewsAPI for NVDA, AMD, TSM, and QQQ. Includes date parsing and pagination logic.

2. **02_article_embeddings.ipynb:** Generates FinBERT embeddings (768-dimensional vectors) for each news article and aggregates them to daily-level vectors using multi-head attention.

3. **03_sec_edgar_pipeline.ipynb:** Downloads SEC filings from EDGAR, extracts text from HTML/XML formats, and uses Qwen 2.5 LLM to classify filings into impact categories with numerical scores.

4. **04_time_series_baseline.ipynb:** Trains ARIMA(1,1,1) and GARCH(1,1) models on alpha (residual returns). Evaluates multiple strategies including momentum-based and ensemble approaches. Establishes baseline performance for comparison.

5. **05_deep_learning_models.ipynb:** Implements the GatedNewsNet architecture with ablation study. Contains model training, evaluation, and visualization code. Includes directional Huber loss implementation and model checkpointing.

## Archival Notice

**This repository serves as an archival record of work completed for the SI 699 capstone project.** The code was developed collaboratively in Google Colab with environment-specific file paths and dependencies. It is not intended to be directly executable and is provided as evidence of the analytical work performed.

To run this code, significant modifications would be required including:
- Updating file paths from Colab environment to local paths
- Installing appropriate Python dependencies
- Acquiring API keys for StockNewsAPI
- Setting up access to SEC EDGAR and appropriate LLM inference

## Data Sources & Citations

- **Yahoo Finance:** Historical price and volume data via yfinance Python package
- **StockNewsAPI:** Financial news articles (commercial API, subscription required)
- **SEC EDGAR:** Public company filings (https://www.sec.gov/edgar)
- **FinBERT:** Pre-trained sentiment model for financial text (Araci, 2019)
- **Qwen 2.5:** Large language model for text classification (Alibaba Cloud)

## Limitations & Future Work

This analysis is limited to a single equity (NVIDIA) over a specific time period that coincides with significant AI sector volatility. Results may not generalize to other stocks, sectors, or market regimes. Future work could explore:

- Multi-stock portfolios to validate cross-sectional predictive power
- Longer time horizons and different market conditions
- Alternative text embedding models (e.g., domain-adapted transformers)
- Real-time deployment considerations including latency and execution costs

## Technical Dependencies

The notebooks utilize the following key libraries (not exhaustive):
- PyTorch for deep learning models
- statsmodels and arch for time-series baselines
- transformers (HuggingFace) for FinBERT embeddings
- pandas, numpy for data manipulation
- yfinance for market data
- requests for API calls

---
