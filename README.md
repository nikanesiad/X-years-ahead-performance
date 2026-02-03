# Interactive Finance Dashboard — yfinance + Plotly

This is a Streamlit app that allows you to visualize historical stock, index, commodity, and bond data, compute forward returns, and explore interactive price charts.

---

## Features

1. **Top panel: X-year ahead returns**
   - Enter tickers (comma separated) and choose `X` (1–20 years) to compute future performance.
   - Shows a time series of forward returns:  
     \[
     \text{Forward return} = \frac{\text{Price at } t+X \text{ years}}{\text{Price at } t} - 1
     \]
   - Interactive Plotly chart with hover tooltips and range slider.

2. **Bottom panel: Price series**
   - Enter tickers (comma separated) to plot historical price series.
   - Interactive Plotly chart with zoom, pan, range slider.
   - Toggle y-axis between linear and logarithmic.

3. **Presets**
   - Global equity markets, USA vs commodities, USA vs bonds, or custom input.

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/nikanesiad/X-years-ahead-performance
cd streamlit_yfinance_dashboard
```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows (PowerShell)
.venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install --upgrade pip
pip install streamlit yfinance pandas numpy plotly
```
## Usage

Run the app locally:
```bash
streamlit run streamlit_yfinance_dashboard.py
```
Select tickers or presets.

Choose a date range and X-year forward return.

Interact with the charts using zoom, pan, and range slider.

## Notes

Uses st.cache_data to speed up repeated data downloads.

Forward returns are not meaningful for dates less than X years from today.

Adjust ttl in caching for faster updates in development.

### Author

Nik Anesiadis