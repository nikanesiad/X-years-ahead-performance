# streamlit_yfinance_dashboard.py
# Streamlit app: Interactive finance dashboard using yfinance + Plotly
# Features:
# 1) Top subplot: default ticker '^GSPC' with a free-text box for multiple tickers (comma separated).
#    - Interactive Plotly time-series with zoom, pan, range slider.
#    - Dropdown inside the chart to switch between linear and log y-axis.
# 2) Second subplot: separate tickers input (comma separated) and integer parameter X (1..20)
#    - Plots forward X-year performance (percentage) computed as (price_{t+X years}/price_t - 1)
# Development notes: run locally with `streamlit run streamlit_yfinance_dashboard.py`.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# ----------------------------- Utilities -----------------------------
@st.cache_data(ttl=3600)
def download_data(tickers, start, end, interval='1d'):
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',') if t.strip()]
    if not tickers:
        return pd.DataFrame()

    start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end).strftime('%Y-%m-%d')

    all_data = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_str,
                end=end_str,
                interval=interval,
                progress=False,
                auto_adjust=False
            )
            if df.empty or len(df) < 50:
                continue

            if "Adj Close" in df.columns:
                series = df[["Adj Close"]].copy()
            elif "Close" in df.columns:
                series = df[["Close"]].copy()
            else:
                continue

            series.columns = [ticker]
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            all_data[ticker] = series[ticker]

        except Exception as e:
            st.warning(f"Failed to fetch {ticker}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    out = pd.DataFrame(all_data)
    out.sort_index(inplace=True)
    return out


def compute_forward_year_return(df, years):
    out = {}
    for t in df.columns:
        s = df[t].dropna()
        if s.empty:
            continue
        idx = s.index
        target_idx = idx + pd.DateOffset(years=years)
        try:
            pos = idx.get_indexer(target_idx, method='nearest')
            forward_prices = s.values[pos]
            base_prices = s.values
            mask = pos != -1
            ret = np.full_like(base_prices, np.nan, dtype=np.double)
            ret[mask] = (forward_prices[mask] / base_prices[mask]) - 1.0
            out[t] = pd.Series(ret, index=idx)
        except Exception:
            forward_s = s.reindex(target_idx, method='nearest')
            ret = forward_s.values / s.values - 1.0
            out[t] = pd.Series(ret, index=idx)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)

# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(layout='wide', page_title='yfinance + Plotly Streamlit Dashboard')
st.title('Interactive Finance Dashboard — yfinance + Plotly')

st.markdown(
    """
    **What this app does**

    - Pulls historical adjusted prices from Yahoo Finance using `yfinance`.
    - Top panel: interactive time series (zoom, pan, range slider) with dropdown to toggle linear/log y-axis.
    - Second panel: computes *X-year ahead* performance (for integer X between 1 and 20) and shows the time series of those forward returns.

    **How to use**: enter tickers separated by commas (example: `AAPL, MSFT, GOOG`) and choose the date range and parameters.
    """
)

# Defaults: start = today - 20 years, end = today - 3 years
now = datetime.now()
default_start = (now - pd.DateOffset(years=40)).date()
default_end = (now - pd.DateOffset(years=0)).date()

with st.expander('Data & view settings', expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        user_tickers = st.text_input('Top subplot tickers (comma separated)', value='^GSPC')
        start_date = st.date_input('Start date', value=default_start)
        end_date = st.date_input('End date', value=default_end)
    with col2:
        second_tickers = st.text_input('Second subplot tickers (comma separated)', value='^GSPC, ^IXIC')
        x_years = st.slider('X — years ahead for performance', min_value=1, max_value=20, value=5)

# Download data
st.write('Downloading data...')
try:
    df_top = download_data(user_tickers, start_date, end_date)
    df_second = download_data(second_tickers, start_date, end_date)
except Exception as e:
    st.error(f'Error downloading data: {e}')
    st.stop()

# First plot
st.subheader('Top plot — Price series with linear/log toggle')
st.markdown("""
This plot shows historical price series for the selected tickers.
Use the dropdown in the top-right of the chart to switch between linear and logarithmic y-axis.
You can also zoom, pan, and adjust the range slider at the bottom.
""")
if df_top.empty:
    st.warning('No data found for the top tickers. Check ticker symbols and date range.')
else:
    # Use the earliest available data
    plot_start = df_top.index.min()
    fig = go.Figure()
    for t in df_top.columns:
        s = df_top[t].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name=t))

    fig.update_layout(
        title="Price series",
        xaxis=dict(rangeslider=dict(visible=True), range=[plot_start, df_top.index.max()]),
        updatemenus=[
            dict(
                type="buttons",
                x=1.05, y=1,
                buttons=[
                    dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
                    dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}])
                ]
            )
        ],
        height=1000  # Adjust top plot height
    )
    st.plotly_chart(fig, use_container_width=True, height=500)

# Second plot: X-year ahead performance
st.subheader('Second plot — X-year ahead performance (time series)')
st.markdown(f"""
This plot shows {x_years}-year ahead returns for the selected tickers.
Returns are calculated as (Price at t+{x_years} years / Price at t - 1) * 100.
The chart stops at {x_years} years before today, as forward returns beyond that are not meaningful.
""")
if df_second.empty:
    st.warning('No data found for the second tickers. Check ticker symbols and date range.')
else:
    forward_df = compute_forward_year_return(df_second, x_years)
    if forward_df.empty:
        st.warning('Could not compute forward returns for the chosen tickers/dates.')
    else:
        # Clip forward returns to end at today - X years
        max_date = pd.Timestamp(now - pd.DateOffset(years=x_years))
        forward_df = forward_df[forward_df.index <= max_date]

        # Use the earliest available data
        plot_start = forward_df.index.min()

        fig2 = go.Figure()
        for t in forward_df.columns:
            s = forward_df[t].dropna()
            fig2.add_trace(go.Scatter(x=s.index, y=s.values*100, mode='lines', name=t))

        fig2.update_layout(
            title=f"{x_years}-year ahead returns",
            yaxis_title=f"Return after {x_years} years (%)",
            xaxis=dict(rangeslider=dict(visible=True), range=[plot_start, forward_df.index.max()]),
            height=1000  # Adjust second plot height
        )
        st.plotly_chart(fig2, use_container_width=True, height=500)

# st.markdown('---')
# st.markdown('**Developer notes**: This app uses `st.cache_data` for downloaded price data to speed up development. When deploying, consider longer cache TTL or a more robust caching/storage for production.')

# st.markdown('**Run locally**')
# st.code('''
# # create a virtualenv, install dependencies and run
# python -m venv .venv
# source .venv/bin/activate   # mac/linux
# .venv\\Scripts\\activate      # windows (powershell)

# pip install --upgrade pip
# pip install streamlit yfinance pandas numpy plotly

# streamlit run streamlit_yfinance_dashboard.py
# ''')

# st.markdown('**End of app file**')
