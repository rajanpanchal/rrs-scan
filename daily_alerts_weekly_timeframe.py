import warnings
import time
import math
import os

import pandas as pd
import yfinance as yf
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import streamlit as st
import io
from st_aggrid import AgGrid, GridOptionsBuilder,JsCode,GridUpdateMode

# Email credentials
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # SSL port
GMAIL_USER = os.getenv("GMAIL_SENDER")

GMAIL_APP_PASSWORD = os.getenv("APP_PASSWORD")

# Email details
TO_EMAIL = os.getenv("TO_EMAILS")



# User-defined parameters
ref_ticker = "SPY"
sec_ticker = "XLY"  # Example sector ticker
rolling_length = 21
rolling_length_short = 21
rolling_length_long = 5
volume_weighted = True
latest_date = None
latest_row = None
# User-defined parameters
ref_ticker = "SPY"
period_1d = 5  # Length for price change calculation
atr_length_1d = 20  # ATR period for normalization
rvol_length = 5  # Relative volume length


#%% Set the display option to show 2 decimal places

pd.set_option('display.float_format', '{:.2f}'.format)
warnings.simplefilter(action='ignore', category=FutureWarning)


#%% Obtain tickers from raw nasdaq table. With data cleanup


def clean_metadata(metadata_csv):
    metadata = pd.read_csv(metadata_csv)
    metadata.dropna(subset=['Market Cap'], inplace=True)
    metadata = metadata.sort_values(by='Market Cap', ascending=False)
    metadata['Symbol'] = metadata['Symbol'].str.replace('/', '-')
    metadata['ticker'] = metadata['Symbol']
    metadata.reset_index(drop=True, inplace=True)
    return metadata 


def get_tickers(metadata, minval=0, maxval=1000):
    tickers = metadata['Symbol'][minval:maxval].tolist()
    return tickers

def get_data(ticker):
    file_path = "stock_data_spy.pkl"
    
    if os.path.exists(file_path):
        data = pd.read_pickle("stock_data_spy.pkl")
    else:
        data = yf.download(ticker, period='6mo', interval='1d', auto_adjust=True, progress=True, multi_level_index=False)
        #data.to_pickle("stock_data_spy.pkl")
        #data.to_csv("stock_data_dl_spy.csv")
    return data


#% Download stock data from yfinance
def download_data_wk(tickers):
    #yf.enable_debug_mode()
    # If this file exists then it wont read live data. This is only for testing. 
    file_path = "stock_data.pkl"
    
    if os.path.exists(file_path):
        data = pd.read_pickle("stock_data.pkl")
    else:
        data = yf.download(tickers, period='6mo', interval='1d', auto_adjust=True, progress=True)
        #data.to_pickle("stock_data.pkl")
        #data.to_csv("stock_data_dl.csv")
    return data

def calc_sma(data, length):
    return data['Close'].rolling(window=length).mean()



#% Input stock data into squeeze screener


import pandas as pd

# Relative Volume Calculation
def compute_rvol(df, length=rvol_length):
    df['avg_volume'] = df['volume'].rolling(window=length).mean()
    df['rvol'] = df['volume'] / df['avg_volume']
    return df

# ATR Calculation
def compute_atr(df, length=atr_length_1d):
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=length).mean()
    return df

# Power Index Calculation
def compute_power_index(df, period, atr_length):
    df = compute_atr(df, atr_length)
    df['price_change'] = df['close'] - df['close'].shift(period)
    df['power_index'] = df['price_change'] / df['atr'].shift(1)
    return df

def calculate_rrs_and_labels_for_last_5_days(data):
    data = data.sort_values(by=['ticker', 'date'])
    
    rrs_values = []
    rrs_labels = []
    date_values = []
    
    label_bins = [-np.inf, -0.66, -0.33, 0, 0.33, 0.66, np.inf]
    label_names = ["RW(Strong)", "RW (Moderate)", 
                   "RW (Slight)", "RS(Slight)", 
                   "RS(Moderate)", "RS(Strong)"]
    
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker]
        rrs_list = []
        rrs_label_list = []
        date_list = []

        for i in range(5):
            if len(ticker_data) > i:
                current_day = ticker_data.iloc[-(i+1)]
                prev_day = ticker_data.iloc[-(i+2)] if i < 4 else None
                rrs_value = current_day['power_index'] - (prev_day['power_index'] if prev_day is not None else 0)
                
                # RRS Labeling
                label = pd.cut([rrs_value], bins=label_bins, labels=label_names)[0]
                
                rrs_list.append(rrs_value)
                rrs_label_list.append(label)
                date_list.append(current_day['date'].strftime('%Y-%m-%d'))
            else:
                rrs_list.append(np.nan)
                rrs_label_list.append('N/A')
                date_list.append(np.nan)

        rrs_values.append(rrs_list)
        rrs_labels.append(rrs_label_list)
        date_values.append(date_list)

    # Create DataFrames for RRS values and the corresponding labels
    rrs_df = pd.DataFrame(rrs_values, index=data['ticker'].unique())
    rrs_label_df = pd.DataFrame(rrs_labels, index=data['ticker'].unique())
    date_df = pd.DataFrame(date_values, index=data['ticker'].unique())

    # Set the dates as column headers
    rrs_df.columns = [f'{date_df.iloc[0, i]}' for i in range(5)]
    rrs_label_df.columns = [f'{date_df.iloc[0, i]}' for i in range(5)]
    
    return rrs_df, rrs_label_df

# Compute RRS
def scanner_wk(data):
    ref_data = get_data(ref_ticker)
    ref_data['volume_average'] = ref_data['Volume'].mean()
    global latest_date, latest_row
    latest_date = ref_data.index[-1]
    latest_row = ref_data.iloc[-1]
    print(f"Latest Date: {latest_date}\n{latest_row}")
    tickers = list(data.columns.get_level_values(1).unique())
    results = pd.DataFrame()
    results2 = pd.DataFrame()
    rrs_list = []
    
    for ticker in tickers:
        #print(f"calculating for {ticker}")
        df = data.loc[:, (slice(None), ticker)].copy()
        df.dropna(inplace=True)
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
        df['ticker'] = ticker
        ref_data.columns = ref_data.columns.str.lower()
        
        if df.empty or len(df) < period_1d or 'close' not in df.columns:
            print(f"Skipping {ticker} due to insufficient data")
            continue  # Skip if 'close' column is missing or not enough data
        
        df['volume_average'] = df['volume'].mean()
        
        # Compute Power Index for stock and reference over last 5 days
        df = compute_power_index(df, period_1d, atr_length_1d)
        ref_data_pi = compute_power_index(ref_data.copy(), period_1d, atr_length_1d)
        
        # Ensure alignment by date
        df = df.merge(ref_data_pi[['power_index']], left_index=True, right_index=True, suffixes=('', '_ref'))
        
        # Compute RRS for the last 5 days
        df['rrs'] = (df['power_index'] - df['power_index_ref']).clip(-1, 1)
        
        # Label RRS values
        df['rrs_label'] = pd.cut(df['rrs'],
            bins=[-np.inf,  -0.66, -0.33, 0, 0.33, 0.66, np.inf],
            labels=["RW(Strong)", "RW(Moderate)", 
                    "RW(Slight)", "RS(Slight)", 
                    "RS(Moderate)", "RS(Strong)"])

        last_5_dates = df.index[df['rrs'].notna()].tolist()[-5:]

        new_cols = {}
        for date in last_5_dates:
            date_str = date.strftime('%Y-%m-%d')
            new_cols[f'rrs_{date_str}'] = df.loc[date, 'rrs']
            new_cols[f'rrs_label_{date_str}'] = df.loc[date, 'rrs_label']

        new_df = pd.DataFrame(new_cols, index=last_5_dates)
        df = pd.concat([df, new_df], axis=1)
       
        # Store only the last row for merging
        results = pd.concat([results, df.iloc[[-1]]])
        

    # Sort results by volume
    
    results = results.sort_values('volume_average', ascending=False)
    results.to_csv("stock_data_dl_rss.csv")
    return results  # Returning a single dataset with all data



    
# Function to format volume as K, M, B, etc.
def format_volume(value):
    """Formats volume or volume_average as K, M, B, etc."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"  # Format as billions
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"  # Format as millions
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"  # Format as thousands
    else:
        return str(value)  # Return the value as is for smaller numbers

#%% Load & download data

def createEmailTable(filter_squeezes_wk, description):
    import pandas as pd
    tickerCount = 1
    print(f"\n🟡 DEBUG: filter_squeezes_wk shape = {filter_squeezes_wk.shape}")
    print(f"🟡 DEBUG: Columns = {list(filter_squeezes_wk.columns)}")
    print(f"🟡 DEBUG: Tickers = {filter_squeezes_wk['ticker'].tolist() if 'ticker' in filter_squeezes_wk.columns else 'NO TICKER COL'}")

    html_snippet = f"""
<div>
    <p><strong>Latest Date:</strong> {latest_date}</p>
    <p><strong>Close:</strong> {latest_row["Close"]:.2f}</p>
    <p><strong>High:</strong> {latest_row["High"]:.2f}</p>
    <p><strong>Low:</strong> {latest_row["Low"]:.2f}</p>
    <p><strong>Open:</strong> {latest_row["Open"]:.2f}</p>
    <p><strong>Volume:</strong> {latest_row["Volume"]:,.2f}</p>
    <p><strong>Volume Average:</strong> {latest_row["volume_average"]:,.2f}</p>
</div>
"""
    emailbodystart = f"<b><font color='blue'>SPY</font></b>\n\n{html_snippet}"
    emailbodystart += description + "<br><br><table border='1' cellpadding='10'><tr>"
    emailbodystart += "<th></th><th>Ticker</th><th>Close Price</th>"

    # Identify only valid RRS dates with corresponding label columns
    rrs_base_dates = sorted(set(
        col.replace('rrs_', '') for col in filter_squeezes_wk.columns
        if col.startswith('rrs_') and not col.startswith('rrs_label_')
        and f"rrs_label_{col.replace('rrs_', '')}" in filter_squeezes_wk.columns
    ))[-5:]

    rrs_dates = [f"rrs_{d}" for d in rrs_base_dates]
    label_dates = [f"rrs_label_{d}" for d in rrs_base_dates]

    print(f"RRS Columns being used: {rrs_dates}")  # DEBUG

    for d in rrs_base_dates:
        emailbodystart += f"<th>{d} RRS</th><th>{d} Signal</th>"

    emailbodystart += "<th>Low</th><th>High</th><th>Finviz</th><th>Profitviz</th><th>Name</th><th>Vol</th><th>Volg</th><th>Vol Avg</th><th>Sector</th></tr>"
    emailbodyend = "</table><br><br>"

    for idx, last_price_row in filter_squeezes_wk.iterrows():
        try:
            ticker = last_price_row.get('ticker', 'N/A')
            print(f"\n=== Processing Ticker: {ticker} ===")
            print(last_price_row.to_dict())

            last_price = last_price_row.get('close', 'N/A')
            company_name = last_price_row.get('name', 'N/A')
            vol = last_price_row.get('volume', 'N/A')
            volavg = last_price_row.get('volume_average', 'N/A')
            sector = last_price_row.get('sector', 'N/A')
            low = last_price_row.get('low', 'N/A')
            high = last_price_row.get('high', 'N/A')

            if tickerCount <= 100:
                emailbodystart += f"<tr><td>{tickerCount}</td>"
                emailbodystart += f"<td><b><a href='https://www.tradingview.com/chart/?symbol={ticker}'>{ticker}</a></b></td>"
                emailbodystart += f"<td><font color='blue'>{last_price:.2f}</font></td>" if pd.notna(last_price) else "<td>N/A</td>"

                for rrs_col, label_col in zip(rrs_dates, label_dates):
                    rrs_val = last_price_row.get(rrs_col, 'N/A')
                    rrs_label = last_price_row.get(label_col, 'N/A')
                    emailbodystart += f"<td>{rrs_val:.2f}</td>" if pd.notna(rrs_val) else "<td>N/A</td>"
                    emailbodystart += f"<td>{rrs_label}</td>"

                emailbodystart += f"<td>{low:.2f}</td>" if pd.notna(low) else "<td>N/A</td>"
                emailbodystart += f"<td>{high:.2f}</td>" if pd.notna(high) else "<td>N/A</td>"
                emailbodystart += f"<td><a href='https://finviz.com/quote.ashx?t={ticker}&p=d'>{ticker}</a></td>"
                emailbodystart += f"<td><a href='https://profitviz.com/{ticker}'>{ticker}</a></td>"
                emailbodystart += f"<td>{company_name}</td>"
                emailbodystart += f"<td>{format_volume(vol)}</td>"
                emailbodystart += f"<td>{format_volume(vol)}</td>"
                emailbodystart += f"<td>{format_volume(volavg)}</td>"
                emailbodystart += f"<td>{sector}</td></tr>"
                tickerCount += 1
            else:
                break
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
            emailbodystart += f"<tr><td colspan='15'> An error occurred: {e} </td></tr>"

    return emailbodystart + emailbodyend




# Sample render_colored_text function for coloring rss_label values
# This function is used to map colors to rss_label values

def stock_alert():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    metadata_csv = csv_files[0]
    print(metadata_csv)
    metadata = clean_metadata(metadata_csv)
    tickers = get_tickers(metadata, minval=0, maxval=1000) 
    data_wk = download_data_wk(tickers)

    squeezes_wk_data = scanner_wk(data_wk)
    squeezes_wk_data = squeezes_wk_data.merge(metadata[['ticker','Name','Market Cap','Sector','Industry']], how='left', on='ticker')
    squeezes_wk_data.to_csv("stock_data_dl_rss2.csv")
    squeezes_wk_data = squeezes_wk_data.loc[squeezes_wk_data['volume_average'] > 250000]
    squeezes_wk_data.to_csv("stock_data_dl_rss2.csv")
    squeezes_wk_data = squeezes_wk_data.sort_values(by=['Market Cap','volume_average'], ascending=False)
    squeezes_wk_data.to_csv("stock_data_dl_rss3.csv")

    filter_squeezes_wk = squeezes_wk_data
    st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="collapsed", menu_items=None)

     # Apply formatting for volume and volume_average columns
    filter_squeezes_wk['volume'] = filter_squeezes_wk['volume'].apply(format_volume)
    filter_squeezes_wk['volume_average'] = filter_squeezes_wk['volume_average'].apply(format_volume)
    filter_squeezes_wk['Market Cap'] = filter_squeezes_wk['Market Cap'].apply(format_volume)

    # Format numerical columns to 2 decimal places
    filter_squeezes_wk['close'] = filter_squeezes_wk['close'].map('{:.2f}'.format)
    filter_squeezes_wk['open'] = filter_squeezes_wk['open'].map('{:.2f}'.format)
    filter_squeezes_wk['high'] = filter_squeezes_wk['high'].map('{:.2f}'.format)
    filter_squeezes_wk['low'] = filter_squeezes_wk['low'].map('{:.2f}'.format)
    filter_squeezes_wk['atr'] = filter_squeezes_wk['atr'].map('{:.2f}'.format)

    # Format `rss_*` columns to 2 decimal places using string formatting
    rrs_columns = [col for col in filter_squeezes_wk.columns if col.startswith('rrs_')]
    for col in rrs_columns:
        if pd.api.types.is_numeric_dtype(filter_squeezes_wk[col]):
            filter_squeezes_wk[col] = filter_squeezes_wk[col].map('{:.2f}'.format)

    # Clean column names by stripping spaces and hidden characters
    filter_squeezes_wk.columns = filter_squeezes_wk.columns.str.strip()

    # Columns to be ordered at the start
    order_columns = ['ticker', 'open', 'close', 'high', 'low', 'atr', 'volume', 'volume_average']

    # Extract first 5 rss_label_* columns and first 5 rss_* columns
    rss_label_columns = [col for col in filter_squeezes_wk.columns if col.startswith('rrs_label')][:6]  # First 5 rss_label columns
    rss_columns = [col for col in filter_squeezes_wk.columns if col.startswith('rrs_')][:10]  # First 5 rss columns

    # Extract other important columns
    other_columns = ['Name', 'Market Cap', 'Sector', 'Industry']

    # Exclude `rss` and `rss_label` columns from the selection
    columns_to_exclude = ['rrs', 'rrs_label']

    # Combine the columns in the desired order
    columns_to_display = order_columns + rss_label_columns + rss_columns + other_columns

    # Ensure columns exist
    available_columns = filter_squeezes_wk.columns
    columns_to_display = [col for col in columns_to_display if col in available_columns]

    # Exclude columns that should be removed (like 'rss', 'rss_label')
    columns_to_display = [col for col in columns_to_display if col not in columns_to_exclude]

    # Update DataFrame
    filtered_data = filter_squeezes_wk[columns_to_display]
    filtered_data = filtered_data.loc[:, ~filtered_data.columns.duplicated()]


    # JavaScript code for coloring cells based on the value of `rss_label_*`
    cellstyle_jscode = JsCode("""
    function(params){
        if (params.value == 'RW(Strong)') {
            return {
                'color': 'white',
                'backgroundColor': 'red',
                'font-weight': 'bold'
            }
        }
        if (params.value == 'RW(Slight)') {
            return {
                'color': 'white',
                'backgroundColor': '#F08080',
                'font-weight': 'bold'
            }
        }
        if (params.value == 'RW(Moderate)') {
            return {
                'color': 'white',
                'backgroundColor': 'darkred',
                'font-weight': 'bold'
            }
        }
        if (params.value == 'RS(Slight)') {
            return {
                'color': 'white',
                'backgroundColor': '#98FB98',
                'font-weight': 'bold'
            }
        }
        if (params.value == 'RS(Moderate)') {
            return {
                'color': 'white',
                'backgroundColor': 'limegreen',
                'font-weight': 'bold'
            }
        }
        if (params.value == 'RS(Strong)') {
            return {
                'color': 'white',
                'backgroundColor': 'green',
                'font-weight': 'bold'
            }
        }
        return null; // default case if no matching value
    }
    """)

    # Create AgGrid options
    grid_options = GridOptionsBuilder.from_dataframe(filtered_data)
    grid_options.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)  # Enable pagination with 50 rows per page
    grid_options.configure_column('ticker', pinned='left')  # Freeze the first column
    grid_options.configure_columns(filtered_data, cellStyle=cellstyle_jscode)  # Apply JS-based cellStyle to all columns
    grid_options.configure_selection(selection_mode="single", use_checkbox=True)
    grid_options.configure_default_column(filter=True)
    grid_options = grid_options.build()

    # Display the table using Ag-Grid
    AgGrid(filtered_data, gridOptions=grid_options,update_mode=GridUpdateMode.NO_UPDATE, allow_unsafe_jscode=True, enable_enterprise_modules=True, height=600)

    return "success"

def lambda_handler(event, context):
    name = event.get("name", "Guest")  # Get 'name' from event, default to 'Guest'
    message = stock_alert()
    
    return {
        "statusCode": 200,
        "body": message
}


if __name__ == "__main__":
    event = {}  # Mock event
    context = None  # Mock context
    print(lambda_handler(event, context))

    


