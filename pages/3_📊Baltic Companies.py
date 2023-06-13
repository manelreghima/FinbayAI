import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from langchain.llms import OpenAI
from datetime import datetime
import yfinance as yf
import plotly.graph_objects as go

def read_data():
    data = pd.read_csv('data/company_ticker.csv')
    return data
llm = OpenAI(temperature=0)

def extract_ticker_symbol(input_text):
    data=read_data()
    prompt = 'Extract ticker symbol from this text: ' + input_text
    response = llm(prompt)
    ticker_symbol = response.split(": ")[-1].strip()  # Extract the ticker symbol from the response
    for index, row in data.iterrows():
    # Check if the symbol is found in symbol1
        if ticker_symbol in row['symbol1']:
        # Update the value symbol to the corresponding value in 'symbol2'
            ticker_symbol=row['symbol2']
    return ticker_symbol

def get_graph(symbol):
    today = datetime.now()
    formatted_today = today.strftime('%Y-%m-%d')
    data = yf.download(symbol, interval = '1mo', start="2023-01-01", end=formatted_today)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], fill='tozeroy', name='Adj Close'))
    fig1.update_layout(title=str(symbol)+' Historical Close Prices', xaxis_title='Date', yaxis_title='Price')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['Volume'], fill='tozeroy', name='Volume'))
    fig2.update_layout(title=str(symbol)+' Historical Volume', xaxis_title='Date', yaxis_title='Volume')
    

    # Display charts using Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

with st.sidebar:
    choose = option_menu("Companies you can currently ask Finbay AI about.",
                         ["APB Apranga", "Arco Vara AS", "Auga Group AB", "AS Baltika", "Coop Pank AS"],
                         
                         default_index=0,
                         styles={
                             "container": {"padding": "5!important", "background-color": "#fafafa"},
                             "nav-link": {"text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                             "nav-link-selected": {"background-color": "#02ab21"},
                         })
data=read_data()
df_company = data[data['company']==choose]
symbol = str(df_company['symbol2'].iloc[0])
# Move get_graph(symbol) outside the sidebar
get_graph(symbol)