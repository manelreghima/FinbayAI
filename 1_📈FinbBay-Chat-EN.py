import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import openai 
import yfinance as yf
import re
import os
from datetime import datetime
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.llms import OpenAI
import numpy as np
import pandas as pd
from requests.exceptions import HTTPError

st.set_page_config(
    page_title="Finbay AI",
    page_icon="data/finbay-logo.jpg",
)
# Load environment variables from .env file
load_dotenv()
# Access the API key
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

st.cache()
def read_data():
    data = pd.read_csv('data/company_ticker.csv')
    return data
llm = OpenAI(temperature=0)
def extract_symbol(input):
        # Generate a response
    prompt =  'Extract ticker symbol from this text:' + input
    return llm(prompt)  

def chat_query(prompt_prefix, text):
        # Generate a response
    prompt =  prompt_prefix + ': ' + text
    return llm(prompt) 

def extract_company_name(input):
        # Generate a response
    prompt =  'Extract company name from this text:' + input
    return llm(prompt) 

st.title("Welcome to FinbayAI")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


# container for text box
container = st.container()
questions=["What is the market cap of DGR1R.RG?","What is the forward PE of PKG1T?",
           "Who is the CEO of LHV1T.TL?","What is the profit margin of TSLA? "]


def process_question(question):
    data=read_data()
    user_input = question
    company_name = extract_company_name(user_input)
    symbol = str(company_name).strip()
    company_list = data['company'].values.tolist()
    symbol_list = data['symbol1'].values.tolist()+data['symbol2'].values.tolist()

    if symbol in data['company'].values:
        symbol = data.loc[data['company'] == symbol, 'symbol2'].iloc[0]
    elif symbol in data['symbol1'].values:
        symbol = data.loc[data['company'].str.contains(symbol), 'symbol2'].iloc[0]

    
    ticker = yf.Ticker(symbol)
    try:
        ticker_info = ticker.info
        
        if 'error' in ticker_info:
            print(f"An error occurred for ticker symbol '{symbol}': {ticker_info['error']}")
        else:
            print(f"Ticker symbol '{symbol}' does not have an error in the info.")
    except HTTPError as e:
        print(f"Sorry, there is currently no data available for the company requested")
    except Exception as e:
        print(f"An error occurred: {e}")

    text = str(ticker.info)
    output = chat_query(user_input, text)

    # Store the output
    st.session_state.past.append(user_input)
    #st.session_state.generated.append(question)  # Append the question
    st.session_state.generated.append(output)

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

@st.cache()
def get_market_data():
    data=read_data()
    column_list = data['symbol2'].values.tolist()

    company_list = []
    sector_list = []
    market_cap_list = []

    for symbol in column_list:
        ticker = yf.Ticker(symbol)
        company_info = ticker.info

        market_cap = company_info.get("marketCap")
        sector = company_info.get("sector")
        company_name = company_info.get("longName")

        market_cap_list.append(market_cap)
        sector_list.append(sector)
        company_list.append(company_name)

    market_data = pd.DataFrame({
        'symbol': column_list,
        'company_name': company_list,
        'sector': sector_list,
        'market_cap': market_cap_list,
        'price_change': np.random.random(size=len(column_list))
    })
    
    return market_data


market_data=get_market_data()

with container:
    for question in questions:
        if st.button(question):
            process_question(question)
    
    st.markdown(
    """
    <style>
    .button-wrapper {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .button-wrapper button {
        background-color: black;
        color: white;
        padding: 10px 20px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# "Start a New Chat" button in the sidebar
    button_wrapper = st.empty()
    if button_wrapper.button("Start a New Chat"):
        st.session_state.past.clear()
        st.session_state.generated.clear()

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Ask a question:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        process_question(user_input)

if st.session_state['generated']:
    num_responses = len(st.session_state['generated'])
    
    for i in reversed(range(num_responses)):
        if i < len(st.session_state['generated']):
            symbol = extract_ticker_symbol(st.session_state['past'][i])
            message(st.session_state['generated'][i], key=str(i))  # Display the answer
            
            
        if i < len(st.session_state['past']):
            ticker = yf.Ticker(symbol)
            try:
                ticker_info = ticker.info
                get_graph(symbol)
            except HTTPError as e:
                print("An HTTPError occurred:", e)
                
            
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # Display the question
        
color_midpoint = np.average(market_data['price_change'], weights=market_data['market_cap'])

            # Create the treemap figure
fig = px.treemap(market_data, path=['sector', 'symbol'], values='market_cap',
                            color='price_change', hover_data=['company_name'],
                            color_continuous_scale='RdBu',
                            color_continuous_midpoint=color_midpoint)

            # Display the figure in Streamlit
st.plotly_chart(fig)