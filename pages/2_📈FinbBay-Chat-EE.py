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

st.set_page_config(
    page_title="Finbay AI",
    page_icon="data/finbay-logo.jpg",
)
# Load environment variables from .env file
load_dotenv()
# Access the API key
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]
data = pd.read_csv('data/company_ticker.csv')
llm = OpenAI(temperature=0)

def extract_symbol(input):
        # Generate a response
    prompt =  'Ekstrakt ticker sümbol sellest tekstist:' + input
    return llm(prompt)  

def chat_query(prompt_prefix, text):
        # Generate a response
    prompt =  prompt_prefix + ': ' + text
    return llm(prompt) 

def extract_company_name(input):
        # Generate a response
    prompt =  'Väljavõte ettevõtte nimest sellest tekstist:' + input
    return llm(prompt) 

st.title("Tere tulemast FinbayAIsse")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


# container for text box
container = st.container()
questions=["Milline on DGR1R.RG turukapitalisatsioon?", "Milline on AUG1LS.VS tulevane keskmine rahaproduktiväärtus?",
           "Kes on ettevõtte LHV1T.TL tegevjuht?", "Milline on ettevõtte TSLA kasumimarginaal?", "Milline on ettevõtte TSLA kasumimarginaal? "]

def process_question(question):
    user_input = question
    company_name = extract_company_name(user_input)
    symbol = str(company_name).strip()

    if symbol not in data['symbol1'].values and symbol not in data['symbol2'].values:
        df_company = data[data['company'].str.contains(symbol)]
        symbol = str(df_company['symbol1'].iloc[0])
    
    #if symbol is not None:
    #symbol = symbol.strip()
    
    ticker = yf.Ticker(symbol)    
    text = str(ticker.info)
    output = chat_query(user_input, text)
    # Store the output
    st.session_state.past.append(user_input)
    #st.session_state.generated.append(question)  # Append the question
    st.session_state.generated.append(output)

def extract_ticker_symbol(input_text):
    prompt = 'Ekstrakt ticker sümbol sellest tekstist: ' + input_text
    response = llm(prompt)
    ticker_symbol = response.split(": ")[-1].strip()  # Extract the ticker symbol from the response
    for index, row in data.iterrows():
    # Check if the symbol is found in symbol2
        if ticker_symbol in row['symbol2']:
        # Update the value symbol to the corresponding value in 'symbol1'
            ticker_symbol=row['symbol1']
    return ticker_symbol

def get_graph(symbol):
    today = datetime.now()
    formatted_today = today.strftime('%Y-%m-%d')
    data = yf.download(symbol, interval = '1mo', start="2023-01-01", end=formatted_today)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], fill='tozeroy', name='Adj Close'))
    fig1.update_layout(title=str(symbol)+' Ajaloolised sulgemishinnad', xaxis_title='Kuupäev', yaxis_title='Hind')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['Volume'], fill='tozeroy', name='Volume'))
    fig2.update_layout(title=str(symbol)+' Ajalooline maht', xaxis_title='Kuupäev', yaxis_title='Köide')
    

    # Display charts using Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)


def get_market_data():
    
    column_list = data['symbol1'].values.tolist()

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
    
    if st.button("Alusta uut vest"):
        st.session_state.past.clear()
        st.session_state.generated.clear()

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Esita küsimus:", key='input')
        submit_button = st.form_submit_button(label='Saada')

    if submit_button and user_input:
        process_question(user_input)

if st.session_state['generated']:
    num_responses = len(st.session_state['generated'])
    
    for i in reversed(range(num_responses)):
        if i < len(st.session_state['generated']):
            symbol = extract_ticker_symbol(st.session_state['past'][i])
            message(st.session_state['generated'][i], key=str(i))  # Display the answer
            
            
        if i < len(st.session_state['past']):
            get_graph(symbol)
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # Display the question
        
color_midpoint = np.average(market_data['price_change'], weights=market_data['market_cap'])

            # Create the treemap figure
fig = px.treemap(market_data, path=['sector', 'symbol'], values='market_cap',
                            color='price_change', hover_data=['company_name'],
                            color_continuous_scale='RdBu',
                            color_continuous_midpoint=color_midpoint)

            # Display the figure in Streamlit
st.plotly_chart(fig)