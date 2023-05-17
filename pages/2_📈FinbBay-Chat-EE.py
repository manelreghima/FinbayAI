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

# Load environment variables from .env file
load_dotenv()
# Access the API key
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

llm = OpenAI(temperature=0)

def extract_symbol(input):
        # Generate a response
    prompt =  'Ekstrakt ticker sümbol sellest tekstist:' + input
    return llm(prompt)  

def chat_query(prompt_prefix, text):
        # Generate a response
    prompt =  prompt_prefix + ': ' + text
    return llm(prompt) 

st.title("Tere tulemast FinbayAI")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []



# container for text box
container = st.container()
questions=["Kui suur on ettevõtte DGR1R.RG turukapitalisatsioon?", 
           "Kui suur on ettevõtte AUG1LS.VS tulevane keskmine kasumimarginaal?",
           "Kes on ettevõtte LHV1T.TL tegevjuht?", "Kui suur on ettevõtte TSLA kasumimarginaal?"]

def process_question(question):
    user_input = question
    symbol = extract_symbol(user_input)
    if symbol is not None:
        symbol = symbol.strip()
    ticker_symbol = symbol 
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
    return ticker_symbol

def get_graph(ticker_symbol):
    today = datetime.now()
    formatted_today = today.strftime('%Y-%m-%d')
    data = yf.download(ticker_symbol, interval = '1mo', start="2023-01-01", end=formatted_today)
    fig1 = px.line(data_frame=data, x=data.index, y='Adj Close', title=str(ticker_symbol)+' Ajaloolised sulgemishinnad')
    fig1.update_xaxes(title='Kuupäev')
    fig1.update_yaxes(title='Hind')

    # Create figure for "Volume"
    fig2 = px.line(data_frame=data, x=data.index, y='Volume', title=str(ticker_symbol)+' Ajalooline maht')
    fig2.update_xaxes(title='Kuupäev')
    fig2.update_yaxes(title='Köide')

    # Display charts using Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

with container:
    for question in questions:
        if st.button(question):
            process_question(question)
            
    if st.button("Alusta uut vestlust"):
        st.session_state.past.clear()
        st.session_state.generated.clear()

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Esitage küsimus:", key='input')
        submit_button = st.form_submit_button(label='Saada')

    if submit_button and user_input:
        process_question(user_input)

if st.session_state['generated']:
    num_responses = len(st.session_state['generated'])
    
    for i in range(num_responses):
        if i < len(st.session_state['past']):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # Display the question
        if i < len(st.session_state['generated']):
            message(st.session_state['generated'][i], key=str(i))  # Display the answer

             # Display the graph
            symbol = extract_ticker_symbol(st.session_state['past'][i])
            get_graph(symbol)

