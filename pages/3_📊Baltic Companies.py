import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from langchain.llms import OpenAI
from datetime import datetime
import yfinance as yf
import plotly.graph_objects as go
from requests.exceptions import HTTPError
from streamlit_chat import message


def read_data():
    data = pd.read_csv('data/company_ticker.csv')
    return data

def chat_query(prompt_prefix, text):
        # Generate a response
    prompt =  prompt_prefix + ': ' + text
    return llm(prompt) 

def get_graph(symbol):
    today = datetime.now()
    formatted_today = today.strftime('%Y-%m-%d')
    data = yf.download(symbol, interval='1mo', start="2023-01-01", end=formatted_today)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], fill='tozeroy', name='Adj Close'))
    fig1.update_layout(title=f'{symbol} Historical Close Prices', xaxis_title='Date', yaxis_title='Price')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['Volume'], fill='tozeroy', name='Volume'))
    fig2.update_layout(title=f'{symbol} Historical Volume', xaxis_title='Date', yaxis_title='Volume')

    # Display charts using Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)


def extract_company_name(input_text):
    prompt = 'Extract company name from this text: ' + input_text
    return llm(prompt)


def process_question(question):
    user_input = question
    company_name = extract_company_name(user_input).strip()
    company_list = data['company'].values.tolist()

    if company_name in data['symbol1'].values:
        df_company = data[data['symbol1'] == company_name]
        symbol = str(df_company['symbol2'].iloc[0])
    elif company_name in company_list:
        df_company = data[data['company'] == company_name]
        symbol = str(df_company['symbol2'].iloc[0])
    else:
        symbol = company_name

    ticker = yf.Ticker(symbol)
    try:
        ticker_info = ticker.info

        if 'error' in ticker_info:
            st.error(f"An error occurred for ticker symbol '{symbol}': {ticker_info['error']}")
        else:
            st.info(f"Ticker symbol '{symbol}' does not have an error in the info.")
    except HTTPError as e:
        st.error("Sorry, there is currently no data available for the requested company.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    text = str(ticker.info)
    output = chat_query(user_input, text)

    # Store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


# Initialize Streamlit
st.sidebar.title("Companies you can currently ask Finbay AI about.")
data = read_data()
choose = option_menu("Choose a company", data['company'].values.tolist())

df_company = data[data['company'] == choose]
symbol = str(df_company['symbol2'].iloc[0])

# Move get_graph(symbol) outside the sidebar
get_graph(symbol)

# Initialize the OpenAI language model
llm = OpenAI(temperature=0)

# Initialize session state variables
if 'past' not in st.session_state:
    st.session_state.past = []
if 'generated' not in st.session_state:
    st.session_state.generated = []

# Process predefined questions
questions = [
    "What is the market cap of " + choose + "?",
    "What is the forward PE of " + choose + "?",
    "Who is the CEO of " + choose + "?",
    "What is the profit margin of " + choose + "?",
    "What is the dividend rate and yield for " + choose + "?",
    "How has the stock price of " + choose + " performed over the past year?"
]

num_columns = 3
num_questions = len(questions)
num_rows = (num_questions + num_columns - 1) // num_columns
columns = st.columns(num_columns)

for i in range(num_questions):
    col_index = i % num_columns
    row_index = i // num_columns

    with columns[col_index]:
        if st.button(questions[i]):
            process_question(questions[i])

# Process user input
container = st.container()
with container:
    if st.sidebar.button("Start a New Chat"):
        st.session_state.past.clear()
        st.session_state.generated.clear()

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Ask a question:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        process_question(user_input)

# Display generated responses and corresponding questions
if st.session_state.generated:
    num_responses = len(st.session_state.generated)

    for i in reversed(range(num_responses)):
        if i < len(st.session_state.generated):
            symbol = extract_company_name(st.session_state.past[i])
            message(st.session_state.generated[i].strip(), key=str(i))  # Display the answer without leading/trailing whitespace

            if i < len(st.session_state.past):
                ticker = yf.Ticker(symbol)
                try:
                    ticker_info = ticker.info
                    get_graph(symbol)
                except HTTPError as e:
                    st.error("An HTTPError occurred:", e)

                message(st.session_state.past[i], is_user=True, key=str(i) + '_user')  # Display the question
