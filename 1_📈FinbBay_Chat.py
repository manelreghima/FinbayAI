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
from streamlit_option_menu import option_menu
from PIL import Image
import base64

st.set_page_config(
    page_title="Finbay AI",
    page_icon="data/finbay-logo.jpg",
)

def add_logo():
    # Read the image file
    with open('data/finbay-logo2.png', 'rb') as f:
        image_data = f.read()
    
    # Encode the image data as base64
    encoded_image = base64.b64encode(image_data).decode()
    
    # Create the CSS style with the encoded image as the background
    css_style = f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url(data:image/png;base64,{encoded_image});
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 70px 20px;
                background-size: 180px;
            }}
            [data-testid="stSidebarNav"]::before {{
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }}
        </style>
    """

    # Apply the CSS style
    st.markdown(css_style, unsafe_allow_html=True)
add_logo()



# Load environment variables from .env file
load_dotenv()
# Access the API key
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

@st.cache_data()
def read_data():
    data = pd.read_csv('data/company_ticker.csv')
    return data
llm = OpenAI(temperature=0)


# Define the language mapping dictionary
language_mapping = {
    'English': 'en',
    'Estonian':'et'
    # Add more languages as needed
}

# Let the user select a language
selected_language = st.sidebar.selectbox('Select a language', list(language_mapping.keys()))

# Get the language code based on the selected language
language_code = language_mapping[selected_language]
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d")
formatted_time = now.strftime("%H:%M")
if language_code=='en':

    def extract_symbol(input):
            # Generate a response
        prompt =  'Extract ticker symbol from this text:' + input
        return llm(prompt)  
    
    def webpilot(input):
            # Generate a response
        prompt =  'Using WebPilot, give me the historical revenue from this page:' + input
        return llm(prompt)
    
    def chat_query(prompt_prefix, text):
            # Generate a response
        prompt =  prompt_prefix + ': ' + text
        return llm(prompt) 

    def extract_company_name(input):
            # Generate a response
        prompt =  'Extract company name from this text:' + input
        return llm(prompt) 

    
    st.markdown(
    f"<h1 style='text-align: center;'>Finbay AI Chat: Making Stock Analysis Easy</h1>",
    unsafe_allow_html=True
)

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    
    def process_question(question):
        data = read_data()
        user_input = question
        
        gratitude_keywords = ["thank you", "thanks", "grateful"]
        if any(keyword in user_input.lower() for keyword in gratitude_keywords):
            output="You're welcome! I'm glad I could help. If you have any more questions or need further assistance, feel free to ask."
        else:
            company_name = extract_company_name(user_input)
            company_name = str(company_name).strip()
            company_list = data['company'].values.tolist()
            if company_name in data['symbol1'].values:
                df_company = data[data['symbol1'] == company_name]
                symbol = str(df_company['symbol2'].iloc[0])
            elif company_name in company_list:
                df_company = data[data['company'] == company_name]
                symbol = str(df_company['symbol2'].iloc[0])
            else:
                symbol = company_name

            if symbol.lower() == "finbay":
                output = "Finbay is a stock analysis platform designed for investors. The objective of Finbay is to assist investors in comprehending complex financial data. However, it's important to note that Finbay is not a financial advisor and does not provide recommendations regarding the purchase, sale, or transfer of any stocks."
            else:
                ticker = yf.Ticker(symbol)
                try:
                    ticker_info = ticker.info

                    if 'error' in ticker_info:
                        print(f"An error occurred for ticker symbol '{symbol}': {ticker_info['error']}")
                    else:
                        print(f"Ticker symbol '{symbol}' does not have an error in the info.")
                except yf.utils.exceptions.HTTPError as e:
                    print(f"Sorry, there is currently no data available for the company requested")
                except Exception as e:
                    print(f"An error occurred: {e}")

                text = str(ticker_info)
                output = chat_query(user_input, text)

            # Assuming 'st.session_state.past' and 'st.session_state.generated'
            # lists are already initialized
        st.session_state.past.append(user_input)
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

    @st.cache_data()
    def get_market_data():
        data=read_data()
        column_list = data['symbol2'].values.tolist()
        company_list = []
        sector_list = []
        market_cap_list = []
        stock_price_list=[]

        for symbol in column_list:
            ticker = yf.Ticker(symbol)
            company_info = ticker.info
            
            market_cap = company_info.get("marketCap")
            sector = company_info.get("sector")
            company_name = company_info.get("longName")
            stock_price = company_info.get("currentPrice")

            
            market_cap_list.append(market_cap)
            sector_list.append(sector)
            company_list.append(company_name)
            stock_price_list.append(stock_price)

        market_data = pd.DataFrame({
            'symbol': column_list,
            'company_name': company_list,
            'sector': sector_list,
            'stock_price':stock_price_list,
            'market_cap': market_cap_list
        })
        
        return market_data

    market_data=get_market_data()

    questions = [
        "What is the market cap of DGR1R.RG?",
        "What is the market capitalization of NCN1T?",
        "Who is the CEO of HPR1T.TL?",
        "What is the profit margin of TSLA?",
        "What is the dividend rate and yield for GRG1L?",
        "How has the stock price of AS Merko Ehitus performed over the past year?",
        "What is the return on assets for Hepsor AS?",
        "What are the gross profits of PZV1L?",
        "What is the total debt of TEL1L.VS?"

    ]


    # Create columns dynamically
    num_columns = 3
    num_questions = len(questions)
    num_rows = (num_questions + num_columns - 1) // num_columns
    columns = st.columns(num_columns)
    st.cache()
    # Add small_logo_images and buttons to the columns
    for i in range(num_questions):
        col_index = i % num_columns
        row_index = i // num_columns

        with columns[col_index]:
            
        
            if columns[col_index].button(questions[i]):
                    process_question(questions[i])
    st.markdown(f"(This data is from {formatted_time} {formatted_date}) ")        
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

    if st.session_state['generated']:
        num_responses = len(st.session_state['generated'])
        
        for i in reversed(range(num_responses)):
            if i < len(st.session_state['generated']):
                symbol = extract_company_name(st.session_state['past'][i])
                
                answer = st.session_state['generated'][i].strip()
                message(answer, key=str(i))

                if i < len(st.session_state['past']):
                    ticker = yf.Ticker(symbol)
                    try:
                        ticker_info = ticker.info
                        get_graph(symbol)
                    except HTTPError as e:
                        print("An HTTPError occurred:", e)
                    
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # Display the question
    
    st.markdown("<h3>TOP 3 by change</h3>", unsafe_allow_html=True)
    # Load the ticker data from the provided URL
    @st.cache_data()
    def load_ticker_data():
        ticker = pd.read_csv('https://huggingface.co/datasets/manelreghima/company_ticker/raw/main/company_ticker.csv')
        return ticker

    # Get the symbol list from the ticker data
    ticker_data = load_ticker_data()
    symbol_list = ticker_data['symbol2'].tolist()


    # Convert the dates to string format
    today_str = now.strftime("%Y-%m-%d")

    # Create an empty dataframe
    df_empty = pd.DataFrame()

    # Iterate over each symbol
    for symbol in symbol_list:
        # Download data with updated start and end dates
        data = yf.download(symbol, interval='1d', start="2023-06-27", end=today_str)
        data['ticker'] = symbol
        df_sorted = data.sort_values(by='Date', ascending=False)
        df_head = df_sorted.head(2)

        # Check if df_head has at least two rows
        if len(df_head) >= 2:
            # Calculate the difference between the two values
            df_head['change'] = (df_head.iloc[0]['Close'] - df_head.iloc[1]['Close']) / df_head.iloc[1]['Close'] * 100
        else:
            # Handle the case when df_head has less than two rows
            df_head['change'] = None

        # Concatenate df_head with the empty dataframe
        df_empty = pd.concat([df_empty, df_head])

    # Filter the dataframe to include only today's rows
    df_change = df_empty.iloc[::2]

    # Sort the resulting dataframe by 'change' column in descending order
    df_sorted = df_change.sort_values(by='change', ascending=False)
    st.write(df_sorted.head(3))

    st.markdown("<h3>Bottom 3 by change</h3>", unsafe_allow_html=True)
    st.write(df_sorted.tail(3))



    st.markdown("<h3>TOP 3 by Volume</h3>", unsafe_allow_html=True)
    # Create an empty list to store dataframes
    dfs = []

    # Iterate over each symbol
    for symbol in symbol_list:
        # Download data with updated start and end dates
        data = yf.download(symbol, interval='1d', start="2023-06-27", end=today_str)
        data['ticker'] = symbol
        df_sorted = data.sort_values(by='Date', ascending=False)
        df_head = df_sorted.head(1)

        # Add df_head to the list
        dfs.append(df_head)

    # Concatenate all dataframes in the list
    df_empty = pd.concat(dfs)

    # Sort the resulting dataframe by 'Volume' column in descending order
    df_sorted_volume = df_empty.sort_values(by='Volume', ascending=False)

    # Display the resulting dataframe using Streamlit
    st.write(df_sorted_volume.head(3))
    # Create the treemap figure
    
    fig = px.treemap(market_data, path=['sector', 'symbol'], values='market_cap',
                 color='stock_price', hover_data=['sector', 'company_name', 'stock_price', 'market_cap'],  
                 color_continuous_scale='RdYlGn',
                 color_continuous_midpoint=np.average(market_data['stock_price'], weights=market_data['market_cap']))

    fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>%{label}<br>Stock Price: %{customdata[2]:.2f}(EUR)<br>Market Cap: %{customdata[3]:,.2f}(EUR)<br><a href="https://finance.yahoo.com/m/20cfd21d-be71-3d76-b865-75320437dc3b/more-firms-on-wall-street-are.html?.tsrc=fin-srch">More info</a>')

    st.plotly_chart(fig)


    st.markdown("---")
    webpilot('https://finance.yahoo.com/quote/LHV1T.TL/financials?p=LHV1T.TL')
    # Add the disclaimer text at the bottom
    st.markdown("**Disclaimer:**")
    st.markdown("DO YOUR OWN RESEARCH") 
    st.markdown("All content provided by Finbay Technology OÜ is for informational and educational purposes only and is not meant to represent trade or investment recommendations.")

elif language_code=='et':

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
        prompt =  'Väljavõte ettevõtte nimi sellest tekstist:' + input
        return llm(prompt) 

    #st.title("Finbay AI vestlus: Aktsiaanalüüsi lihtsaks tegemine")
    st.markdown(
    f"<h1 style='text-align: center;'>Finbay AI vestlus: Aktsiaanalüüsi lihtsaks tegemine</h1>",
    unsafe_allow_html=True
)

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def process_question(question):
        data=read_data()
        user_input = question
        company_name = extract_company_name(user_input)
        company_name = str(company_name).strip()
        company_list = data['company'].values.tolist()
    
        if company_name in data['symbol1'].values:
            df_company = data[data['symbol1']==company_name]
            symbol = str(df_company['symbol2'].iloc[0])  
        elif company_name in company_list:
            df_company = data[data['company']==company_name]
            symbol = str(df_company['symbol2'].iloc[0])     
        else:
            symbol=company_name

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
        prompt = 'Ekstrakt ticker sümbol sellest tekstist: ' + input_text
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
        fig1.update_layout(title=str(symbol)+' Ajaloolised ', xaxis_title='Kuupäev', yaxis_title='Hind')

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data['Volume'], fill='tozeroy', name='Volume'))
        fig2.update_layout(title=str(symbol)+' Ajalooline maht', xaxis_title='Kuupäev', yaxis_title='Köide')
        

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
        stock_price_list=[]

        for symbol in column_list:
            ticker = yf.Ticker(symbol)
            company_info = ticker.info
            
            market_cap = company_info.get("marketCap")
            sector = company_info.get("sector")
            company_name = company_info.get("longName")
            stock_price = company_info.get("currentPrice")

            
            market_cap_list.append(market_cap)
            sector_list.append(sector)
            company_list.append(company_name)
            stock_price_list.append(stock_price)

        market_data = pd.DataFrame({
            'symbol': column_list,
            'company_name': company_list,
            'sector': sector_list,
            'stock_price':stock_price_list,
            'market_cap': market_cap_list
        })

        
        return market_data

    market_data=get_market_data()

    questions = [
        "Milline on DGR1R.RG turukapitali suurus?",
        "Kui suur on TEL1L.VS koguvõlg?",
        "Kes on HPR1T.TL tegevjuht?",
        "Kui suur on TSLA kasumimarginaal?",
        "Milline on GRG1Li dividendimäär ja tootlus?",
        "Kuidas on AS Merko Ehitus aktsia hind viimase aasta jooksul arenenud?",
        "Milline on Hepsor ASi varade tootlus?",
        "Kui suur on PZV1Li brutokasum?",
        "Mis on NCN1T turuväärtus?"
        
    ]


    # Create columns dynamically
    num_columns = 3
    num_questions = len(questions)
    num_rows = (num_questions + num_columns - 1) // num_columns
    columns = st.columns(num_columns)

    # Add small_logo_images and buttons to the columns
    for i in range(num_questions):
        col_index = i % num_columns
        row_index = i // num_columns

        with columns[col_index]:
            
        
            if columns[col_index].button(questions[i]):
                    process_question(questions[i])
            
    container = st.container()
    with container:
        if st.sidebar.button("Alusta uut vestlust"):
            st.session_state.past.clear()
            st.session_state.generated.clear()

        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Esitage küsimus:", key='input')
            submit_button = st.form_submit_button(label='Saada')

        if submit_button and user_input:
            process_question(user_input)    

    if st.session_state['generated']:
        num_responses = len(st.session_state['generated'])
        
        for i in reversed(range(num_responses)):
            if i < len(st.session_state['generated']):
                symbol = extract_company_name(st.session_state['past'][i])
                formatted_date = now.strftime("%Y-%m-%d")
                formatted_time = now.strftime("%H:%M")
                prompt = f"(Need andmed pärinevad {formatted_time} {formatted_date}). "
                answer = prompt+ st.session_state['generated'][i].strip()
                message(answer, key=str(i))
                if i < len(st.session_state['past']):
                    ticker = yf.Ticker(symbol)
                    try:
                        ticker_info = ticker.info
                        get_graph(symbol)
                    except HTTPError as e:
                        print("An HTTPError occurred:", e)
                    
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # Display the question
    
    # Create the treemap figure
    #color_midpoint = np.average(market_data['price_change'], weights=market_data['market_cap'])           
    fig = px.treemap(market_data, path=['sector', 'symbol'], values='market_cap',
                 color='stock_price', hover_data=['sector', 'company_name', 'stock_price', 'market_cap'],
                 color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.average(market_data['stock_price'], weights=market_data['market_cap']))

# Remove the 'id', 'parent', and 'label' from the hover tooltip for companies' symbols
    fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>%{label}<br>Stock Price: %{customdata[2]:.2f}<br>Market Cap: %{customdata[3]:,.2f}')

    st.plotly_chart(fig)
    
    st.markdown("---")
    st.markdown("**Disclaimer:**")
    st.markdown("TEHKE OMAENDA UURINGUID")
    st.markdown("Kogu Finbay Technology OÜ poolt esitatud sisu on mõeldud ainult informatiivsetel ja hariduslikel eesmärkidel ning ei ole mõeldud kaubandus- või investeerimissoovituste esitamiseks.")
