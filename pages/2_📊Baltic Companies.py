import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from langchain.llms import OpenAI
from datetime import datetime
import yfinance as yf
import plotly.graph_objects as go
from requests.exceptions import HTTPError
from streamlit_chat import message
import base64

now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d")
formatted_time = now.strftime("%H:%M")

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
def read_data():
    data = pd.read_csv('data/company_ticker.csv')
    return data

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

data=read_data()
def clear_session_state():
    if st.session_state.past or st.session_state.generated:  # Check if either 'past' or 'generated' is not empty
        st.session_state.past.clear()
        st.session_state.generated.clear()


with st.sidebar:
    choose = option_menu("Companies you can currently ask Finbay AI about.",
                         ['APB Apranga',
                        'Arco Vara AS',
                        'Auga Group AB',
                        'AS Baltika',
                        "Coop Pank AS",
                        'DelfinGroup AS',
                        'AS Ekspress Grupp',
                        'EfTEN Real Estate Fund III AS',
                        'Enefit Green AS',
                        'Grigeo AB',
                        'AS Harju Elekter',
                        'AS HansaMatrix',
                        'Hepsor AS',
                        'IPAS Indexo AS',
                        'AB Ignitis grupe',
                        'AB Klaipedos nafta',
                        'AS LHV Group',
                        'AB Linas Agro Group',
                        'AS Merko Ehitus',
                        'Nordecon AS',
                        'Novaturas AB',
                        'AS Pro Kapital Grupp',
                        'AS PRFoods',
                        'AB Panevezio Statybos Trestas',
                        'AB Pieno Zvaigzdes',
                        'Rokiskio Suris AB',
                        'AB Siauliu Bankas',
                        'SAF Tehnika A/S',
                        'AS Silvano Fashion Group',
                        'AS Tallink Grupp',
                        'Telia Lietuva, AB',
                        'Tallinna Kaubamaja Grupp AS',
                        'AS Tallinna Sadam',
                        'AS Tallinna Vesi',
                        'AB Vilkyskiu pienine'],
                         
                         default_index=0,
                         on_change=clear_session_state(),
                         styles={
                             "container": {"padding": "5!important", "background-color": "#1D1D1D"},
                             "nav-link": {"text-align": "left", "margin": "0px", "--hover-color": "#262626"},
                             "nav-link-selected": {"background-color": "#00A767"},
                         },
                         key="option_menu")
df_company = data[data['company']==choose]
symbol = str(df_company['symbol2'].iloc[0])
symbol1 = str(df_company['symbol1'].iloc[0])
image_path='data/companies_logos/'+symbol1+'.png'

#st.image('data/companies_logos/'+symbol1+'.png')

import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# URL of the image
image_url = "https://github.com/manelreghima/FinbayAI/blob/1b240dae8c8946a9a4075cfe99ae9cc2e47a2e45/data/companies_logos/APG1L.png"

# URL to redirect to when the image is clicked
redirect_url = "https://your_redirect_url.com"


# Display the image as a clickable link
st.markdown(f'<a href="{redirect_url}"> <img src="{image_url}" width="200" height="200"> </a>', unsafe_allow_html=True)



llm = OpenAI(temperature=0)

def chat_query(prompt_prefix, text):
        # Generate a response
    prompt =  prompt_prefix + ': ' + text
    return llm(prompt) 

def extract_company_name(input):
        # Generate a response
    prompt =  'Extract company name from this text:' + input
    return llm(prompt) 


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


# container for text box
questions = [
    "What is the market cap of "+str(choose)+"?",
    "What is the forward PE of "+str(choose)+"?",
    "Who is the CEO of "+str(choose)+"?",
    "What is the profit margin of "+str(choose)+"?",
    "What is the dividend rate and yield for "+str(choose)+"?",
    "How has the stock price of "+str(choose)+" performed over the past year?",
    "What is the return on assets for "+str(choose)+"?",
    "What are the gross profits of "+str(choose)+"?",
    "What is the total debt of "+str(choose)+"?"
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

st.markdown(f"(This data is from {formatted_time} {formatted_date}) ")                  
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Ask a question:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        process_question(user_input) 

df_company = data[data['company']==str(choose)]
symbol = str(df_company['symbol2'].iloc[0])
ticker = yf.Ticker(symbol)
company_info=ticker.info
company_description = company_info.get("longBusinessSummary")


def create_description(input):
    # Check if input is None
    if input is None:
        raise ValueError("Input cannot be None")

    # Generate a response
    prompt =  'Correct and improve this text: ' + input
    return llm(prompt)

if company_description is not None:
    resp = create_description(company_description)
else:
    print("company_description is None")

@st.cache_data()
def webpilot(input):
        # Generate a response
    prompt = 'Using WebPilot, give me the historical revenue in euro from this page as a json, also include a column with the year ' + input
    return llm(prompt)

@st.cache_data()
def generate_income(input_url):

    # Retrieve historical revenue
    response_json = webpilot(input_url)

    df = pd.read_json(response_json)
    
    df = df.sort_values(by='Year', ascending=True)
    

    def create_plotly_bar_plot(df, title):
        # Create a bar plot
        fig = go.Figure(data=[
            go.Bar(name='Revenue (Euro)', x=df['Year'], y=df['Revenue (Euro)'])
        ])

        # Change the bar mode
        fig.update_layout(barmode='group', title=title)

        return fig

    # Create bar plots
    income_statement_plot = create_plotly_bar_plot(df, 'Income Statement - Historical Revenue of '+str(choose))

    # Display the plots in Streamlit
    st.dataframe(df)
    st.plotly_chart(income_statement_plot)


def generate_balnace(input_url):
    #llm = OpenAI(temperature=0)
    # Retrieve historical revenue
    response_json_balance = webpilot(input_url)
    df_balance = pd.read_json(response_json_balance)
    df_balance = df_balance.sort_values(by='Year', ascending=True)

    def create_plotly_bar_plot(df, title):
        # Create a bar plot
        fig = go.Figure(data=[
            go.Bar(name='Revenue (Euro)', x=df['Year'], y=df['Revenue (Euro)'])
        ])

        # Change the bar mode
        fig.update_layout(barmode='group', title=title)

        return fig

    # Create bar plots
    balance_sheet_plot = create_plotly_bar_plot(df_balance, 'Balance sheet - Total Asset of '+str(choose))

    # Display the plots in Streamlit

    st.dataframe(df_balance)
    st.plotly_chart(balance_sheet_plot)

if st.session_state['generated']:
    num_responses = len(st.session_state['generated'])
    
    for i in reversed(range(num_responses)):
        if i < len(st.session_state['generated']):
            
            #message(st.session_state['generated'][i].strip(), key=str(i) + '_generated') # Display the answer
            
            answer = st.session_state['generated'][i].strip()
            message(answer, key=str(i) + '_answer')
            
        if i < len(st.session_state['past']):
            
            get_graph(symbol)
            input_url_income = 'https://finance.yahoo.com/quote/'+symbol+'/financials?p='+symbol
            input_url_balance='https://finance.yahoo.com/quote/'+symbol+'/balance-sheet?p='+symbol
            generate_income(input_url_income)
            generate_balnace(input_url_balance)
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # Display the question
