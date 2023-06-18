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
                         styles={
                             "container": {"padding": "5!important", "background-color": "#1D1D1D"},
                             "nav-link": {"text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                             "nav-link-selected": {"background-color": "#262626"},
                         })
df_company = data[data['company']==choose]
symbol = str(df_company['symbol2'].iloc[0])
# Move get_graph(symbol) outside the sidebar
#get_graph(symbol)


llm = OpenAI(temperature=0)

def chat_query(prompt_prefix, text):
        # Generate a response
    prompt =  prompt_prefix + ': ' + text
    return llm(prompt) 

def extract_company_name(input):
        # Generate a response
    prompt =  'Extract company name from this text:' + input
    return llm(prompt) 

# container for text box
container = st.container()
questions=["What is the market cap of "+str(choose)+"?","What is the forward PE of "+str(choose)+"?",
           "Who is the CEO of "+str(choose)+"?","What is the profit margin of "+str(choose)+"?"]


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

with container:
    for question in questions:
        if st.button(question):
            process_question(question)
    
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
            #symbol = extract_company_name(st.session_state['past'][i])
            message(st.session_state['generated'][i], key=str(i))  # Display the answer
            
            
        if i < len(st.session_state['past']):
           
            get_graph(symbol)
            
                
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # Display the question
 