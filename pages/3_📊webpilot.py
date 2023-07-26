from langchain.llms import OpenAI
import streamlit as st
import pandas as pd
import plotly.express as px

llm = OpenAI(temperature=0)

@st.cache_data()
def webpilot(input):
    # Generate a response
    prompt = 'Using WebPilot, give me the historical revenue in euro from this page as a json, also include a column with the year' + input
    return llm(prompt)

def graph(input):
    # Generate a response
    prompt = 'based on this df plot a bar plot using daigr.am' + input
    return llm(prompt)



def Cash_flow_statement(input_url):
    # Generate a response
    prompt = 'Using WebPilot, give me each of these items feature on the table at this url: ' + input_url
    response = llm(prompt)  # Assuming llm generates the response using the given prompt

    # Save the response to a text file
    output_file = "cash_flow_output.txt"
    with open(output_file, 'w') as file:
        file.write(response)

    # Print the response to the console
    print(response)

    # Return the response (optional, you can remove this line if not needed)
    return response



# Input URL
input_url = 'https://finance.yahoo.com/quote/LHV1T.TL/financials?p=LHV1T.TL'

# Retrieve historical revenue
response_json = webpilot(input_url)
response_json_balance = webpilot(input_url)


df = pd.read_json(response_json)
df_balance = pd.read_json(response_json_balance)
df['Year'] = df['Year'].astype(int)

# Display DataFrame
st.title("Income Statement:")
st.dataframe(df)
fig = px.bar(df, x='Revenue in Euro', y='Year', title='Historical Revenue of AS LHV Group')
# Display the plot in Streamlit
st.plotly_chart(fig)

st.title("Balance sheet:")
st.dataframe(df_balance)
fig = px.bar(df_balance, x='Revenue in Euro', y='Year', title='Historical Revenue of AS LHV Group')
st.plotly_chart(fig)

st.title("Cash flow statement:")
output_response = Cash_flow_statement(input_url)

st.write(output_response)

