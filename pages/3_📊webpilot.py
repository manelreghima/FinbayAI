from langchain.llms import OpenAI
import os
import streamlit as st
import pandas as pd

llm = OpenAI(temperature=0)

@st.cache
def webpilot(input_url):
    # Generate a response
    prompt = 'Using WebPilot, give me the historical revenue in euro from this page as a json: ' + input_url
    response = llm.generate([prompt])
    return response[0]  # Assuming the generate function returns a list

def graph(df):
    # Generate a response
    prompt = 'Based on this dataframe, plot a bar plot using daigr.am: ' + df.to_json()
    figure = llm.generate([prompt])
    return figure[0]  # Assuming the generate function returns a list

# Page title
st.title("WebPilot Revenue Viewer")

# Input URL
input_url = 'https://finance.yahoo.com/quote/LHV1T.TL/financials?p=LHV1T.TL'

# Retrieve historical revenue
response_json = webpilot(input_url)
df = pd.read_json(response_json)

# Display DataFrame
st.write("Historical Revenue Data:")
st.dataframe(df)

# Generate and display graph
fig = graph(df)
st.plotly_chart(fig, use_container_width=True)
