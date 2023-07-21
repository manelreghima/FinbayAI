from langchain.llms import OpenAI
import os
import streamlit as st
import pandas as pd
import plotly.graph_objs as go  # Import Plotly graph objects

llm = OpenAI(temperature=0)

@st.cache
def webpilot(input):
    # Generate a response
    prompt = 'Using WebPilot, give me the historical revenue in euro from this page as a json' + input
    return llm(prompt)

def graph(input):
    # Generate a response
    prompt = 'based on this json data plot a bar plot using daigr.am' + input
    response_json = llm(prompt)
    
    # Convert the JSON response to a dictionary
    data_dict = json.loads(response_json)
    
    # Extract the data for the bar plot
    x_values = list(data_dict.keys())
    y_values = list(data_dict.values())
    
    # Create a bar plot using Plotly
    fig = go.Figure(data=[go.Bar(x=x_values, y=y_values)])
    fig.update_layout(title="Historical Revenue Bar Plot")
    
    return fig

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

# Display Plotly chart
fig = graph(response_json)
st.plotly_chart(fig, use_container_width=True)
