from langchain.llms import OpenAI
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


llm = OpenAI(temperature=0)

@st.cache_data()
def webpilot(input):
    # Generate a response
    prompt = 'Using WebPilot, give me the historical revenue in euro from this page as a json, also include a column with the year' + input
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

def create_daigram_bar_plot(df, title):
    # Prepare data for daigram
    data = {
        "version": "1.0",
        "model": {
            "nestingLevels": [
                {
                    "name": "Year",
                    "labels": df['Year'].tolist(),
                    "aggregator": "sum"
                }
            ],
            "data": df['Revenue in Euro'].tolist()
        },
        "view": {
            "title": title,
            "type": "Bar Chart",
            "scale": "linear",
            "notes": ""
        }
    }

    # Generate a response
    prompt = 'Create a bar plot using daigram' + str(data)
    return llm(prompt)  # Assuming llm generates the response using the given prompt

def create_plotly_bar_plot(df, title):
    # Create a bar plot
    fig = go.Figure(data=[
        go.Bar(name='Revenue in Euro', x=df['Year'], y=df['Revenue in Euro'])
    ])

    # Change the bar mode
    fig.update_layout(barmode='group', title=title)

    return fig

# Create bar plots
income_statement_plot = create_plotly_bar_plot(df, 'Historical Revenue of AS LHV Group')
balance_sheet_plot = create_plotly_bar_plot(df_balance, 'Historical Revenue of AS LHV Group')

# Display the plots in Streamlit
st.title("Income Statement:")
st.dataframe(df)
st.plotly_chart(income_statement_plot)

st.title("Balance sheet:")
st.dataframe(df_balance)
st.plotly_chart(balance_sheet_plot)
