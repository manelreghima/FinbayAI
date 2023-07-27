from langchain.llms import OpenAI
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def generate_plots_from_url(input_url):
    llm = OpenAI(temperature=0)

    @st.cache_data()
    def webpilot(input):
        # Generate a response
        prompt = 'Using WebPilot, give me the historical revenue in euro from this page as a json, also include a column with the year ' + input
        return llm(prompt)

    # Retrieve historical revenue
    response_json = webpilot(input_url)
    response_json_balance = webpilot(input_url)

    df = pd.read_json(response_json)
    df_balance = pd.read_json(response_json_balance)
    df = df.sort_values(by='Year', ascending=True)
    df_balance = df_balance.sort_values(by='Year', ascending=True)

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
    st.plotly_chart(income_statement_plot)

    st.title("Balance sheet:")
    st.plotly_chart(balance_sheet_plot)

# Input URL
input_url = 'https://finance.yahoo.com/quote/LHV1T.TL/financials?p=LHV1T.TL'

# Generate and display the plots using the function
generate_plots_from_url(input_url)
