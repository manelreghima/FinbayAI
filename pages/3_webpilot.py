import openai
import pandas as pd
import re
import os
import streamlit as st

from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

def webpilot(input):
    # Generate a response
    prompt = 'Using WebPilot, give me the historical revenue from this page:' + input
    return llm(prompt)

response_json = webpilot('https://finance.yahoo.com/quote/LHV1T.TL/financials?p=LHV1T.TL')
df = pd.DataFrame(response_json)
st.write(df)