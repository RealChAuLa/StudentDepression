import streamlit as st
import pandas as pd
import numpy as np

st.title('My first app')

st.write("Here's our first attempt at using data to create a table:")
#st.write(pd.DataFrame(pd.read_csv('Student Depression Dataset.csv')))
st.write("Here's our first attempt at using data to create a table:")

df = pd.read_csv('Student Depression Dataset.csv')

st.write(df.head())

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

st.write("Here's our first attempt at using data to create a table:")