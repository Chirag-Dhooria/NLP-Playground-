import pandas as pd
import streamlit as st
from io import StringIO

def load_data(uploaded_file):
    if uploaded_file is None:
        return None

    try:
        if uploaded_file.name.endswith('.csv'):
            string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(string_data)
        elif uploaded_file.name.endswith('.json'):
            string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_json(string_data, lines=True)
        else:
            st.error("Unsupported file format. Please upload a CSV or JSON file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None