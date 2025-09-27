import pandas as pd
import streamlit as st
from io import StringIO
import json

def load_data(uploaded_file):
    """
    Reads an uploaded CSV or JSON file into a Pandas DataFrame with enhanced error handling.
    """
    if uploaded_file is None:
        return None

    try:
        file_extension = uploaded_file.name.split('.')[-1]
        string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))

        if file_extension == 'csv':
            return pd.read_csv(string_data)
        elif file_extension == 'json':
            return pd.read_json(string_data, lines=True)
        else:
            st.error("Unsupported file format. Please upload a CSV or JSON file.")
            return None
            
    except UnicodeDecodeError:
        st.error("Encoding Error: The file could not be decoded. Please ensure it is saved with UTF-8 encoding.")
        return None
    except json.JSONDecodeError:
        st.error("JSON Decode Error: The JSON file is malformed. Please check its structure.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the file: {e}")
        return None