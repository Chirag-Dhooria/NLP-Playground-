import streamlit as st
from utils.data_loader import load_data
from utils.preprocessing import preprocess_text
from utils.visualizer import plot_label_distribution, generate_wordcloud

st.set_page_config(
    page_title="NLP Playground",
    page_icon="📜",
    layout="wide"
)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

st.title("NLP Playground: Exploratory Data Analysis 📜")
st.markdown("Upload your text data, apply basic preprocessing, and visualize the results.")

with st.sidebar:
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader(
        "Upload a CSV or JSON file",
        type=["csv", "json"]
    )
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.success("File uploaded successfully!")

    if st.session_state.df is not None:
        st.header("2. Column Selection")
        df_columns = st.session_state.df.columns.tolist()
        text_column = st.selectbox("Select Text Column", df_columns)
        target_column = st.selectbox("Select Target/Label Column", df_columns)

        st.header("3. Preprocessing")
        options = {
            'lowercase': st.checkbox("Convert to Lowercase", value=True),
            'remove_punctuation': st.checkbox("Remove Punctuation", value=True),
            'remove_stopwords': st.checkbox("Remove Stopwords")
        }

        if st.button("Process Data", use_container_width=True):
            with st.spinner("Processing..."):
                st.session_state.processed_df = preprocess_text(
                    st.session_state.df.copy(), text_column, options
                )
            st.success("Data processed successfully!")

if st.session_state.processed_df is None:
    st.info("Upload data and process it to see the results here.")
else:
    processed_df = st.session_state.processed_df
    original_df = st.session_state.df

    tab1, tab2, tab3 = st.tabs(["Data Preview", "Label Distribution", "Word Cloud"])

    with tab1:
        st.header("Original Data Preview")
        st.dataframe(original_df.head())
        st.header("Processed Data Preview")
        st.dataframe(processed_df.head())

    with tab2:
        st.header("Distribution of Labels")
        fig = plot_label_distribution(processed_df, target_column)
        st.pyplot(fig)

    with tab3:
        st.header("Most Common Words")
        fig = generate_wordcloud(processed_df, 'processed_text')
        st.pyplot(fig)