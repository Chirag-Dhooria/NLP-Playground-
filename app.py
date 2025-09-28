import streamlit as st
from utils.data_loader import load_data
from utils.preprocessing import preprocess_text
from utils.visualizer import plot_label_distribution, generate_wordcloud, plot_confusion_matrix
from utils.model_handler import get_model, train_model, evaluate_model

# --- Page Config ---
st.set_page_config(page_title="NLP Playground", page_icon="🚀", layout="wide")

# --- State Management ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# --- Header ---
st.title("NLP Playground 🚀")
st.markdown("A no-code platform to preprocess data, train, and evaluate NLP models.")

# --- Sidebar UI ---
with st.sidebar:
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.success("File uploaded successfully!")
            # Clear previous results if a new file is uploaded
            st.session_state.processed_df = None
            st.session_state.model_results = None

    if st.session_state.df is not None:
        df = st.session_state.df
        st.header("2. Column Selection")
        text_column = st.selectbox("Select Text Column", df.columns)
        target_column = st.selectbox("Select Target/Label Column", df.columns)

        st.header("3. Preprocessing")
        options = {
            'lowercase': st.checkbox("Convert to Lowercase", value=True),
            'remove_punctuation': st.checkbox("Remove Punctuation", value=True),
            'remove_stopwords': st.checkbox("Remove Stopwords")
        }

        st.header("4. Model Selection")
        model_name = st.selectbox(
            "Choose a Model",
            ["Logistic Regression", "Naive Bayes", "Support Vector Machine (SVM)"]
        )
        
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

        if st.button("🚀 Run Experiment", use_container_width=True):
            with st.spinner("Running experiment... This may take a moment."):
                # 1. Preprocess Data
                processed_df = preprocess_text(df.copy(), text_column, options)
                st.session_state.processed_df = processed_df
                
                # 2. Get Model
                model = get_model(model_name)
                
                # 3. Train Model
                trained_model, vectorizer, X_test_vec, y_test = train_model(
                    processed_df, 'processed_text', target_column, model, test_size
                )
                
                # 4. Evaluate Model
                metrics, y_pred = evaluate_model(trained_model, X_test_vec, y_test)
                
                # 5. Store results for display
                st.session_state.model_results = {
                    'metrics': metrics,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
            st.success("Experiment finished successfully!")

# --- Main Panel ---
if st.session_state.df is None:
    st.info("Please upload a dataset to begin.")
else:
    # Define tabs
    tab_titles = ["Data Preview & EDA"]
    if st.session_state.model_results:
        tab_titles.append("Model Performance")

    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.header("Original Data Preview")
        st.dataframe(st.session_state.df.head())
        
        if st.session_state.processed_df is not None:
            st.header("Processed Data Preview")
            st.dataframe(st.session_state.processed_df.head())

        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Label Distribution")
            fig_dist = plot_label_distribution(st.session_state.df, target_column)
            st.pyplot(fig_dist)
        with col2:
            st.subheader("Word Cloud (from processed text)")
            if st.session_state.processed_df is not None:
                fig_wc = generate_wordcloud(st.session_state.processed_df, 'processed_text')
                st.pyplot(fig_wc)

    if st.session_state.model_results:
        with tabs[1]:
            results = st.session_state.model_results
            st.header("Performance Metrics")
            
            # Display metrics in columns
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Accuracy Score", f"{results['metrics']['accuracy']:.4f}")
            metric_col2.metric("F1-Score (Weighted)", f"{results['metrics']['f1_score']:.4f}")

            st.header("Confusion Matrix")
            fig_cm = plot_confusion_matrix(results['y_test'], results['y_pred'])
            st.pyplot(fig_cm)