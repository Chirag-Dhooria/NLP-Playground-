import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle

from utils.data_loader   import load_data
from utils.preprocessing import preprocess_text
from utils.visualizer    import (
    plot_label_distribution, generate_wordcloud,
    plot_confusion_matrix,   plot_top_ngrams,
    plot_model_comparison,   plot_feature_importance,
    plot_learning_curve,
    plot_lime_explanation,   plot_lime_html,
    plot_shap_summary,       plot_shap_waterfall,  plot_shap_force,
)
from utils.model_handler import run_sklearn_pipeline, run_hf_inference, get_sklearn_model
from utils.explainer     import run_lime, run_shap
from utils.consultant    import render_consultant_tab, render_consultant_sidebar_widget

st.set_page_config(page_title="NLP Playground", page_icon="🛝", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] h2 { font-size:.85rem;text-transform:uppercase;letter-spacing:.08em;color:#a0aec0;margin-bottom:4px; }
[data-testid="stTabs"] button { font-size:.9rem;font-weight:600; }
[data-testid="stMetric"] { background:#f7fafc;border-left:4px solid #667eea;padding:10px 16px;border-radius:6px; }
</style>""", unsafe_allow_html=True)

for k, v in [('df',None),('model_results',[]),('shap_result',None),('raw_texts',None),('class_names',None)]:
    if k not in st.session_state: st.session_state[k] = v

def generate_automated_report(result, df):
    r  = "# NLP Playground — Experiment Report\n\n"
    r += f"## Dataset Summary\n- Total Rows: {len(df)}\n\n"
    r += f"## Model Details\n- Task: {result.get('task')}\n- Model: {result['model_name']}\n"
    if 'accuracy' in result:
        r += f"\n## Performance\n- Accuracy: {result['accuracy']:.4f}\n- F1: {result['f1_score']:.4f}\n"
    r += "\n## Automated Insights\n"
    if result.get('type') == 'sklearn':
        acc, f1 = result.get('accuracy',0), result.get('f1_score',0)
        r += ("- ✅ Excellent accuracy (≥90%).\n" if acc>=0.90 else
              "- 🟡 Good accuracy (75–90%).\n"   if acc>=0.75 else
              "- 🔴 Low accuracy (<75%).\n")
        if abs(acc-f1)>0.05: r += "- ⚠️ Class imbalance suspected.\n"
    return r

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Task Selection")
    nlp_task = st.selectbox("Select NLP Task",
        ["Text Classification","Text Summarization","Question Answering","Sentiment Analysis"])

    st.header("2. Data Upload")
    uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv","json"])
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None: st.success("✅ Data Loaded")

    if st.session_state.df is not None:
        st.header("3. Mapping & Preprocessing")
        if nlp_task == "Question Answering":
            ctx = st.selectbox("Context Column", st.session_state.df.columns)
            qst = st.selectbox("Question Column", st.session_state.df.columns)
            text_cols = [ctx, qst]
        else:
            text_cols  = st.multiselect("Input Columns", st.session_state.df.columns,
                                         default=[st.session_state.df.columns[0]])
            target_col = st.selectbox("Label Column", [None]+list(st.session_state.df.columns))

        options = {
            'lowercase':          st.checkbox("Lowercase",   True),
            'remove_punctuation': st.checkbox("Punctuation", True),
            'remove_stopwords':   st.checkbox("Stopwords"),
            'lemmatization':      st.checkbox("Lemmatize"),
            'stemming':           st.checkbox("Stemming"),
        }

        st.header("4. Model Selection")
        source = st.radio("Source", ["Scikit-Learn","Transformers"]) \
            if nlp_task == "Text Classification" else "Transformers"

        if source == "Scikit-Learn":
            m_name = st.selectbox("Model",
                ["Logistic Regression","Naive Bayes","Support Vector Machine (SVM)",
                 "Random Forest","Gradient Boosting"])
            params    = {'C': st.slider("C (Regularisation)", 0.1, 10.0, 1.0)} if "Logistic" in m_name else {}
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        else:
            presets = {
                "Text Classification": ["distilbert-base-uncased-finetuned-sst-2-english"],
                "Text Summarization":  ["facebook/bart-large-cnn","google/pegasus-xsum"],
                "Question Answering":  ["deepset/roberta-base-squad2"],
                "Sentiment Analysis":  ["cardiffnlp/twitter-roberta-base-sentiment-latest"],
            }
            m_name = st.selectbox("HuggingFace Presets", presets[nlp_task])

        st.divider()
        if st.button("🚀 Run Experiment", use_container_width=True, type="primary"):
            with st.status("Executing experiment…") as status:
                try:
                    is_qa   = nlp_task == "Question Answering"
                    proc_df = preprocess_text(st.session_state.df.copy(), text_cols, options,
                                              is_qa, ctx if is_qa else None, qst if is_qa else None)
                    if source == "Scikit-Learn":
                        model_obj,vec,met,X_tr,y_tr,y_te,y_pred = run_sklearn_pipeline(
                            proc_df,'processed_text',target_col,m_name,params,test_size)
                        st.session_state.raw_texts   = proc_df['processed_text'].tolist()
                        st.session_state.class_names = [str(c) for c in sorted(proc_df[target_col].unique())]
                        st.session_state.shap_result = None
                        st.session_state.model_results.append({
                            'type':'sklearn','task':nlp_task,'model_name':m_name,
                            'accuracy':met['accuracy'],'f1_score':met['f1_score'],
                            'y_test':y_te,'y_pred':y_pred,'trained_model':model_obj,
                            'vectorizer':vec,'X_train_vec':X_tr,'y_train':y_tr,
                        })
                    else:
                        results = run_hf_inference(proc_df, nlp_task.lower().replace(" ","-"), m_name,
                            'processed_text' if not is_qa else None,
                            ctx if is_qa else None, qst if is_qa else None)
                        st.session_state.model_results.append(
                            {'type':'hf','task':nlp_task,'model_name':m_name,'results':results})
                    status.update(label="✅ Complete!", state="complete")
                except Exception as e:
                    status.update(label="❌ Failed", state="error"); st.error(f"Error: {e}")

    render_consultant_sidebar_widget()

# ── Main panel ─────────────────────────────────────────────────────────────────
st.title("NLP Playground 🛝")
st.markdown("A no-code platform to explore text data and evaluate NLP models.")

if st.session_state.df is not None:
    t_eda, t_res, t_xai, t_comp, t_consultant = st.tabs(
        ["📊 EDA","📈 Results","🔍 Explainability","⚖️ Comparison","💬 AI Consultant"])

    # EDA ─────────────────────────────────────────────────────────────────────
    with t_eda:
        st.subheader("Exploratory Data Analysis")
        with st.expander("📋 Dataset Preview", expanded=False):
            st.dataframe(st.session_state.df.head(50), use_container_width=True)
            r,c,n = st.columns(3)
            r.metric("Rows",f"{len(st.session_state.df):,}")
            c.metric("Columns",len(st.session_state.df.columns))
            n.metric("Total Nulls",int(st.session_state.df.isnull().sum().sum()))
        st.divider()
        if nlp_task != "Question Answering" and text_cols:
            c1,c2 = st.columns(2)
            with c1: st.markdown("**Word Cloud**"); st.pyplot(generate_wordcloud(st.session_state.df,text_cols[0]))
            with c2: st.markdown("**Top Bi-grams**"); st.pyplot(plot_top_ngrams(st.session_state.df,text_cols[0]))
            if 'target_col' in dir() and target_col:
                st.markdown("**Label Distribution**"); st.pyplot(plot_label_distribution(st.session_state.df,target_col))

    # Results ─────────────────────────────────────────────────────────────────
    with t_res:
        if not st.session_state.model_results:
            st.info("No experiments run yet.")
        else:
            latest = st.session_state.model_results[-1]
            st.subheader(f"Latest: {latest['model_name']}")
            if latest['type'] == 'sklearn':
                m1,m2,m3 = st.columns(3)
                m1.metric("Accuracy",f"{latest['accuracy']:.4f}")
                m2.metric("F1 Score (weighted)",f"{latest['f1_score']:.4f}")
                m3.metric("Task",latest['task'])
                acc = latest['accuracy']
                if acc>=0.90:   st.success("✅ **Excellent accuracy** (≥90%).")
                elif acc>=0.75: st.warning("🟡 **Good accuracy** (75–90%).")
                else:           st.error("🔴 **Low accuracy** (<75%).")
                if abs(latest['accuracy']-latest['f1_score'])>0.05:
                    st.warning("⚠️ Accuracy and F1 differ — possible class imbalance.")
                st.divider()
                v1,v2 = st.columns(2)
                with v1:
                    st.markdown("**Confusion Matrix**")
                    st.pyplot(plot_confusion_matrix(latest['y_test'],latest['y_pred']))
                with v2:
                    st.markdown("**Learning Curve**")
                    st.pyplot(plot_learning_curve(get_sklearn_model(latest['model_name']),
                                                  latest['X_train_vec'],latest['y_train']))
                st.markdown("**Feature Significance (Top 20 TF-IDF Features)**")
                fi = plot_feature_importance(latest['trained_model'],latest['vectorizer'])
                if fi: st.pyplot(fi)
                else:  st.info("Feature importance not available for this model type.")

                # Quick SHAP summary expander
                with st.expander("🔍 Quick SHAP Summary", expanded=False):
                    st.caption("Global word impact across all predictions. Full breakdown in the **🔍 Explainability** tab.")
                    if st.button("Compute SHAP Summary", key="shap_quick_res"):
                        with st.spinner("Computing SHAP values…"):
                            sr = run_shap(latest['trained_model'],latest['vectorizer'],
                                          st.session_state.raw_texts, st.session_state.class_names)
                        if sr:
                            st.session_state.shap_result = sr
                            fig = plot_shap_summary(sr)
                            if fig: st.pyplot(fig)
                            st.success("Full breakdown available in the Explainability tab.")
                        else:
                            st.error("Install `shap` to use this feature: `pip install shap`")

                st.divider()
                dl1,dl2 = st.columns(2)
                with dl1:
                    st.download_button("⬇️ Download Report (.md)",
                        generate_automated_report(latest,st.session_state.df),
                        file_name="nlp_playground_report.md", mime="text/markdown",
                        use_container_width=True)
                with dl2:
                    model_bytes = pickle.dumps({'model':latest['trained_model'],'vectorizer':latest['vectorizer']})
                    st.download_button("⬇️ Download Model (.pkl)", data=model_bytes,
                        file_name=f"{latest['model_name'].replace(' ','_')}.pkl",
                        mime="application/octet-stream", use_container_width=True)
            else:
                st.markdown(f"**Model:** `{latest['model_name']}`  |  **Task:** {latest['task']}")
                st.dataframe(pd.DataFrame(latest['results'],columns=["Output"]),use_container_width=True)
                st.download_button("⬇️ Download Report (.md)",
                    generate_automated_report(latest,st.session_state.df),
                    file_name="nlp_playground_report.md",mime="text/markdown")

    # Explainability ──────────────────────────────────────────────────────────
    with t_xai:
        st.subheader("🔍 Explainability — SHAP & LIME")
        st.caption("Understand *why* your model made each prediction. SHAP explains the whole model; LIME explains a single sample.")

        latest_sk = next((r for r in reversed(st.session_state.model_results) if r['type']=='sklearn'), None)

        if latest_sk is None:
            st.info("Train a **Scikit-Learn** model first — explainability is only available for Scikit-Learn models.")
        else:
            model       = latest_sk['trained_model']
            vectorizer  = latest_sk['vectorizer']
            raw_texts   = st.session_state.raw_texts or []
            class_names = st.session_state.class_names or []
            n_samples   = len(raw_texts)
            n_classes   = len(class_names)

            st.markdown(f"Explaining: **{latest_sk['model_name']}** — {n_samples} samples, {n_classes} classes")
            st.divider()

            # ── SHAP ──────────────────────────────────────────────────────────
            st.markdown("### SHAP — global & per-prediction analysis")
            st.markdown(
                "SHAP (SHapley Additive exPlanations) measures how much each word "
                "contributed to a prediction, averaged fairly across all possible combinations."
            )

            c_btn, c_info = st.columns([2,3])
            with c_btn:
                if st.button("⚡ Compute SHAP Values", use_container_width=True, key="shap_xai_btn"):
                    with st.spinner("Running SHAP… (up to 30s for large datasets)"):
                        sr = run_shap(model, vectorizer, raw_texts, class_names)
                    if sr:
                        st.session_state.shap_result = sr
                        st.success("SHAP values ready.")
                    else:
                        st.error("Install `shap`: `pip install shap`")
            with c_info:
                method_map = {
                    "LogisticRegression":"LinearExplainer (fast)",
                    "LinearSVC":"LinearExplainer (fast)",
                    "RandomForestClassifier":"TreeExplainer (fast)",
                    "GradientBoostingClassifier":"TreeExplainer (fast)",
                    "MultinomialNB":"KernelExplainer (sampled to 50 rows)",
                }
                st.info(f"Method for **{latest_sk['model_name']}**: "
                        f"{method_map.get(type(model).__name__,'KernelExplainer')}", icon="ℹ️")

            if st.session_state.shap_result:
                sr = st.session_state.shap_result

                # Summary
                st.markdown("#### Summary plot")
                st.caption("Each dot = one sample. X-axis = SHAP value (impact magnitude). Color = feature value (high/low).")
                fig_s = plot_shap_summary(sr)
                if fig_s: st.pyplot(fig_s)

                st.divider()

                # Waterfall
                st.markdown("#### Waterfall plot — single prediction breakdown")
                st.caption("How each word pushed the model's output from the base (average) value to the final prediction.")
                wc1,wc2 = st.columns(2)
                with wc1: w_samp = st.slider("Sample", 0, max(0,n_samples-1), 0, key="wf_samp")
                with wc2: w_cls  = st.selectbox("Class", range(n_classes),
                                                 format_func=lambda i: class_names[i] if i<n_classes else str(i),
                                                 key="wf_cls")
                fig_w = plot_shap_waterfall(sr, sample_idx=w_samp, class_idx=w_cls)
                if fig_w: st.pyplot(fig_w)
                else:     st.warning("Could not render waterfall for this combination.")

                st.divider()

                # Force plot
                st.markdown("#### Force plot — interactive push/pull view")
                st.caption("Red = features pushing prediction higher. Blue = pushing it lower. Bar width = magnitude.")
                fc1,fc2 = st.columns(2)
                with fc1: f_samp = st.slider("Sample", 0, max(0,n_samples-1), 0, key="fp_samp")
                with fc2: f_cls  = st.selectbox("Class", range(n_classes),
                                                 format_func=lambda i: class_names[i] if i<n_classes else str(i),
                                                 key="fp_cls")
                force_html = plot_shap_force(sr, sample_idx=f_samp, class_idx=f_cls)
                if force_html: components.html(force_html, height=160, scrolling=False)
                else:          st.warning("Force plot unavailable for this model/class combination.")

            st.divider()

            # ── LIME ──────────────────────────────────────────────────────────
            st.markdown("### LIME — per-prediction word highlights")
            st.markdown(
                "LIME (Local Interpretable Model-agnostic Explanations) perturbs one text "
                "sample hundreds of times, then fits a simple model to explain what drove "
                "that single prediction."
            )

            if not raw_texts:
                st.info("Run a Scikit-Learn experiment to enable LIME.")
            else:
                lc1,lc2 = st.columns(2)
                with lc1: l_samp = st.slider("Sample to explain", 0, max(0,len(raw_texts)-1), 0, key="lime_samp")
                with lc2: l_cls  = st.selectbox("Class to explain", range(n_classes),
                                                  format_func=lambda i: class_names[i] if i<n_classes else str(i),
                                                  key="lime_cls")

                st.markdown("**Text being explained:**")
                st.info(f"> {raw_texts[l_samp][:500]}")

                if st.button("⚡ Run LIME", key="lime_run"):
                    with st.spinner("Running LIME — 300 perturbations…"):
                        lime_exp = run_lime(model, vectorizer, raw_texts, class_names, sample_idx=l_samp)
                    if lime_exp is None:
                        st.error("Install `lime`: `pip install lime`")
                    else:
                        st.session_state["lime_exp"]       = lime_exp
                        st.session_state["lime_cls_stored"] = l_cls
                        st.success("LIME explanation ready.")

                if st.session_state.get("lime_exp"):
                    lime_exp    = st.session_state["lime_exp"]
                    display_cls = st.session_state.get("lime_cls_stored", l_cls)

                    tab_bar, tab_html = st.tabs(["📊 Bar chart","🌈 Highlighted text"])
                    with tab_bar:
                        st.caption("Green = words supporting this class. Red = words opposing it.")
                        fig_l = plot_lime_explanation(lime_exp, label_idx=display_cls)
                        if fig_l: st.pyplot(fig_l)
                        else:     st.warning("No weights returned for this class.")
                    with tab_html:
                        st.caption("Original text with words coloured by their LIME contribution.")
                        lhtml = plot_lime_html(lime_exp, label_idx=display_cls)
                        if lhtml: components.html(lhtml, height=300, scrolling=True)
                        else:     st.warning("HTML rendering unavailable.")

    # Comparison ──────────────────────────────────────────────────────────────
    with t_comp:
        if not st.session_state.model_results:
            st.info("Run at least one experiment to see comparison data here.")
        else:
            st.subheader("Experiment Comparison Dashboard")
            sk_runs = [r for r in st.session_state.model_results if r['type']=='sklearn']
            if sk_runs:
                st.markdown("**Performance Chart (Sklearn Models)**")
                st.pyplot(plot_model_comparison(sk_runs))
                st.divider()
            st.markdown("**All Experiments**")
            all_rows = [{"Model":r['model_name'],"Task":r.get('task','N/A'),
                         "Type":r.get('type','N/A').upper(),
                         "Accuracy":f"{r['accuracy']:.4f}" if 'accuracy' in r else "—",
                         "F1 Score":f"{r['f1_score']:.4f}" if 'f1_score' in r else "—"}
                        for r in st.session_state.model_results]
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True)
            if st.button("🗑️ Clear All Experiments"):
                for k in ('model_results','shap_result','raw_texts','class_names'):
                    st.session_state[k] = [] if k=='model_results' else None
                st.rerun()

    # AI Consultant ───────────────────────────────────────────────────────────
    with t_consultant:
        render_consultant_tab(df=st.session_state.df,
                              model_results=st.session_state.model_results,
                              nlp_task=nlp_task)

else:
    st.markdown("---")
    cl, cr = st.columns([3,2])
    with cl:
        st.markdown("""
        ### 👋 Welcome to NLP Playground!
        1. **Select an NLP Task** in the sidebar
        2. **Upload a CSV or JSON** dataset
        3. **Map your columns** and choose preprocessing options
        4. **Pick a model** and click 🚀 Run Experiment
        """)
    with cr:
        st.markdown("""
        #### 🧩 Supported Tasks
        | Task | Engine |
        |---|---|
        | Text Classification | Scikit-Learn / 🤗 |
        | Text Summarization | 🤗 Transformers |
        | Question Answering | 🤗 Transformers |
        | Sentiment Analysis | 🤗 Transformers |
        """)