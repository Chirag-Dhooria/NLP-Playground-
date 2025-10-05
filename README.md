#  NLP Playground

## Overview

**NLP Playground** is a **no-code web application** designed to democratize **Natural Language Processing (NLP)**.  
It provides an intuitive, user-friendly interface that enables users—including researchers, students, and domain experts without a programming background—to:

- Upload text datasets  
- Perform preprocessing  
- Train classical machine learning models  
- Evaluate their performance through **interactive visualizations** and **comparisons**

The platform streamlines the **end-to-end NLP workflow**, from data ingestion to model analysis, making it a powerful tool for **rapid prototyping, experimentation, and education**.

---

##  Key Features

###  Intuitive UI
A clean, multi-page web interface built with **Streamlit**.

###  Flexible Data Loading
Supports dataset uploads in both **CSV** and **JSON** formats.

###  Dynamic Configuration
Interactively select **feature (text)** and **target (label)** columns from uploaded data.

###  Comprehensive Preprocessing
Includes a suite of standard text cleaning options:
- Lowercasing  
- Punctuation Removal  
- Stopword Removal  
- Lemmatization  
- Stemming  

###  Classical Model Training
Train and evaluate a variety of **Scikit-learn** models:
- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest  
- Gradient Boosting  

###  Exploratory Data Analysis (EDA)
Generate insightful visualizations to understand the dataset:
- Label Distribution plots  
- Word Clouds  
- N-gram Analysis  

###  In-depth Evaluation
Assess model performance using:
- Accuracy and F1-Score metrics  
- Confusion Matrix visualizations  

###  Experiment Tracking
A **model comparison dashboard** caches the results of the last 10 experiments, enabling **side-by-side analysis** of models and preprocessing configurations.

###  Export Functionality
- Download trained models as `.pkl` files  
- Export performance metrics as CSV reports for further analysis  

---

##  Tech Stack

| Layer | Technology |
|--------|-------------|
| **Backend** | Python |
| **Web Framework** | Streamlit |
| **Data Manipulation** | Pandas |
| **Machine Learning** | Scikit-learn |
| **Text Processing** | NLTK |
| **Visualization** | Matplotlib, Seaborn, WordCloud |

---

##  Getting Started

Follow these instructions to set up and run **NLP Playground** locally.

###  Prerequisites
- **Python 3.8+**
- **pip** for package management

---

###  Installation

#### 1. Clone the repository
```bash
git clone <your-repository-url>
cd nlp-playground-
```

#### 2. Create and activate a virtual environment

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
.env\Scriptsctivate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Download NLTK Data Models
The app requires specific **NLTK packages** for preprocessing.  
Run the setup script to download them:

```bash
python setup.py
```

> 💡 *Note:* You’ll need to create a `setup.py` file to handle the NLTK data downloads (e.g., stopwords, punkt, wordnet).

---

###  Running the Application

Once setup is complete, launch the Streamlit application:

```bash
streamlit run app.py
```

The app will open automatically in your default web browser.

---

##  Project Structure

```
nlp-playground-/
├── app.py                  
├── utils/                  
│   ├── data_loader.py
│   ├── model_handler.py
│   ├── preprocessing.py
│   └── visualizer.py
├── static/                
├── .streamlit/             
│   └── config.toml
├── requirements.txt        
└── README.md              
```

---

##  Future Work

Planned enhancements include:

-  **Integration of deep learning models** using Hugging Face Transformers  
-  **Model Explainability (XAI)** using LIME and SHAP  
-  **Support for advanced NLP tasks** like Named Entity Recognition (NER) and Summarization  
-  **MLOps features** such as MLflow experiment tracking and one-click deployment  

---

##  License

This project is open-source and available under the **MIT License**.

---


