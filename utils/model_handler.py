from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

def get_model(model_name, params={}):
    """
    Returns an untrained instance of the selected Scikit-learn model,
    configured with the provided hyperparameters.
    """
    if model_name == "Logistic Regression":
        return LogisticRegression(random_state=42, **params)
    elif model_name == "Naive Bayes":
        return MultinomialNB() 
    elif model_name == "Support Vector Machine (SVM)":
        return LinearSVC(random_state=42, dual=False, **params)
    elif model_name == "Random Forest":
        return RandomForestClassifier(random_state=42, **params)
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42, **params)
    return None

def train_model(df, text_column, target_column, model, test_size):
    X = df[text_column]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)

    return model, vectorizer, X_test_vec, y_test

def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics = {'accuracy': accuracy, 'f1_score': f1}
    return metrics, y_pred