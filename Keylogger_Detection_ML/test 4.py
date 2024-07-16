import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(file_path):
    print("Loading and cleaning data...")
    df = pd.read_csv(file_path, low_memory=False)
    print("Data loaded successfully!")
    df.dropna(inplace=True)
    print("Rows with missing values dropped successfully!")
    df['is_keylogger'] = df['is_keylogger'].map({'malicious': 1, 'benign': 0})
    print("'is_keylogger' column converted to numerical values successfully!")
    return df

def split_data(df):
    print("Splitting data into features and labels...")
    X = df['Process name']
    y = df['is_keylogger']
    print("Data split into features and labels successfully!")
    return X, y

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    print("Splitting data into training and testing sets...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def tfidf_vectorize(X_train, X_test):
    print("Converting text data to numerical features using TF-IDF...")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Text data converted to numerical features successfully!")
    return X_train_tfidf, X_test_tfidf

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"Training and evaluating {model_name} model...")
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores for {model_name}: {scores}")
    print(f"Mean cross-validation score for {model_name}: {np.mean(scores)}")
    model.fit(X_train, y_train)
    print(f"{model_name} model trained successfully!")
    evaluate_model(X_train, y_train, X_test, y_test, model, model_name, scores)

def evaluate_model(X_train, y_train, X_test, y_test, model, model_name, scores):
    print("Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Testing accuracy: {test_accuracy:.3f}")
    print("Training classification report:")
    print(classification_report(y_train, y_train_pred))
    print("Testing classification report:")
    print(classification_report(y_test, y_test_pred))

    # Plotting the cross-validation scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(scores)
    plt.title(f'Cross-validation scores for {model_name}')
    plt.ylabel('Score')
    plt.show()

    # Plotting the confusion matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f'Confusion Matrix for {model_name} on Testing Data')
    plt.show()

# Verify the directory path
directory = r'C:\Users\himan\Downloads\Keylogger-Detection\Keylogger_Detection_ML'
if os.path.exists(directory):
    print("Directory exists!")
    print("Files in the directory:")
    print(os.listdir(directory))
else:
    print(f"Directory does not exist: {directory}")

# If the directory exists, proceed with loading the data
file_path = os.path.join(directory, 'process_data.csv')
if os.path.exists(file_path):
    print("File exists!")
    print("Starting data processing...")
    df = load_and_clean_data(file_path)
    print("Data loaded and cleaned successfully!")

    X, y = split_data(df)
    print("Data split into features and labels successfully!")

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print("Data split into training and testing sets successfully!")

    X_train_tfidf, X_test_tfidf = tfidf_vectorize(X_train, X_test)
    print("Text data converted to numerical features successfully!")

    # Train and evaluate logistic regression model
    logreg = LogisticRegression()
    train_and_evaluate_model(logreg, X_train_tfidf, y_train, X_test_tfidf, y_test, 'LogisticRegression')
    print("Logistic regression model trained and evaluated successfully!")

    # Train and evaluate Random Forest classifier
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(random_forest, X_train_tfidf, y_train, X_test_tfidf, y_test, 'RandomForestClassifier')
    print("Random Forest classifier trained and evaluated successfully!")

    # Train and evaluate Gradient Boosting classifier
    gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(gradient_boosting, X_train_tfidf, y_train, X_test_tfidf, y_test, 'GradientBoostingClassifier')
    print("Gradient Boosting classifier trained and evaluated successfully!")
else:
    print(f"File does not exist: {file_path}")
    print("Please check the file path and try again.")
