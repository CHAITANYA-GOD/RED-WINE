import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import streamlit as st
import joblib
import logging
import lime
from lime.lime_tabular import LimeTabularExplainer
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('winequality-red (1).csv')
        # Rename columns for better readability
        data.rename(columns={
            "fixed acidity": "fixed_acidity",
            "volatile acidity": "volatile_acidity",
            "citric acid": "citric_acid",
            "residual sugar": "residual_sugar",
            "chlorides": "chlorides",
            "free sulfur dioxide": "free_sulfur_dioxide",
            "total sulfur dioxide": "total_sulfur_dioxide"
        }, inplace=True)
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        st.error("Dataset file not found!")
        logger.error("Dataset file not found!")
        return None

# Preprocess the data
def preprocess_data(data):
    # Convert quality to categories
    data['quality'] = data['quality'].replace({
        8: 'Good',
        7: 'Good',
        6: 'Middle',
        5: 'Middle',
        4: 'Bad',
        3: 'Bad'
    })
    # Split features and target
    X = data.drop(columns='quality')
    y = data['quality']

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y, scaler

# Train and evaluate the model
def train_model(X, y, model):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Perform hyperparameter tuning with GridSearchCV (if applicable)
    if isinstance(model, SVC):
        param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    elif isinstance(model, RandomForestClassifier):
        param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, 30]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

    # Cross-validation for performance evaluation
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, random_state=0, shuffle=True), scoring='accuracy')
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean()}")

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, y_pred, accuracy, cv_scores

# Save the model
def save_model(model, filename):
    try:
        joblib.dump(model, filename)
        logger.info(f"Model saved successfully as {filename}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

# Load the model
def load_model(filename):
    try:
        model = joblib.load(filename)
        logger.info(f"Model loaded successfully from {filename}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Define prediction function
def predict_wine_quality(new_wine_data, scaler, model):
    """
    Predicts the quality of a new wine.
    """
    new_wine_df = pd.DataFrame([new_wine_data])
    normalized_new_wine = scaler.transform(new_wine_df)
    normalized_new_wine_df = pd.DataFrame(normalized_new_wine, columns=new_wine_df.columns)
    prediction = model.predict(normalized_new_wine_df)[0]
    return prediction

# Streamlit App
def main():
    st.title("🍷 Wine Quality Prediction")
    st.write("This app predicts the quality of red wine based on its features.")
    st.write("Use the sliders in the sidebar to input the features of the wine.")

    # Sidebar for user input
    st.sidebar.header("User Input Features")

    def user_input_features():
        feature_values = {feature: st.sidebar.slider(feature, float(data[feature].min()), float(data[feature].max()),
                                                     float(data[feature].mean())) for feature in data.columns if
                          feature != 'quality'}
        return feature_values

    user_data = user_input_features()

    # Display user input
    st.subheader("User Input Features")
    st.write(pd.DataFrame([user_data]))

    # Select model
    model_selection = st.sidebar.selectbox("Select Model",
                                           ["SVC", "Random Forest", "Logistic Regression", "KNN", "Decision Tree"])

    # Initialize the selected model
    model_filename = f"{model_selection.replace(' ', '_').lower()}_model.pkl"
    if os.path.exists(model_filename):
        model = load_model(model_filename)
        # Calculate accuracy for the loaded model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
    else:
        if model_selection == "SVC":
            model = SVC(probability=True, random_state=0)  # Ensure SVC has probability=True
        elif model_selection == "Random Forest":
            model = RandomForestClassifier(random_state=0)
        elif model_selection == "Logistic Regression":
            model = LogisticRegression(random_state=0)
        elif model_selection == "KNN":
            model = KNeighborsClassifier()
        elif model_selection == "Decision Tree":
            model = DecisionTreeClassifier(random_state=0)

        # Train and evaluate the model
        model, X_train, X_test, y_train, y_test, y_pred, accuracy, cv_scores = train_model(X, y, model)
        save_model(model, model_filename)

    # Display the prediction
    prediction = predict_wine_quality(user_data, scaler, model)
    st.subheader("Prediction")
    st.write(f"The predicted wine quality is: **{prediction}**")

    # Model evaluation
    st.subheader(f"{model_selection} Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    st.write("Confusion Matrix:")
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Classification Report
    st.write("Classification Report:")
    cr = metrics.classification_report(y_test, y_pred)
    st.text(cr)

    # Cross-validation results (only for newly trained models)
    if not os.path.exists(model_filename):
        st.write("Cross-validation Results:")
        st.write(f"Cross-validation scores: {cv_scores}")
        st.write(f"Mean Cross-validation score: {cv_scores.mean():.2f}")

    # LIME for model interpretability
    explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=np.unique(y),
                                     discretize_continuous=True)

    # Convert user input to a pandas DataFrame for LIME
    user_data_df = pd.DataFrame([user_data])

    # Now pass the DataFrame to the explain_instance method
    lime_exp = explainer.explain_instance(user_data_df.values[0], model.predict_proba)

    # Show the explanation as a plot in Streamlit
    fig = lime_exp.as_pyplot_figure()
    st.pyplot(fig)

    # Feature importance (only for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        st.write("Feature Importances:")
        st.write(feature_importance_df)

        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    data = load_data()
    if data is None:
        st.stop()  # Exit the app if no data is loaded
    X, y, scaler = preprocess_data(data)
    main()