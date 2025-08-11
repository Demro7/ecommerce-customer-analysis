import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from app_logic import load_data, preprocess_data, train_models, plot_eda
import pickle
import os

# Set page configuration
st.set_page_config(page_title="E-commerce Data Analysis", page_icon="🛍️", layout="wide")

# Page title
st.title("E-commerce Customer Analysis & Spending Prediction")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Data Overview", "Data Preprocessing", "EDA", "Model Training", "Model Evaluation", "Prediction"]
)

# Load data
df = load_data()

if df is not None:
    if page == "Data Overview":
        st.header("Data Overview")
        st.write(f"Number of Rows: {df.shape[0]}")
        st.write(f"Number of Columns: {df.shape[1]}")
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        st.subheader("Missing Values Check")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.warning("There are missing values in the dataset.")
            st.write(missing_values[missing_values > 0])
        else:
            st.success("No missing values found in the dataset.")
            
    elif page == "Data Preprocessing":
        st.header("Data Preprocessing")
        test_size = st.slider("Test size (%):", 10, 40, 20)
        try:
            X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = preprocess_data(df, test_size/100)
            st.write("Numerical features:", numerical_features)
            st.write("Categorical features:", categorical_features)
            st.write("Target variable: Yearly Amount Spent")
            st.write(f"Training set: {X_train.shape[0]} samples")
            st.write(f"Testing set: {X_test.shape[0]} samples")
            st.success("Data processed and saved successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
            
    elif page == "EDA":
        st.header("Exploratory Data Analysis")
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        feature_options = [col for col in numeric_df.columns if col != "Yearly Amount Spent"]
        selected_feature = st.selectbox("Select Feature", feature_options)
        
        plots = plot_eda(df, selected_feature)
        
        st.subheader("Distribution of Yearly Amount Spent")
        st.pyplot(plots[0])
        
        st.subheader("Correlation Matrix")
        st.pyplot(plots[1])
        
        st.subheader("Feature vs Target Analysis")
        if plots[2] is not None:
            fig, correlation = plots[2]
            st.pyplot(fig)
            st.write(f"Correlation: {correlation:.4f}")
            
    elif page == "Model Training":
        st.header("Model Training")
        try:
            X_train = pd.read_csv("data/X_train.csv")
            X_test = pd.read_csv("data/X_test.csv")
            y_train = pd.read_csv("data/y_train.csv").iloc[:, 0]
            y_test = pd.read_csv("data/y_test.csv").iloc[:, 0]
            
            st.success("Data loaded successfully!")
            
            st.subheader("Select Models to Train")
            col1, col2 = st.columns(2)
            with col1:
                train_lr = st.checkbox("Linear Regression", value=True)
                train_ridge = st.checkbox("Ridge Regression")
                train_lasso = st.checkbox("Lasso Regression")
            with col2:
                train_rf = st.checkbox("Random Forest", value=True)
                train_gb = st.checkbox("Gradient Boosting")
                train_svr = st.checkbox("SVR")
                
            models_to_train = []
            if train_lr:
                models_to_train.append("Linear Regression")
            if train_ridge:
                models_to_train.append("Ridge")
            if train_lasso:
                models_to_train.append("Lasso")
            if train_rf:
                models_to_train.append("Random Forest")
            if train_gb:
                models_to_train.append("Gradient Boosting")
            if train_svr:
                models_to_train.append("SVR")
                
            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    results_df = train_models(X_train, y_train, X_test, y_test, models_to_train)
                    st.subheader("Results")
                    st.dataframe(results_df)
                    best_model_idx = results_df["R²"].idxmax()
                    best_model_name = results_df.loc[best_model_idx, "Model"]
                    best_model_r2 = results_df.loc[best_model_idx, "R²"]
                    st.success(f"Best model: {best_model_name} (R² = {best_model_r2:.4f})")
                    with open("data/best_model.txt", "w") as f:
                        f.write(best_model_name)
        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please complete the Data Preprocessing first!")

    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        try:
            results_df = pd.read_csv("data/model_results.csv")
            st.subheader("Model Comparison")
            st.dataframe(results_df)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(results_df["Model"], results_df["R²"])
            plt.title("Model Comparison - R² Score")
            plt.ylabel("R² Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            best_model_idx = results_df["R²"].idxmax()
            best_model_name = results_df.loc[best_model_idx, "Model"]
            st.subheader(f"Best Model: {best_model_name}")
            
            best_model_filename = best_model_name.lower().replace(" ", "_")
            with open(f"models/{best_model_filename}.pkl", "rb") as f:
                best_model = pickle.load(f)
                
            X_test = pd.read_csv("data/X_test.csv")
            y_test = pd.read_csv("data/y_test.csv").iloc[:, 0]
            
            y_pred = best_model.predict(X_test)
            
            st.subheader("Actual vs Predicted Values")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
            plt.title(f"{best_model_name} - Actual vs Predicted")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            st.pyplot(fig)
        
            if hasattr(best_model, "feature_importances_"):
                st.subheader("Feature Importance")
                feature_names = X_test.columns
                importances = best_model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values("Importance", ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df["Feature"], importance_df["Importance"])
                plt.title(f"{best_model_name} - Feature Importance")
                st.pyplot(fig)
                
            elif hasattr(best_model, "coef_"):
                st.subheader("Feature Coefficients")
                feature_names = X_test.columns
                coefficients = best_model.coef_
                coef_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Coefficient": coefficients
                }).sort_values("Coefficient", key=abs, ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(coef_df["Feature"], coef_df["Coefficient"])
                plt.title(f"{best_model_name} - Coefficients")
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please train models first!")

    elif page == "Prediction":
        st.header("Make Predictions")
        try:
            with open("data/best_model.txt", "r") as f:
                best_model_name = f.read().strip()
            best_model_filename = best_model_name.lower().replace(" ", "_")
            with open(f"models/{best_model_filename}.pkl", "rb") as f:
                model = pickle.load(f)
            st.success(f"Using model: {best_model_name}")
            
            with open("models/preprocessor.pkl", "rb") as f:
                preprocessor = pickle.load(f)
                
            numerical_features = preprocessor.transformers_[0][2]
            categorical_features = preprocessor.transformers_[1][2]
            try:
                feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
            except AttributeError:
                feature_names = numerical_features + categorical_features
                
            st.subheader("Enter Customer Data")
            col1, col2 = st.columns(2)
            input_data = {}
            original_df = load_data()
            
            for i, feature in enumerate(numerical_features + categorical_features):
                if feature in numerical_features:
                    default_value = float(original_df[feature].mean()) if feature in original_df.columns else 0.0
                    if i % 2 == 0:
                        with col1:
                            input_data[feature] = st.number_input(feature, value=default_value, format="%.2f")
                    else:
                        with col2:
                            input_data[feature] = st.number_input(feature, value=default_value, format="%.2f")
                elif feature in categorical_features:
                    options = original_df[feature].unique().tolist() if feature in original_df.columns else ["Unknown"]
                    if i % 2 == 0:
                        with col1:
                            input_data[feature] = st.selectbox(feature, options=options)
                    else:
                        with col2:
                            input_data[feature] = st.selectbox(feature, options=options)
                            
            input_df = pd.DataFrame([input_data])
            input_processed = preprocessor.transform(input_df)
            
            if st.button("Predict"):
                prediction = model.predict(input_processed)[0]
                st.subheader("Prediction Result")
                st.markdown(
                    f"<div style='text-align:center; padding:20px; background-color:#000380; border-radius:10px;'>"
                    f"<h2>Predicted Yearly Amount Spent</h2>"
                    f"<h1 style='color:#000000; font-size:3em;'>${prediction:.2f}</h1>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"Error: {e}")
