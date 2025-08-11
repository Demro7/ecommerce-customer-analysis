#__________________________________________________________________________________________1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def load_data():
    try:
        df = pd.read_csv("ecommerce.csv")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def save_model(model, model_name):
    if not os.path.exists("models"):
        os.makedirs("models")
    with open(f"models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
#_____________________________________________________________________________________________________________________2
def preprocess_data(df, test_size=0.2):
    df_processed = df.copy()
    
    # Drop irrelevant columns
    df_processed = df_processed.drop(['Email', 'Address', 'Avatar'], axis=1)
    
    # Separate features and target
    X = df_processed.drop('Yearly Amount Spent', axis=1)
    y = df_processed['Yearly Amount Spent']
    
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    # Outlier handling (IQR capping)
    for col in numerical_features:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Fit and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    try:
        feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    except AttributeError:
        feature_names = numerical_features + categorical_features
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Save preprocessor and data
    if not os.path.exists("models"):
        os.makedirs("models")
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    X_train_df.to_csv("data/X_train.csv", index=False)
    X_test_df.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    
    return X_train_df, X_test_df, y_train, y_test, preprocessor, numerical_features, categorical_features
#________________________________________________________________________________________________________________________4
def train_models(X_train, y_train, X_test, y_test, models_to_train):
    results = {}
    
    for model_name in models_to_train:
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge()
        elif model_name == "Lasso":
            model = Lasso()
        elif model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_name == "SVR":
            model = SVR()
        else:
            continue
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            "model": model,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2
        }
        save_model(model, model_name.lower().replace(" ", "_"))
    
    results_df = pd.DataFrame({
        "Model": list(results.keys()),
        "MSE": [results[model]["mse"] for model in results],
        "RMSE": [results[model]["rmse"] for model in results],
        "R²": [results[model]["r2"] for model in results]
    })
    
    results_df.to_csv("data/model_results.csv", index=False)
    
    return results_df
#_____________________________________________________________________________________________________________3
def plot_eda(df, feature=None):
    plots = []
    
    # Distribution plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Yearly Amount Spent"], kde=True, ax=ax1)
    ax1.set_title("Distribution of Yearly Amount Spent")
    plots.append(fig1)
    
    # Correlation heatmap
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr_matrix = numeric_df.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    plots.append(fig2)
    
    # Feature vs Target scatter plot
    if feature in numeric_df.columns and feature != "Yearly Amount Spent":
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df[feature], y=df["Yearly Amount Spent"], ax=ax3)
        ax3.set_title(f"{feature} vs Yearly Amount Spent")
        correlation = df[feature].corr(df["Yearly Amount Spent"])
        plots.append((fig3, correlation))
    else:
        plots.append(None)
    
    return plots