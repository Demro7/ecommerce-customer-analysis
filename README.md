# 🛍️ E-commerce Customer Analysis

**Interactive Streamlit web app for E-commerce customer data analysis, model training, and yearly spending prediction.**

An interactive **Streamlit** web app for analyzing e-commerce customer data, exploring relationships between features, training predictive models, and estimating yearly spending for new customers.

## 📌 Features

- **Data Overview** → View dataset statistics, sample records, and missing values.
- **Data Preprocessing** → Scaling, encoding categorical variables, handling outliers, and train/test split.
- **Exploratory Data Analysis (EDA)** → Visualize distributions, correlations, and feature relationships with the target.
- **Model Training** → Train multiple regression models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, SVR).
- **Model Evaluation** → Compare models using R², RMSE, and visualize feature importance or coefficients.
- **Prediction** → Predict yearly spending for new customer inputs.

---

## 📂 Project Structure

```
ecommerce-analysis/
│
├── app.py               # Main Streamlit app (UI and navigation)
├── app_logic.py         # Core logic (EDA, preprocessing, model training)
├── ecommerce.csv        # Dataset
├── requirements.txt     # Required dependencies
├── models/              # Saved trained models (created after training)
└── data/                # Processed datasets (created after preprocessing)
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Demro7/ecommerce-customer-analysis.git
cd ecommerce-customer-analysis
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📊 Example Workflow

1. **Data Overview** → Inspect dataset.
2. **Data Preprocessing** → Clean and prepare data.
3. **EDA** → Explore correlations and feature impacts.
4. **Model Training** → Train multiple models and compare performance.
5. **Model Evaluation** → Review best model metrics and feature importance.
6. **Prediction** → Input customer data and get spending prediction.

---

## 🖼 Screenshots (Optional)

You can add example screenshots here once the app is running, like:

- Dataset preview
- EDA plots
- Model comparison chart
- Prediction form

---

## 🛠 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---
