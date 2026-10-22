# 👗 FashionIQ — AI Fashion Trends Intelligence & Forecasting System

### Data Analytics | Machine Learning | Time Series Forecasting | Streamlit

FashionIQ is an end-to-end fashion retail analytics and demand forecasting project developed using Python, Pandas, Scikit-learn, Plotly, and Streamlit.

The system analyzes historical fashion sales, identifies seasonal demand patterns, compares forecasting techniques, and generates category-level demand estimates through an interactive web application.

The project demonstrates practical applications of data preprocessing, exploratory data analysis, feature engineering, machine learning, model evaluation, and dashboard development.

---

## 🎯 Project Objectives

* Analyze historical fashion retail transactions to understand purchasing patterns.
* Identify high-performing fashion categories and subcategories.
* Examine monthly demand fluctuations and seasonal trends.
* Develop and evaluate multiple demand forecasting approaches.
* Compare machine learning models against statistical baselines.
* Build an interactive dashboard for fashion demand analysis and forecasting.

---

## 📊 Dataset Overview

The project uses a fashion retail transaction dataset obtained from Kaggle.

The original dataset includes transaction records and supporting product, customer, store, employee, and discount information.

### Data Processing

The transaction dataset was processed and aggregated into monthly fashion sales.

| Attribute                     | Description                  |
| ----------------------------- | ---------------------------- |
| Historical period             | January 2023 – February 2025 |
| Months analyzed               | 26                           |
| Fashion categories            | 3                            |
| Category–subcategory segments | 26                           |
| Total units sold              | 6,537,934                    |
| Forecasting target            | Monthly units sold           |

### Fashion Categories

* Feminine
* Masculine
* Children

March 2025 was excluded from the historical analysis because the source dataset contains only a partial month.

The forecasting analysis uses unit sales rather than combined revenue because the original dataset contains transactions involving different currencies.

---

## 🔍 Exploratory Data Analysis

Exploratory data analysis was conducted to identify important retail demand patterns.

The analysis includes:

* Monthly fashion sales trends
* Category-level demand distribution
* Best-selling fashion segments
* Seasonal purchasing patterns
* Historical demand comparisons
* Demand fluctuations across fashion categories

Interactive visualizations were developed using Plotly.

---

## 🤖 Machine Learning and Forecasting

FashionIQ evaluates multiple approaches for forecasting monthly fashion demand.

### Forecasting Methods

**1. Seasonal Baseline**

Predicts demand using sales from the corresponding calendar month in the previous year.

**2. Random Forest Regressor**

Uses historical demand features and fashion segment information to estimate future sales.

**3. Growth-Adjusted Forecast**

Adjusts seasonal predictions using historical demand growth patterns.

**4. Residual Random Forest**

Uses machine learning to estimate adjustments to baseline predictions.

### Feature Engineering

Features developed for the forecasting models include:

* Previous-month sales
* Two-month and three-month lagged sales
* Previous-year sales
* Three-month rolling average of historical sales
* Month and year
* Fashion category
* Fashion subcategory

Lagged features were constructed using historical observations to avoid incorporating future sales into model inputs.

---

## 📈 Model Evaluation

Forecasting approaches were compared using:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

The project includes an extended historical evaluation covering:

**July 2024 – February 2025**

| Evaluation Metric                | Result            |
| -------------------------------- | ----------------- |
| Evaluation months                | 8                 |
| Category-month predictions       | 208               |
| Best-performing evaluated method | Seasonal Baseline |
| Seasonal Baseline MAE            | 1,069.95 units    |
| Seasonal Baseline RMSE           | 1,482.53 units    |

The seasonal baseline achieved the lowest errors in the extended evaluation.

These results demonstrate that a relatively simple forecasting approach can outperform more complex machine learning models on a particular dataset.

**Evaluation limitation:** The extended results represent one-month-ahead historical predictions using observed lagged sales. They should not be interpreted as verified accuracy for the full 12-month future forecast.

---

## 🖥️ Streamlit Application

The FashionIQ web application contains four main modules.

### 1. Fashion Intelligence Dashboard

Provides an overview of historical retail performance.

Features include:

* Total units sold
* Number of fashion segments
* Historical coverage
* Monthly demand trends
* Best-selling segments
* Fashion category distribution

### 2. Fashion Trends Intelligence

Allows users to explore demand patterns across different fashion categories.

Features include:

* Category filtering
* Monthly sales analysis
* Top-performing subcategories
* Seasonal demand analysis
* Historical sales tables

### 3. Demand Forecasting

Generates seasonal demand estimates for selected fashion segments.

Features include:

* Category and subcategory selection
* Monthly demand estimates
* Historical sales versus forecast visualization
* Twelve-month forecast summary
* Peak demand month identification
* Downloadable forecast CSV

The application presents historical seasonal projections for March 2025 through February 2026.

These projections are not live market forecasts and have not been validated against actual sales for those months.

### 4. Model Performance

Displays the results of forecasting model comparisons.

Features include:

* Best-performing model identification
* MAE comparison charts
* RMSE comparison charts
* Historical evaluation results
* Prediction comparison tables

---

## 🛠️ Technologies Used

| Category                | Technologies                |
| ----------------------- | --------------------------- |
| Programming             | Python                      |
| Data Processing         | Pandas, NumPy               |
| Machine Learning        | Scikit-learn                |
| Data Visualization      | Plotly, Matplotlib, Seaborn |
| Web Application         | Streamlit                   |
| Development Environment | Visual Studio Code          |
| Version Control         | Git, GitHub                 |

---

## 📁 Project Structure

```text
ai-fashion-trends-intelligence/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── fashion_monthly_sales.csv
│   └── fashion_ml_features.csv
│
├── models/
│   ├── extended_model_comparison.csv
│   ├── extended_model_predictions.csv
│   ├── model_comparison.csv
│   └── random_forest_predictions.csv
│
├── prepare_fashion_data.py
├── fashion_trend_analysis.py
├── prepare_ml_features.py
├── train_fashion_model.py
├── train_random_forest.py
├── train_growth_forecast.py
├── train_residual_model.py
├── evaluate_fashion_model.py
├── evaluate_models_extended.py
└── model_comparison.py
```

Additional analysis scripts, model artifacts, and visualizations are included in the repository.

Large raw transaction datasets are excluded from version control.

---

## 🚀 Run the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/anuvarna-r/ai-fashion-trends-intelligence.git
```

### 2. Open the project directory

```bash
cd ai-fashion-trends-intelligence
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Launch the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 💡 Key Project Insights

* Fashion demand exhibits significant seasonal fluctuations.
* Demand patterns vary across product categories and subcategories.
* Historical sales analysis helps identify high-performing retail segments.
* Seasonal forecasting provides a useful benchmark for evaluating machine learning models.
* More complex models do not necessarily outperform simpler forecasting approaches.
* Model evaluation is essential before selecting a forecasting technique for business applications.

---

## ⚠️ Limitations and Future Improvements

The project currently uses historical retail transactions ending in February 2025.

Future improvements may include:

* Integrating newer transaction datasets
* Evaluating longer forecast horizons
* Incorporating product pricing and promotional effects
* Exploring additional time series forecasting models
* Improving uncertainty estimation
* Developing inventory planning recommendations

---

## 👩‍💻 Author

**Anuvarna R.**

BCA Data Science Graduate

Python Developer Intern | Data Analytics & Machine Learning Enthusiast

GitHub: [anuvarna-r](https://github.com/anuvarna-r)

---

*FashionIQ demonstrates an end-to-end approach to retail data analytics, machine learning experimentation, and interactive demand forecasting.*
