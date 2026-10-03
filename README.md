# 🚲 Bike Rental Demand Prediction

A machine learning project that predicts daily bike rental demand using historical weather and calendar information.

This project was built as an independent machine learning learning project, covering data preprocessing, regression, time-aware validation, model comparison, hyperparameter tuning, feature engineering, and a Streamlit prediction interface.

## 📌 Project Overview

Bike-sharing companies need to estimate how many bikes may be rented on a particular day.

This project uses historical bike-sharing data to predict the total number of daily rentals (`cnt`) based on factors such as:

- Temperature
- Humidity
- Wind speed
- Season
- Month
- Weekday
- Working day
- Weather conditions
- Holiday status

## 📊 Dataset

The project uses the **UCI Bike Sharing Dataset**.

The dataset contains daily bike rental information from a bike-sharing system in Washington, D.C., covering 2011–2012.

Target variable:

```text
cnt
```

which represents the total number of bike rentals.

Dataset source:

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

## 🧠 Machine Learning Workflow

The project followed these steps:

1. Loaded and inspected the dataset.
2. Converted the date column into a datetime format.
3. Sorted the data chronologically.
4. Removed target-leaking columns such as `casual` and `registered`.
5. Used an 80/20 chronological train-test split.
6. Standardized numerical features.
7. One-hot encoded categorical features.
8. Compared multiple regression approaches:
   - Dummy Regression
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
9. Used `TimeSeriesSplit` for time-aware cross-validation.
10. Experimented with Random Forest and Gradient Boosting hyperparameters.
11. Investigated feature engineering, including time-based and cyclical representations.
12. Compared models using MAE, RMSE and R².
13. Built a Streamlit interface for making predictions.

## 📏 Evaluation Metrics

The project uses:

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual rental counts.

Lower is better.

### RMSE — Root Mean Squared Error

Similar to MAE but penalizes larger errors more heavily.

Lower is better.

### R² — R-squared

Measures how much variation in rental demand is explained by the model relative to a mean-prediction baseline.

Higher is better.

## 🔬 Model Comparison

The project found that model performance depended strongly on the validation period.

A simple Linear Regression model performed surprisingly competitively, while the tree-based models showed substantially different behavior between training and future unseen periods.

This was especially useful for demonstrating the importance of:

- Time-aware validation
- Detecting overfitting
- Comparing models using the same evaluation procedure
- Testing rather than assuming that more complex models are better

The exact experimental results are included in the project code and were obtained during development.

## 🌐 Streamlit Application

The project also includes a small Streamlit application that allows a user to enter weather and calendar information and receive an estimated number of bike rentals.

Run it with:

```bash
streamlit run app.py
```

## 🗂️ Project Structure

```text
Bike Rental Prediction/
│
├── Bike_renting.py
├── app.py
├── bike_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/krishihuang/Bike-Rental-Demand-Prediction.git
cd "Bike Rental Prediction"
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## ⚠️ Limitations

- This is an educational machine learning project.
- The dataset is historical and does not guarantee future real-world performance.
- Model performance varies across different time periods.
- The final test period was examined during development, so the reported results should be considered exploratory rather than a completely untouched final evaluation.
- Weather inputs in the current Streamlit application use the normalized scale used by the original dataset.

## 🤖 AI Assistance

The core machine learning work was developed independently, including data preparation, feature selection, preprocessing, model experimentation, time-aware validation, hyperparameter experiments, debugging, and interpretation of results.

AI assistance was used for parts of the Streamlit application implementation, explanations, debugging assistance, and development guidance.

This project was created as a learning exercise with the goal of understanding the underlying machine learning concepts rather than simply producing a trained model.
