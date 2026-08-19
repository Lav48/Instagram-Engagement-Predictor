# Instagram Engagement Predictor

A machine learning-based system designed to predict the potential engagement of Instagram posts using content, audience, and post-level features. The project combines data preprocessing, feature engineering, exploratory data analysis, multiple regression algorithms, model evaluation, and a web-based prediction interface.

## 📌 Project Overview

Social media engagement plays an important role in understanding how effectively content performs on platforms such as Instagram. However, predicting the performance of a post before publishing it can be challenging because engagement depends on multiple factors, including audience size, caption characteristics, hashtags, and other post attributes.

This project develops an **Instagram Engagement Prediction system** that uses machine learning techniques to estimate post engagement and identify the factors that contribute to better performance.

The system follows a complete machine learning pipeline:

**Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction**

## 🎯 Objectives

* Predict Instagram post engagement using machine learning.
* Identify features that influence post performance.
* Compare multiple regression algorithms.
* Evaluate models using standard performance metrics.
* Build a prediction interface that can be used to estimate engagement for new posts.
* Provide insights that can help content creators understand potential post performance.

## ✨ Key Features

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Automated feature engineering
* Instagram-specific feature extraction
* Multiple machine learning models
* Cross-validation for model comparison
* Model performance evaluation
* Feature importance analysis
* Engagement prediction
* Web-based prediction interface
* Ranking functionality for comparing predicted engagement

## 🧠 Machine Learning Models

The project evaluates multiple regression algorithms:

| Model                   | Purpose                                                                    |
| ----------------------- | -------------------------------------------------------------------------- |
| Linear Regression       | Establishes a simple baseline relationship between features and engagement |
| Ridge Regression        | Reduces overfitting using L2 regularization                                |
| Lasso Regression        | Performs regularization and can help identify important features           |
| ElasticNet              | Combines L1 and L2 regularization                                          |
| Random Forest Regressor | Captures nonlinear relationships using an ensemble of decision trees       |

The models are compared using multiple evaluation metrics to determine their predictive performance.

## 📊 Features Used

Feature engineering is performed to extract meaningful information from the Instagram data.

Some of the engineered features include:

* **Follower-Following Ratio**
  Represents the relationship between an account's followers and following count.

* **Caption Length**
  Measures the number of characters in the post caption.

* **Word Count**
  Represents the number of words contained in the caption.

* **Hashtag Count**
  Measures the number of hashtags used in the post.

These features allow the machine learning models to learn relationships between post characteristics and engagement.

## 🔬 Machine Learning Pipeline

The project follows these major stages:

### 1. Data Collection

Instagram-related post and account information is collected and organized into a structured dataset.

### 2. Data Preprocessing

The raw dataset is cleaned and transformed to make it suitable for machine learning.

This includes:

* Handling missing values
* Removing unnecessary information
* Formatting variables
* Preparing numerical and categorical features

### 3. Exploratory Data Analysis

EDA is performed to understand:

* Data distributions
* Feature relationships
* Engagement patterns
* Potential outliers
* Important correlations

### 4. Feature Engineering

Additional features are generated from existing data to improve the predictive capability of the models.

### 5. Model Training

Multiple regression algorithms are trained using the processed dataset.

The dataset is divided into training and testing subsets using a train-test split.

### 6. Cross-Validation

Cross-validation is used to evaluate model stability and reduce dependence on a single train-test split.

### 7. Model Evaluation

The models are evaluated using:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

### 8. Prediction

The trained model can be used to estimate engagement for new Instagram posts.

## 📈 Evaluation Metrics

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted engagement values.

Lower MAE indicates better performance.

### Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction error.

Lower RMSE indicates better performance and gives greater weight to larger errors.

### R² Score

Measures how well the model explains the variation in the target variable.

A higher R² score generally indicates better explanatory performance.

## 🏗️ Project Structure

```text
Instagram-Engagement-Predictor/
│
├── backend/
│   ├── app.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── ranking.py
│   └── train.py
│
├── frontend/
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
│
├── reports/
│   ├── feature_importance.csv
│   └── model_results.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠️ Technology Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Regression
* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet

### Data Processing

* Pandas
* NumPy

### Data Visualization / Analysis

* Matplotlib
* Seaborn

### Web Application

* Flask
* HTML
* CSS

### Development Tools

* Git
* GitHub
* Jupyter / Python development environment

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Lav48/Instagram-Engagement-Predictor.git
```

### 2. Navigate to the project

```bash
cd Instagram-Engagement-Predictor
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python backend/app.py
```

The application can then be accessed through the local Flask server shown in the terminal.

## 📁 Dataset

The original dataset and trained model artifacts are not included in this repository because of their large file sizes.

The repository therefore contains the source code required to reproduce the preprocessing, feature engineering, training, evaluation, and prediction workflow.

## 📋 Results

The project compares the performance of multiple regression models using MAE, RMSE, and R² Score.

Detailed model results and feature importance information are available in:

```text
reports/model_results.csv
reports/feature_importance.csv
```

## 🔍 Research Relevance

The project demonstrates how machine learning can be applied to social media analytics and predictive modeling.

It combines:

* Data preprocessing
* Feature engineering
* Statistical analysis
* Supervised machine learning
* Model comparison
* Cross-validation
* Predictive analytics
* Web application development

This makes the project relevant to areas such as **Artificial Intelligence, Machine Learning, Data Science, Social Media Analytics, and Predictive Modeling**.

## 🔮 Future Scope

Possible future improvements include:

* Incorporating image-based features using computer vision.
* Using NLP techniques to analyze caption sentiment and semantic content.
* Incorporating posting time and day-of-week information.
* Using deep learning models for more complex prediction patterns.
* Integrating real-time Instagram data through appropriate APIs.
* Developing personalized recommendations for content creators.
* Deploying the application as a cloud-based service.
* Adding explainable AI techniques to provide more detailed reasons behind predictions.

## 👩‍💻 Author

**Lavanya Maithani**

B.Tech Computer Science & Engineering

GitHub: [Lav48](https://github.com/Lav48)

## 📄 License

This project is developed for academic and educational purposes.
