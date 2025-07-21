# Employee Salary Prediction

This project predicts whether a person earns more than 50K or not, based on their details like education, job type, hours worked per week, and a few other factors. I’ve used machine learning to train the model and built a simple web app using Streamlit where you can input the details and get a prediction instantly.

## What’s the idea?

The main goal was to build a salary prediction system using a dataset called `adult 3.csv`. It includes both numbers and categories (like work class and education). This kind of tool can be helpful for HR teams, job portals, or anyone who wants to analyze income patterns.

## Tools and Tech I Used

- Python (the core language)
- pandas and numpy for data handling
- scikit-learn for training the ML model
- joblib to save the model and encoders
- Streamlit to build the web interface

## How I built it

1. Cleaned the dataset (it had missing values and extra columns)
2. Picked only the important features to avoid errors later
3. Encoded the text values into numbers
4. Trained a Decision Tree Classifier
5. Saved the model and the encoders
6. Created a Streamlit app where users can enter inputs and see the prediction

## Accuracy

The model gives about **80.3% accuracy**, which is decent for this kind of classification problem.

## To Run the Project

First, install the required libraries if you haven't already:

```bash
pip install pandas numpy scikit-learn streamlit joblib




## run file
streamlit run app.py
