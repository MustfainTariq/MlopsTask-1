# Lightweight Linear Regression API

This project is a lightweight linear regression model built using **NumPy** and deployed using **Flask**. The model predicts a target variable based on a set of input features. The goal is to provide a minimalistic machine learning solution that fits within the size constraints of serverless platforms like **Vercel**.

## Features

- **Simple Linear Regression**: The model uses NumPy to implement a linear regression model, trained using the Normal Equation.
- **Flask API**: The Flask web framework is used to serve predictions via a RESTful API.
- **Minimal Dependencies**: The project uses only `Flask`, `NumPy`, and `joblib`, making it lightweight and efficient.
- **Vercel Deployment**: The API is designed to be deployed on Vercel, with minimal file size to avoid the 250 MB limit on serverless functions.

## Model Overview

The model is a simple linear regression trained on a small, synthetic dataset generated using NumPy. The **Normal Equation** method is used to calculate the regression coefficients. The model predicts the target value (output) for a given set of input features.

### Model Training

The training data consists of a single feature and target values, with the relationship defined by:
Where `x` is the input feature and `y` is the target value. Noise is added to simulate real-world data.

## Project Structure

```bash
.
├── app.py               # Flask API for making predictions
├── main.py              # Script to train the linear regression model
├── linear_regression_model.joblib # Saved model parameters
├── requirements.txt     # Project dependencies
└── vercel.json          # Configuration file for Vercel deployment


