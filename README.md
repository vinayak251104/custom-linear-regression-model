# Custom Linear Regression Model

A from-scratch implementation of linear regression using NumPy and gradient descent — built to predict house prices and benchmarked against scikit-learn’s model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinayak251104/custom-linear-regression-model/blob/main/notebooks/HousePredictionWithCustomLinearRegressionModel.ipynb)



## Features

- Implemented linear regression using NumPy (no ML libraries)
- Optimized using gradient descent
- Custom `.score()` and `.squared_error()` methods
- Compared with scikit-learn’s `LinearRegression`
- Achieved R² score: **0.6611** vs sklearn's **0.6801**
- Plotted squared error vs learning rate (log-scale)
- Packaged model as reusable class

## Getting Started

Clone this repo and install the dependencies:

```bash
git clone https://github.com/vinayak251104/custom-linear-regression-model.git
cd custom-linear-regression-model
pip install -r requirements.txt

