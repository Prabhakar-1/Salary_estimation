# Salary Prediction using k-Nearest Neighbors (k-NN)

This repository contains code for a salary prediction project using the k-Nearest Neighbors (k-NN) model. The project aims to estimate salaries based on various factors using a machine learning approach.

## Detailed Description

- **Problem Identification**: The project begins by understanding the application context and defining the problem statement. In this case, the goal is to predict salaries based on factors such as age, education level, capital gain, and hours worked per week.

- **Data Collection and Processing**: A dataset containing information about individuals, including their features and corresponding salaries, is gathered. The dataset is then processed to prepare it for machine learning tasks. This involves steps such as loading the data, summarizing its structure, and mapping categorical salary data to binary values (<=50K mapped to 0, >50K mapped to 1). Additionally, the dataset is segregated into feature (X) and target (Y) variables, with X representing the input features and Y representing the target variable (salary).

- **Machine Learning Pipeline**:
  - **Data Splitting**: The dataset is split into training and testing sets to train the model on a subset of the data and evaluate its performance on unseen data.
  - **Feature Scaling**: Feature scaling is applied to normalize the data and ensure that all features contribute equally to the prediction. This step helps improve the model's performance.
  - **Model Development**: The k-Nearest Neighbors (k-NN) algorithm is chosen as the predictive model. The optimal value of k is determined through an iterative process, where the model is trained and evaluated with different values of k. Once the optimal k value is found, the model is trained on the training data.
  - **Validation and Prediction**: The trained model is validated using the testing data to assess its performance. Predictions are made for new data points to estimate salaries based on the input features.

## Usage

1. Clone the repository: `gh repo clone Prabhakar-1/Salary_estimation>`
2. Open the Jupyter Notebook file (`Salary_Estimation.ipynb`) using Jupyter Notebook or Google Colab.
3. Run the code cells sequentially to load the dataset, preprocess the data, train the model, make predictions, and evaluate the model's performance.

## Requirements

The following Python libraries are required to run the code:
- Pandas
- NumPy
- scikit-learn
- Matplotlib

