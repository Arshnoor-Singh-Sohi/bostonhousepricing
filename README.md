# Boston House Pricing Prediction

A linear regression model implementation to predict housing prices in Boston based on the famous Boston Housing Dataset.

## Project Overview

This project demonstrates a simple machine learning approach to predict housing prices in Boston using linear regression. The implementation explores the relationship between housing features (such as crime rate, number of rooms, etc.) and median home values, providing a practical example of regression analysis for real estate valuation.

## Dataset Description

The project uses the Boston Housing Dataset, which contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. This dataset has been widely used in machine learning literature as a standard benchmark.

### Features in the Dataset:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxide concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)² where Bk is the proportion of Black people by town
- **LSTAT**: % lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000s (Target variable)

## Project Files

- **Linear Regression ML Implementation.ipynb**: Jupyter notebook containing the complete implementation of the linear regression model, including data preprocessing, model training, and evaluation
- **regmodel.pkl**: Serialized trained model saved using pickle
- **requirements.txt**: List of Python dependencies required to run the project

## Implementation Details

The Jupyter notebook walks through several key steps in the machine learning pipeline:

1. **Data Loading and Exploration**:
   - Loading the Boston Housing Dataset
   - Examining basic statistics and feature distributions
   - Visualizing relationships between features and target variable

2. **Data Preprocessing**:
   - Handling any missing values (if present)
   - Scaling numerical features
   - Splitting data into training and testing sets

3. **Linear Regression Model**:
   - Implementation using scikit-learn
   - Training the model on the prepared dataset
   - Making predictions on test data

4. **Model Evaluation**:
   - Calculating metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score
   - Analyzing model coefficients to understand feature importance
   - Visualizing actual vs. predicted values

5. **Model Persistence**:
   - Saving the trained model to a pickle file for future use

## Requirements

To run this project, you'll need Python installed along with the following libraries:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

All dependencies are listed in the `requirements.txt` file and can be installed using:

```bash
pip install -r requirements.txt
```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Arshnoor-Singh-Sohi/bostonhousepricing.git
   cd bostonhousepricing
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook "Linear Regression ML Implementation.ipynb"
   ```

4. To use the saved model for predictions:
   ```python
   import pickle
   
   # Load the model
   with open('regmodel.pkl', 'rb') as file:
       model = pickle.load(file)
   
   # Make predictions (example)
   prediction = model.predict([[0.02, 15.0, 2.5, 0, 0.4, 6.5, 50.0, 4.5, 5, 350, 15.0, 380, 10.0]])
   print(f"Predicted house price: ${prediction[0] * 1000:.2f}")
   ```

## Results

The linear regression model demonstrates how various features of housing properties in Boston relate to their market values. By analyzing the coefficients of the model, we can determine which features have the most significant impact on housing prices in this dataset.

## License

This project is licensed under the terms of the LICENSE file included in the repository.

---

Created by [Arshnoor Singh Sohi](https://github.com/Arshnoor-Singh-Sohi)
