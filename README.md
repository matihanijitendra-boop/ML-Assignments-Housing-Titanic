# Machine Learning Assignments

This ZIP contains two complete beginner-friendly machine-learning assignments:

1. **Linear Regression – House Price Prediction**
2. **Logistic Regression – Titanic Survival Prediction**

Both projects use Python, pandas, scikit-learn and Matplotlib.

## 1. Install Python libraries

Open the project folder in VS Code. Open the terminal and run:

```bash
pip install -r requirements.txt
```

## 2. Download the datasets

The datasets are not bundled in this ZIP. Download the CSV files from Kaggle and place them in the correct `data` folders.

### House Prices
Dataset: [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

Download `train.csv` and place it here:

`housing_price/data/train.csv`

### Titanic
Dataset: [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

Download `train.csv` and place it here:

`titanic_survival/data/train.csv`

## 3. Run the assignments

From the main project folder:

```bash
python housing_price/house_price_prediction.py
```

Then run:

```bash
python titanic_survival/titanic_survival_prediction.py
```

You can also run each script from its own folder:

```bash
python house_price_prediction.py
python titanic_survival_prediction.py
```

## Assignment 1 / Mini Project 2: House Price Prediction

**Objective:** Train a Linear Regression model to predict house sale prices.

Tasks completed by the script:
- Load and inspect the housing dataset.
- Separate input features and target (`SalePrice`).
- Handle missing values and categorical columns.
- Split the data into training and testing sets.
- Train a Linear Regression model.
- Predict house prices.
- Calculate the R² score and mean absolute error.
- Plot predicted values against actual values.
- Save the trained model as `house_price_model.joblib`.

## Assignment 2: Titanic Survival Prediction

**Objective:** Train a Logistic Regression model to predict whether a passenger survived.

Tasks completed by the script:
- Load and inspect the Titanic dataset.
- Use passenger details as input features and `Survived` as the target.
- Handle missing values and categorical columns.
- Split the data into training and testing sets.
- Train a Logistic Regression model.
- Predict survival and calculate accuracy.
- Display a classification report and confusion matrix.
- Save the trained model as `titanic_survival_model.joblib`.

## Important

- Use the Kaggle **training** CSV file named `train.csv` for each project.
- Keep the original column names in the CSV files.
- The scripts use a fixed random seed so results are reproducible.
- Model scores depend on the dataset and train/test split.
