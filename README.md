# Data Science & Machine Learning in Python

Notebooks and exercises from a Python for Data Science and Machine Learning course. They cover the core workflow end to end: exploring and cleaning data, feature engineering, training and evaluating supervised and unsupervised models, and serving a trained model through an API.

**Stack:** Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Plotly · PyCaret · FastAPI

## What's covered

| Topic | Notebook(s) | Techniques |
|---|---|---|
| Python, NumPy & pandas fundamentals | `unit2.ipynb` | Slicing, array operations, DataFrames, descriptive statistics |
| Exploratory data analysis & visualisation | `mod2ex.ipynb`, `Guide - Exploring Data with Pandas and Matplot.ipynb` | Distributions, correlations and plotting with Matplotlib and Seaborn |
| Statistics | `day2.ipynb` | Two-sample t-test with `scipy.stats` |
| Data preparation & feature engineering | `mod4ex.ipynb` | Cleaning, train/test split, one-hot encoding, scaling |
| Linear regression | `day3mod5.ipynb`, `Linear Regression - Singapore Real Estate .ipynb` | `LinearRegression`, feature scaling, MAE / MSE / R² |
| Classification | `day3mod6.ipynb`, `Guide - Classification modelling - Iris Species.ipynb` | Logistic regression, decision trees, random forests, k-NN; confusion matrices, classification reports, ROC curves |
| Clustering | `day4mod8.ipynb`, `Guide-Clustering-KMeans.ipynb`, `PyCaret - Clustering Demo.ipynb` | k-means, choosing *k* with silhouette scores, PyCaret clustering |
| Dimensionality reduction | `PCA.ipynb`, `Foundational_Dimensionality reduction demo.ipynb` | PCA from scratch and with scikit-learn, explained variance, kernel PCA, t-SNE, PCA + logistic regression pipeline tuned with `GridSearchCV` |
| Model deployment | `QADSMLP-learner/10_Deploying Models & Insights/` | Random forest price model saved with `joblib` and served through a FastAPI `/predict` endpoint |

Notebooks whose names start with **Guide** or end in **Demo** are course reference material. The module exercises (`unit2`, `mod*`, `day*`) are my own work.

## Datasets

The CSV files aren't committed (see `.gitignore`). Some notebooks use scikit-learn's built-in datasets (Iris, Wine, Digits). The others expect their CSVs in `QADSMLP-learner/data/`:

`housing_data.csv`, `loan_data.csv`, `renfe_trains.csv`, `Real estate.csv`, `mortgage_applicants.csv`, `train_viz.csv`, `titanic.csv`, `iris.csv`, `experimental_investment_performance.csv`

## Running locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install jupyter pandas numpy scipy scikit-learn matplotlib seaborn plotly
jupyter notebook
```

To try the model API:

```bash
pip install fastapi uvicorn joblib
cd "QADSMLP-learner/10_Deploying Models & Insights"
python retrain_model.py         # trains and saves the model
uvicorn main:app --reload
# then open http://127.0.0.1:8000/predict/1?ds=BARCELONA&vc=Turista&fr=Promo
```

PyCaret is optional and only needed for the PyCaret demo. Some reference notebooks were written for older scikit-learn versions and may need small import changes.
