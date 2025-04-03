# BINARY-CLASSIFICATION-ON-CENCUS-INCOME-DATA-
Developed a binary classifier to predict income >$50K from census data using PySpark. Preprocessed data with feature engineering (StringIndexer, OneHotEncoder), trained Decision Tree, Random Forest, and Gradient Boosted Tree models. Demonstrated PySpark's scalability for ML workflows.

# Income Prediction with PySpark

![PySpark](https://img.shields.io/badge/PySpark-3.5.3-red)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)

A PySpark implementation for binary classification predicting whether an individual's income exceeds $50K/year based on census data.

## Features

- **Data Preprocessing**: Handles categorical and numerical features using PySpark's `StringIndexer`, `OneHotEncoder`, and `VectorAssembler`
- **Multiple Models**: Implements Decision Tree, Random Forest, and Gradient Boosted Tree classifiers
- **Hyperparameter Tuning**: Custom cross-validation for optimizing GBT parameters
- **Performance Evaluation**: AUC metric comparison across models

## Results

| Model                | AUC Score |
|----------------------|----------|
| Decision Tree        | 0.604    |
| Random Forest        | 0.891    |
| Gradient Boosted Tree| 0.906    |

**Best Model**: GBT with maxDepth=6, maxBins=60, maxIter=40

## Requirements

- PySpark 3.5.3
- Python 3.11+
- pandas
- numpy
- matplotlib

## Usage

1. Clone the repository
```bash
git clone https://github.com/yourusername/income-prediction-pyspark.git
cd income-prediction-pyspark

2. Run the Jupyter notebook
jupyter notebook BINARY_CLASSIFICATION_ON_CENCUS_INCOME_DATA.ipynb
