# Bank-Deposit-Prediction-Model
A predictive model using machine learning to forecast bank term deposit subscriptions based on UCI dataset. Includes EDA, Random Forest modeling, and evaluation.
# Bank Deposit Prediction Model

## Overview
This project builds a machine learning model to predict if a bank client will subscribe to a term deposit based on marketing data. It uses the UCI Bank Marketing dataset.

## Purpose
To demonstrate data analysis and ML skills: handling imbalanced data, EDA, model training (Logistic Regression and Random Forest), and evaluation. Great for showcasing on resumes as an entry-level data science project.

## Methodology
1. **Data Loading**: Loaded `bank-full.csv` using pandas.
2. **EDA**: Summary stats, visualizations (histograms, boxplots, heatmaps) to spot imbalances and correlations (e.g., 'duration' is key).
3. **Preprocessing**: Encoded categories, scaled numbers, split data (80/20).
4. **Modeling**: Trained Logistic Regression (baseline) and Random Forest (main) with class weights for imbalance.
5. **Evaluation**: Used F1-score, ROC-AUC, confusion matrices. Results: Random Forest F1 ~0.60 for 'yes' class.
6. **Inference**: Predicted on sample data.

## How to Run
1. Clone the repo: `git clone (https://github.com/Jamilath/Bank-Deposit-Prediction-Model).git`
2. Install dependencies: `pip install -r requirements.txt`
3. Open in Jupyter: `jupyter notebook Bank_Deposit_Prediction.ipynb`
4. Run all cells.

## Files
- `Bank_Deposit_Prediction.ipynb`: Main notebook with code.
- `bank-full.csv`: Dataset (source: UCI Machine Learning Repository).
- `Bank_Deposit_Prediction.html`: Exported notebook for easy viewing.
- `requirements.txt`: List of Python libraries.
- `visualizations/`: Folder for images (e.g., confusion_matrix.png).

## Results
- Model Accuracy: ~89% (Random Forest).
- Key Insight: Longer call durations increase subscription likelihood.
- View full results in the notebook or HTML file.

## Technologies
- Python 3
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- Tools: Jupyter Notebook

For questions, contact me on LinkedIn:(https://www.linkedin.com/in/mohammed-jameel-2866028/)
