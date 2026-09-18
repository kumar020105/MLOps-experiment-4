\# MLOps Experiment 4 – MLflow Experiment Tracking and Model Management



\## Objective



This experiment demonstrates MLflow-based experiment tracking, multiple model training, model comparison, hyperparameter tuning, artifact logging, model versioning, and production model management using the Iris dataset.



\## Technologies Used



\- Python

\- MLflow

\- pandas

\- scikit-learn

\- matplotlib



\## Dataset



The Iris dataset from scikit-learn is used.



\- Samples: 150

\- Features: 4

\- Classes: 3

\- Training samples: 120

\- Testing samples: 30



\## Models



The experiment trains and tracks four classification models:



1\. Logistic Regression

2\. Decision Tree

3\. Random Forest

4\. Support Vector Machine



\## MLflow Tracking



The experiment logs:



\- Model parameters

\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Cross-validation accuracy

\- Trained models

\- Confusion matrix

\- Classification report

\- Predictions

\- Feature importance

\- Model summary



\## Hyperparameter Tuning



Random Forest hyperparameters are optimized using GridSearchCV with 5-fold cross-validation.



The tuned parameters are:



\- `n\_estimators`

\- `max\_depth`

\- `min\_samples\_split`



\## Model Registry



The trained models are registered in the MLflow Model Registry as:



`IrisClassifier`



Multiple model versions are created. The tuned Random Forest model is assigned the `production` alias.



\## Production Inference



The production model is loaded from the MLflow Model Registry and used for inference on unseen test samples.



The completed experiment achieved:



```text

Correct predictions: 5/5

