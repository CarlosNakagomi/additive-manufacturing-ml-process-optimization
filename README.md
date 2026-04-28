# Additive Manufacturing ML Process Optimization

This project is a portfolio demonstration of a machine learning workflow for additive manufacturing process optimization.

The application simulates an L-PBF (Laser Powder Bed Fusion) process advisor that predicts whether a selected manufacturing parameter setup is likely to result in a successful build outcome or a risk of failure. It also provides rule-based explanations, process recommendations, and model-level visual insights.

## Project Overview

Additive manufacturing processes such as L-PBF depend heavily on the correct balance of process parameters, including laser power, scan speed, hatch spacing, layer thickness, beam diameter, and powder characteristics.

This project demonstrates how machine learning can be used as a decision-support tool to analyze process settings and provide interpretable recommendations for parameter adjustment.

The dataset used in this repository is synthetic/demo data created for portfolio purposes. It is designed to represent realistic additive manufacturing process features but is not a validated industrial production dataset.

## Key Features

- Streamlit portfolio application
- Machine learning-based build outcome prediction
- Process advisor for L-PBF parameter settings
- Success probability output
- Energy density calculation
- Risk-level interpretation
- Rule-based explanation details
- Optimization insight cards
- Model-level visualizations
  - feature importance
  - confusion matrix
  - ROC curve
- Clean dark-themed UI for portfolio presentation

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly / Matplotlib
- Machine learning classification models

## Repository Structure

```text
additive-manufacturing-ml-process-optimization/
├── app.py
├── assets/
├── data/
│   └── processed/
├── models/
├── outputs/
│   └── figures/
├── static/
├── .gitignore
└── README.md
