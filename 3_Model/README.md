# ESG Risk Classification Model (PyTorch + FinBERT)

## Overview
This folder contains the main deep learning model for ESG risk classification in the banking sector.

The model is developed using:
- PyTorch
- FinBERT (Financial BERT Transformer)
- Google Colab environment

It classifies financial news and text into ESG risk categories:
- Low ESG Risk
- Medium ESG Risk
- High ESG Risk

---

## Google Colab Notebook

The full implementation is available in Google Colab:

👉 ESG Model Notebook:
[Open in Google Colab](https://colab.research.google.com/drive/1fhzJA6PwbiN7lbNm_FIYp6W4qd1EcCXb#scrollTo=K7eQ2hVMquy4)

---

## How to Run

1. Open the Colab notebook using the link above
2. Click **Runtime → Run all**
3. Ensure GPU is enabled:
   - Runtime → Change runtime type → GPU
4. Dataset will be loaded from Kaggle or uploaded manually

---

## Model Architecture

- Base Model: FinBERT (Transformer)
- Framework: PyTorch
- Tokenization: HuggingFace Transformers
- Classifier Head: Fully Connected Layer

---

## Workflow

1. Load ESG / financial news dataset
2. Preprocess and clean text
3. Tokenize using FinBERT tokenizer
4. Train PyTorch model
5. Evaluate using F1-score and accuracy
6. Perform inference on new text

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- Weighted F1-Score
- Confusion Matrix

---

## Results

Baseline models (Logistic Regression, TF-IDF) are compared with deep learning models (FinBERT).

Expected outcome:
- Significant improvement using transformer-based model
- Better detection of ESG-related financial risks

---

## Business Relevance

This model can be used by banks for:
- ESG risk monitoring
- Counterparty risk assessment
- Reputation risk detection
- Regulatory compliance support

Relevant frameworks:
- Basel Committee guidelines
- ECB climate risk expectations
- EBA ESG risk integration

---

## Tech Stack

- Python
- PyTorch
- Transformers (HuggingFace)
- Scikit-learn
- Pandas
- Google Colab

---

## Future Improvements

- Add SHAP explainability
- Deploy Streamlit dashboard
- Add real-time ESG news ingestion
- Extend to multi-language ESG detection

---

## Author(s)

- Sajjad Haider
- Waseem Abbas

---

## Status
Work in progress 🚧
