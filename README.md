# Sephora: Worth the Hype? 💄🧴  
### Machine Learning Classification & Streamlit Web App

## Live Demo
👉 https://sephora-worth-the-hype-appucts-gv8cbfw4avvtce6vfx5dyk.streamlit.app

This project is an end-to-end machine learning application that predicts whether a Sephora product is **worth the hype**, **underrated**, or **overrated** based on product features such as ratings, reviews, popularity, and pricing.

An interactive **Streamlit web application** is provided to explore products and get real-time predictions from a trained ML model.

---

##  Project Overview

- Extracted and engineered product-level features from a Sephora dataset
- Designed a **3-class classification problem**:
  - `worth_it`
  - `underrated`
  - `overrated`
- Built and evaluated multiple machine learning models
- Deployed the final model with an interactive web interface

---

## Machine Learning Pipeline

1. **Data Cleaning & Feature Engineering**
   - Removed missing / noisy entries
   - Engineered hype labels using rating and popularity percentiles

2. **Preprocessing**
   - Numerical feature scaling (StandardScaler)
   - Categorical feature encoding (OneHotEncoder)
   - Unified preprocessing with `ColumnTransformer`

3. **Model Training**
   - Logistic Regression (baseline)
   - Random Forest Classifier (final model)

4. **Evaluation**
   - Confusion Matrix
   - Precision, Recall, F1-score
   - Class-wise performance analysis

5. **Deployment**
   - Streamlit-based web application
   - Interactive product selection and prediction
   - Confidence score displayed for each prediction

---

## Model Performance (Random Forest)

- Overall accuracy: **~80%**
- Strong performance on `worth_it` and `overrated` classes
- Model outputs probabilistic confidence for interpretability

---

## Streamlit Web App Features

- Filter products by **brand** and **category**
- Select a product from the dataset
- View product details
- Get:
  - Predicted hype label
  - Model confidence
- Clean, minimal, soft-toned UI inspired by Sephora aesthetics

---

##  Project Structure
sephora-worth-the-hype-products/
│
├── app.py                    # Streamlit web app
├── train_classifier.py       # Model training & evaluation
├── label_engineering.py      # Label creation logic
├── requirements.txt
├── README.md
│
├── data/
│   ├── product_info.csv
│   └── labeled_products.csv
│
└── models/
├── logreg_hype_classifier.joblib
└── rf_hype_classifier.joblib

---

## ▶️ How to Run Locally

```bash
# Clone repository
git clone https://github.com/zeynepnazbenli/sephora-worth-the-hype-products.git
cd sephora-worth-the-hype-products

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

 Technologies Used
	•	Python
	•	Pandas, NumPy
	•	Scikit-learn
	•	Streamlit
	•	Joblib

⸻

 Notes
	•	This project focuses on data-driven hype prediction, not sentiment scraping or live API usage.
	•	Labels are engineered using statistical thresholds to simulate user perception.

⸻

Author

Zeynep Naz Benli