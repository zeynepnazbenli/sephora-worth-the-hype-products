import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

DATA_PATH = "data/labeled_products.csv"
TARGET = "hype_label"
df = pd.read_csv("data/labeled_products.csv")

feature_cols = [
    "price_usd", "value_price_usd", "sale_price_usd",
    "sephora_exclusive", "limited_edition", "new", "online_only", "out_of_stock",
    "brand_name", "primary_category", "secondary_category", "tertiary_category"
]
feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y )

# 4) Preprocessing (with imputers)
numeric_features = [c for c in X.columns if X[c].dtype != "object"]
categorical_features = [c for c in X.columns if X[c].dtype == "object"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
#Logistic Regression
logreg = Pipeline(steps = [
    ("preprocess", preprocess),
    ("model",LogisticRegression(max_iter=2000, class_weight="balanced"))
])
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

print("\n===== Logistic Regression (Baseline) =====")
print(confusion_matrix(y_test, pred_lr))
print(classification_report(y_test, pred_lr, digits=3))

#Random Forest
rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    ))
])
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("\n===== Random Forest =====")
print(confusion_matrix(y_test, pred_rf))
print(classification_report(y_test, pred_rf, digits=3))

import joblib

joblib.dump(rf, "models/rf_hype_classifier.joblib")
joblib.dump(logreg, "models/logreg_hype_classifier.joblib")

print("Models saved.")









