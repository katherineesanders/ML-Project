import numpy as np
import pandas as pd
import xgboost as xgb
 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
 

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target) 
 
print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts())
 

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
 

dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest  = xgb.DMatrix(X_test_scaled,  label=y_test)
 

params = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss", 
    "max_depth":        4, 
    "eta":              0.1, 
    "subsample":        0.8, 
    "colsample_bytree": 0.8,
    "seed":             42
}
 
evals = [(dtrain, "train"), (dtest, "eval")]
 
model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=20
)
 
print("\nBest iteration:", model.best_iteration)
 
y_prob = model.predict(dtest)
y_pred = (y_prob >= 0.5).astype(int)
 
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred,
                                         target_names=data.target_names))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
importance = model.get_score(importance_type="gain")
top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 Features by Gain:")
for feat, score in top_features:
    print(f"  {feat:<30} {score:.2f}")
 

model.save_model("xgboost_model.json")
loaded_model = xgb.Booster()
loaded_model.load_model("xgboost_model.json")