import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

df = pd.read_csv('koi.csv')
X = df.iloc[:, 3:]
y = df['koi_disposition']

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("training ensemble (5 models)...")
models = []

for seed in [42, 123, 456, 789, 999]:
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0
    )
    model.fit(X_train, y_train)
    models.append(model)

y_pred_proba = np.mean([m.predict_proba(X_test) for m in models], axis=0)
y_pred = np.argmax(y_pred_proba, axis=1)
accuracy = accuracy_score(y_test, y_pred)

print(f"\naccuracy: {accuracy*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\ntop 30 most important features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(30).iterrows():
    print(f"{row['feature']:30s} {row['importance']:.4f}")

# Save
model.save_model('xgboost_model.json')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nmodel saved.")