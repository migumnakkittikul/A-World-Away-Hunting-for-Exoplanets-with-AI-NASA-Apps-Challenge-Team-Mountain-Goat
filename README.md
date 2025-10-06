# 🌟 Exoplanet Detection System with XGBoost Ensemble

Interactive web platform for classifying exoplanet candidates using an ensemble of five XGBoost models. Achieves **87%+ accuracy** on NASA's Kepler mission data.

---

## 🚀 Quick Setup

### Prerequisites
- Python 3.7+

### Installation & Run

```bash
# 1. Navigate to project folder
cd nasa_webb

# 2. Install dependencies
pip install flask pandas numpy xgboost scikit-learn

# 3. Run the application
python app.py
```

Open browser: `http://localhost:5000`

---

## 🎯 What This Does

An **ensemble learning system** that combines 5 XGBoost models to classify Kepler observations as:
- **CONFIRMED** - Verified exoplanets
- **CANDIDATE** - Potential exoplanets
- **FALSE POSITIVE** - Non-planetary signals

**Key Features:**
- Comprehensive performance dashboard
- Single-sample predictions (only need 15 features instead of 88)
- Upload new data and retrain models
- Tune hyperparameters through web interface
- Automatic model backups and approval workflow

---

## 📊 Dashboard Page

View model performance with 10+ visualizations including confusion matrix, ROC curves, feature importance, error analysis, and cross-validation scores.

<img width="1706" height="4346" alt="Image" src="https://github.com/user-attachments/assets/3ada8be9-91ae-4b73-a7b9-179277164ed6" />

---

## 🔮 Prediction Page

Enter values for the **top 15 most important features** to get probability predictions. Remaining 73 features are automatically filled with median values.

<img width="1203" height="861" alt="Image" src="https://github.com/user-attachments/assets/ce5eb01b-7af8-4511-aa52-22279c8e29a1" />

<img width="972" height="762" alt="Image" src="https://github.com/user-attachments/assets/c80a318f-6b5f-476e-87c6-1266eea88318" />

---

## 🔄 Data Ingestion Page

Add new exoplanet data via **CSV upload** or **manual entry queue**. System trains new models and shows side-by-side comparison. You decide whether to accept or reject the new model.

**Safety features:**
- Automatic timestamped backups
- Model approval workflow (prevents automatic degradation)
- Data validation

<img width="1706" height="4906" alt="Image" src="https://github.com/user-attachments/assets/2ab45816-1df1-476b-a24b-d3d5492653ea" />

<img width="1037" height="286" alt="Image" src="https://github.com/user-attachments/assets/c1e1addb-c009-45f8-ba38-7a2d2ee33966" />

---

## ⚙️ Hyperparameter Tuning Page

Experiment with model configurations using **6 presets** (Default, Fast, Accurate, Regularized, Deep, Large Ensemble) or adjust parameters manually with sliders.

**Tunable parameters:**
- Number of models (1-10)
- Trees per model (50-1000)
- Learning rate, max depth, regularization, etc.

<img width="1706" height="1962" alt="Image" src="https://github.com/user-attachments/assets/dc5082ed-909c-401d-aab5-4eecd93f5a24" />

<img width="536" height="746" alt="Image" src="https://github.com/user-attachments/assets/1bbec3cd-62ac-4390-8f69-9262a338103d" />

---

## 🏗️ How It Works

**Ensemble Architecture:**
- 5 XGBoost models trained with different random seeds (42, 123, 456, 789, 999)
- **Soft voting:** Averages probability outputs from all models
- Produces more reliable predictions than single models

**Dataset:**
- 9,500+ Kepler Objects of Interest
- 88 features per sample (orbital period, radius, transit depth, stellar properties, etc.)
- Missing values handled via median imputation

**Tech Stack:**
- Backend: Flask
- ML: XGBoost, scikit-learn
- Data: Pandas, NumPy
- Visualization: Chart.js

---

## 🔍 Key Innovations

1. **Smart Feature Reduction** - Only 15/88 features needed for predictions (83% reduction)
2. **Queue System** - Batch multiple manual entries before training
3. **User Approval Workflow** - Review metrics before accepting new models
4. **Automatic Backups** - Timestamped snapshots before each retrain
5. **Preset Configurations** - Quick hyperparameter experimentation

---

## 📂 Project Structure

```
nasa_webb/
├── app.py                    # Flask backend (899 lines)
├── main/
│   ├── xgboost_model.json    # Trained ensemble model
│   ├── label_encoder.pkl     # Label encoder
│   ├── koi.csv               # Dataset (9,500+ samples)
│   └── xgboost_train.py      # Training script
├── templates/
│   ├── dashboard.html        # Main dashboard
│   ├── predict.html          # Prediction interface
│   ├── ingest.html           # Data ingestion
│   └── tune.html             # Hyperparameter tuning
├── uploads/                  # Temporary CSV storage
└── backups/                  # Model backups
```

---

## 📈 Performance

- **Accuracy:** 87.66%
- **F1-Score:** 0.87 (weighted)
- **AUC:** 0.96+ (one-vs-rest)
- **5-Fold CV:** Consistent across folds

---

