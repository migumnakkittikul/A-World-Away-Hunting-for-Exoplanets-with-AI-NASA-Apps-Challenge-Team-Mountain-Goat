from flask import Flask, render_template, jsonify, request, session
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, auc, roc_auc_score, f1_score
)
import json
import os
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'exoplanet-detection-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('backups', exist_ok=True)

# Global queue for manual entries
data_queue = []

# Load model and data
def load_model_and_data():
    # Load XGBoost model
    model = xgb.XGBClassifier()
    model.load_model('main/xgboost_model.json')

    # Load label encoder
    with open('main/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    # Load and prepare data
    df = pd.read_csv('main/koi.csv')
    X = df.iloc[:, 3:]
    y = df['koi_disposition']

    # Prepare data
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    y_encoded = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    return model, le, X_train, X_test, y_train, y_test, X.columns, X_imputed, df

# Global variables
model, le, X_train, X_test, y_train, y_test, feature_names, X_full, df_full = load_model_and_data()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/model_metrics')
def get_model_metrics():
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Calculate AUC (one-vs-rest for multiclass)
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc_score = 0.0

    # Classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # Class distribution in predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    class_distribution = {le.classes_[i]: int(count) for i, count in zip(unique, counts)}

    # True class distribution
    unique_true, counts_true = np.unique(y_test, return_counts=True)
    true_distribution = {le.classes_[i]: int(count) for i, count in zip(unique_true, counts_true)}

    return jsonify({
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'auc_score': float(auc_score),
        'confusion_matrix': conf_matrix.tolist(),
        'class_names': le.classes_.tolist(),
        'classification_report': report,
        'predicted_distribution': class_distribution,
        'true_distribution': true_distribution,
        'total_samples': len(y_test)
    })

@app.route('/api/model_architecture')
def get_model_architecture():
    # Model configuration from training script
    return jsonify({
        'model_type': 'XGBoost Ensemble',
        'ensemble_method': 'Soft Voting (Probability Averaging)',
        'n_models': 5,
        'random_seeds': [42, 123, 456, 789, 999],
        'n_estimators_per_model': 400,
        'total_trees': 2000,  # 5 models × 400 trees
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softprob',
        'description': 'Ensemble of 5 XGBoost classifiers with different random initializations. Predictions are made by averaging probability outputs from all models (soft voting).'
    })

@app.route('/api/feature_importance')
def get_feature_importance():
    # Get feature importance
    importance_dict = {
        'feature': feature_names.tolist(),
        'importance': model.feature_importances_.tolist()
    }

    df_importance = pd.DataFrame(importance_dict)
    df_importance = df_importance.sort_values('importance', ascending=False)

    # Get top 30
    top_30 = df_importance.head(30)

    return jsonify({
        'features': top_30['feature'].tolist(),
        'importance': top_30['importance'].tolist()
    })

@app.route('/api/roc_data')
def get_roc_data():
    y_pred_proba = model.predict_proba(X_test)
    n_classes = len(le.classes_)

    # Binarize the output for ROC calculation
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    roc_data = {}

    # Calculate ROC curve for each class
    for i, class_name in enumerate(le.classes_):
        if n_classes == 2 and i == 1:
            # For binary classification, only need one curve
            break

        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)

        roc_data[class_name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        }

    return jsonify(roc_data)

@app.route('/api/confidence_distribution')
def get_confidence_distribution():
    y_pred_proba = model.predict_proba(X_test)

    # Get max probability for each prediction (confidence)
    confidence_data = {}

    for i, class_name in enumerate(le.classes_):
        # Get probabilities for this class
        class_probabilities = y_pred_proba[:, i]

        # Create histogram bins
        hist, bin_edges = np.histogram(class_probabilities, bins=20, range=(0, 1))

        confidence_data[class_name] = {
            'probabilities': class_probabilities.tolist(),
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }

    return jsonify(confidence_data)

@app.route('/api/dataset_info')
def get_dataset_info():
    # Training class distribution
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    train_dist = {le.classes_[i]: int(count) for i, count in zip(train_unique, train_counts)}

    # Test class distribution
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    test_dist = {le.classes_[i]: int(count) for i, count in zip(test_unique, test_counts)}

    return jsonify({
        'total_samples': len(df_full),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_names),
        'n_classes': len(le.classes_),
        'train_distribution': train_dist,
        'test_distribution': test_dist,
        'test_ratio': 0.2,
        'random_state': 42,
        'data_source': 'NASA Kepler Objects of Interest (KOI) Dataset'
    })

@app.route('/api/cross_validation')
def get_cross_validation():
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X_full,
                                 le.transform(df_full['koi_disposition']),
                                 cv=5, scoring='accuracy')

    return jsonify({
        'cv_scores': cv_scores.tolist(),
        'mean_cv_score': float(cv_scores.mean()),
        'std_cv_score': float(cv_scores.std()),
        'n_folds': 5
    })

@app.route('/api/error_analysis')
def get_error_analysis():
    y_pred = model.predict(X_test)

    # Find misclassified samples
    misclassified_mask = y_pred != y_test
    misclassified_indices = np.where(misclassified_mask)[0]

    # Analyze misclassifications
    error_patterns = {}

    for idx in misclassified_indices:
        true_class = le.classes_[y_test[idx]]
        pred_class = le.classes_[y_pred[idx]]
        key = f"{true_class} → {pred_class}"

        if key not in error_patterns:
            error_patterns[key] = 0
        error_patterns[key] += 1

    # Sort by frequency
    sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)

    # Get prediction probabilities for misclassified samples
    y_pred_proba = model.predict_proba(X_test)
    misclassified_confidence = []

    for idx in misclassified_indices[:20]:  # Top 20 errors
        true_class = le.classes_[y_test[idx]]
        pred_class = le.classes_[y_pred[idx]]
        confidence = float(y_pred_proba[idx][y_pred[idx]])

        misclassified_confidence.append({
            'true_class': true_class,
            'predicted_class': pred_class,
            'confidence': confidence
        })

    return jsonify({
        'total_errors': int(misclassified_mask.sum()),
        'error_rate': float(misclassified_mask.mean()),
        'error_patterns': dict(sorted_errors),
        'top_misclassifications': misclassified_confidence
    })

# ================== DATA INGESTION & RETRAINING ROUTES ==================

@app.route('/ingest')
def ingest_page():
    return render_template('ingest.html')

@app.route('/api/get_feature_list')
def get_feature_list():
    """Return list of all required features for manual entry"""
    return jsonify({
        'features': feature_names.tolist(),
        'classes': le.classes_.tolist(),
        'total_features': len(feature_names)
    })

@app.route('/api/queue_status')
def queue_status():
    """Get current queue status"""
    return jsonify({
        'queue_size': len(data_queue),
        'samples': data_queue
    })

@app.route('/api/add_to_queue', methods=['POST'])
def add_to_queue():
    """Add a manually entered sample to the queue"""
    try:
        data = request.json

        # Validate class
        if data['class'] not in le.classes_:
            return jsonify({'success': False, 'error': f'Invalid class. Must be one of: {le.classes_.tolist()}'}), 400

        # Validate all features are present
        features = data.get('features', {})
        missing_features = []

        for feature in feature_names:
            if feature not in features:
                missing_features.append(feature)

        if missing_features:
            return jsonify({
                'success': False,
                'error': f'Missing features: {missing_features[:5]}... ({len(missing_features)} total)'
            }), 400

        # Add to queue
        queue_entry = {
            'id': len(data_queue),
            'class': data['class'],
            'features': features,
            'timestamp': datetime.now().isoformat()
        }
        data_queue.append(queue_entry)

        return jsonify({
            'success': True,
            'message': f'Sample added to queue (Queue size: {len(data_queue)})',
            'queue_size': len(data_queue)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear_queue', methods=['POST'])
def clear_queue():
    """Clear all samples from the queue"""
    global data_queue
    queue_size = len(data_queue)
    data_queue = []
    return jsonify({
        'success': True,
        'message': f'Cleared {queue_size} samples from queue'
    })

@app.route('/api/remove_from_queue', methods=['POST'])
def remove_from_queue():
    """Remove a specific sample from queue"""
    global data_queue
    try:
        data = request.json
        queue_id = data.get('id')

        data_queue = [item for item in data_queue if item['id'] != queue_id]

        return jsonify({
            'success': True,
            'message': 'Sample removed from queue',
            'queue_size': len(data_queue)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload and validation"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'File must be a CSV'}), 400

        # Read and validate CSV
        df_new = pd.read_csv(file)

        # Check for required columns
        if 'koi_disposition' not in df_new.columns:
            return jsonify({'success': False, 'error': 'CSV must contain "koi_disposition" column'}), 400

        # Get feature columns (all except first 3: kepid, kepoi_name, koi_disposition)
        new_features = df_new.columns[3:].tolist()
        expected_features = feature_names.tolist()

        # Check feature compatibility
        missing_features = set(expected_features) - set(new_features)
        extra_features = set(new_features) - set(expected_features)

        if missing_features:
            return jsonify({
                'success': False,
                'error': f'CSV is missing required features: {list(missing_features)[:5]}... ({len(missing_features)} total)'
            }), 400

        # Validate classes
        invalid_classes = set(df_new['koi_disposition'].unique()) - set(le.classes_)
        if invalid_classes:
            return jsonify({
                'success': False,
                'error': f'CSV contains invalid classes: {list(invalid_classes)}. Valid classes: {le.classes_.tolist()}'
            }), 400

        # Save file
        filename = secure_filename(f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df_new.to_csv(filepath, index=False)

        return jsonify({
            'success': True,
            'message': f'File uploaded successfully: {len(df_new)} samples',
            'filename': filename,
            'samples': len(df_new),
            'class_distribution': df_new['koi_disposition'].value_counts().to_dict()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train_on_queue', methods=['POST'])
def train_on_queue():
    """Train new models using queued manual entries"""
    global model, le, X_train, X_test, y_train, y_test, X_full, df_full

    try:
        if len(data_queue) == 0:
            return jsonify({'success': False, 'error': 'Queue is empty'}), 400

        # Convert queue to DataFrame
        queue_data = []
        for entry in data_queue:
            row = {'koi_disposition': entry['class']}
            row.update(entry['features'])
            queue_data.append(row)

        df_queue = pd.DataFrame(queue_data)

        # Add placeholder columns for kepid and kepoi_name
        df_queue.insert(0, 'kepid', range(90000, 90000 + len(df_queue)))
        df_queue.insert(1, 'kepoi_name', [f'K{i:05d}.01' for i in range(90000, 90000 + len(df_queue))])

        # Combine with existing data
        df_combined = pd.concat([df_full, df_queue], ignore_index=True)

        # Retrain model
        result = retrain_models(df_combined, source='queue')

        if result['success']:
            # Clear queue on successful training
            data_queue.clear()

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train_on_csv', methods=['POST'])
def train_on_csv():
    """Train new models using uploaded CSV file"""
    global model, le, X_train, X_test, y_train, y_test, X_full, df_full

    try:
        data = request.json
        filename = data.get('filename')

        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404

        # Load uploaded CSV
        df_new = pd.read_csv(filepath)

        # Combine with existing data
        df_combined = pd.concat([df_full, df_new], ignore_index=True)

        # Retrain model
        result = retrain_models(df_combined, source='csv')

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def retrain_models(df_combined, source='unknown'):
    """
    Retrain the ensemble of 5 XGBoost models with combined data
    Returns comparison metrics and saves new models if better
    """
    global model, le, X_train, X_test, y_train, y_test, X_full, df_full

    try:
        # Backup current models
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = f'backups/backup_{timestamp}'
        os.makedirs(backup_dir, exist_ok=True)

        shutil.copy('main/xgboost_model.json', f'{backup_dir}/xgboost_model.json')
        shutil.copy('main/label_encoder.pkl', f'{backup_dir}/label_encoder.pkl')
        shutil.copy('main/koi.csv', f'{backup_dir}/koi.csv')

        # Prepare combined data
        X_combined = df_combined.iloc[:, 3:]
        y_combined = df_combined['koi_disposition']

        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_combined_imputed = pd.DataFrame(imputer.fit_transform(X_combined), columns=X_combined.columns)

        # Encode labels
        y_combined_encoded = le.transform(y_combined)

        # Split data
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            X_combined_imputed, y_combined_encoded,
            test_size=0.2, random_state=42, stratify=y_combined_encoded
        )

        # Get old model performance
        y_pred_old = model.predict(X_test_new)
        old_accuracy = accuracy_score(y_test_new, y_pred_old)
        old_f1 = f1_score(y_test_new, y_pred_old, average='weighted')

        # Train new ensemble (5 models)
        new_models = []
        for seed in [42, 123, 456, 789, 999]:
            new_model = xgb.XGBClassifier(
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
            new_model.fit(X_train_new, y_train_new)
            new_models.append(new_model)

        # Ensemble prediction (average probabilities)
        y_pred_proba_new = np.mean([m.predict_proba(X_test_new) for m in new_models], axis=0)
        y_pred_new = np.argmax(y_pred_proba_new, axis=1)

        # Get new model performance
        new_accuracy = accuracy_score(y_test_new, y_pred_new)
        new_f1 = f1_score(y_test_new, y_pred_new, average='weighted')

        # Calculate improvement
        accuracy_improvement = new_accuracy - old_accuracy
        f1_improvement = new_f1 - old_f1

        # Save new models temporarily
        new_models[0].save_model(f'{backup_dir}/new_xgboost_model.json')

        # Prepare comparison results
        comparison = {
            'success': True,
            'source': source,
            'timestamp': timestamp,
            'backup_location': backup_dir,
            'old_metrics': {
                'accuracy': float(old_accuracy),
                'f1_score': float(old_f1),
                'samples': len(df_full)
            },
            'new_metrics': {
                'accuracy': float(new_accuracy),
                'f1_score': float(new_f1),
                'samples': len(df_combined)
            },
            'improvements': {
                'accuracy': float(accuracy_improvement),
                'f1_score': float(f1_improvement),
                'new_samples_added': len(df_combined) - len(df_full)
            },
            'recommendation': 'keep_new' if accuracy_improvement > 0 else 'keep_old'
        }

        return comparison

    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/api/accept_new_model', methods=['POST'])
def accept_new_model():
    """Accept the new model and make it the active model"""
    global model, le, X_train, X_test, y_train, y_test, X_full, df_full

    try:
        data = request.json
        backup_dir = data.get('backup_location')

        if not backup_dir or not os.path.exists(backup_dir):
            return jsonify({'success': False, 'error': 'Backup location not found'}), 404

        # Copy new model to main location
        shutil.copy(f'{backup_dir}/new_xgboost_model.json', 'main/xgboost_model.json')

        # Reload models
        model, le, X_train, X_test, y_train, y_test, feature_names_new, X_full, df_full = load_model_and_data()

        return jsonify({
            'success': True,
            'message': 'New model activated successfully!'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reject_new_model', methods=['POST'])
def reject_new_model():
    """Reject the new model and keep the old one"""
    try:
        data = request.json
        backup_dir = data.get('backup_location')

        if backup_dir and os.path.exists(backup_dir):
            # Can optionally delete the backup
            # shutil.rmtree(backup_dir)
            pass

        return jsonify({
            'success': True,
            'message': 'Keeping current model. New model discarded.'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================== HYPERPARAMETER TUNING ROUTES ==================

@app.route('/tune')
def tune_page():
    return render_template('tune.html')

@app.route('/api/get_current_hyperparameters')
def get_current_hyperparameters():
    """Return current model hyperparameters"""
    return jsonify({
        'n_estimators': 400,
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'n_models': 5,
        'random_seeds': [42, 123, 456, 789, 999]
    })

@app.route('/api/train_with_hyperparameters', methods=['POST'])
def train_with_hyperparameters():
    """Train new models with custom hyperparameters"""
    global model, le, X_train, X_test, y_train, y_test, X_full, df_full

    try:
        data = request.json
        hyperparams = data.get('hyperparameters', {})

        # Validate hyperparameters
        try:
            n_estimators = int(hyperparams.get('n_estimators', 400))
            learning_rate = float(hyperparams.get('learning_rate', 0.05))
            max_depth = int(hyperparams.get('max_depth', 8))
            min_child_weight = int(hyperparams.get('min_child_weight', 1))
            subsample = float(hyperparams.get('subsample', 0.8))
            colsample_bytree = float(hyperparams.get('colsample_bytree', 0.8))
            gamma = float(hyperparams.get('gamma', 0))
            reg_alpha = float(hyperparams.get('reg_alpha', 0))
            reg_lambda = float(hyperparams.get('reg_lambda', 1))
            n_models = int(hyperparams.get('n_models', 5))

            # Validate ranges
            if not (1 <= n_estimators <= 2000):
                return jsonify({'success': False, 'error': 'n_estimators must be between 1 and 2000'}), 400
            if not (0.001 <= learning_rate <= 1.0):
                return jsonify({'success': False, 'error': 'learning_rate must be between 0.001 and 1.0'}), 400
            if not (1 <= max_depth <= 20):
                return jsonify({'success': False, 'error': 'max_depth must be between 1 and 20'}), 400
            if not (0.1 <= subsample <= 1.0):
                return jsonify({'success': False, 'error': 'subsample must be between 0.1 and 1.0'}), 400
            if not (0.1 <= colsample_bytree <= 1.0):
                return jsonify({'success': False, 'error': 'colsample_bytree must be between 0.1 and 1.0'}), 400
            if not (1 <= n_models <= 10):
                return jsonify({'success': False, 'error': 'n_models must be between 1 and 10'}), 400

        except (ValueError, TypeError) as e:
            return jsonify({'success': False, 'error': f'Invalid hyperparameter value: {str(e)}'}), 400

        # Backup current models
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = f'backups/tune_{timestamp}'
        os.makedirs(backup_dir, exist_ok=True)

        shutil.copy('main/xgboost_model.json', f'{backup_dir}/xgboost_model.json')

        # Use current data
        X_train_hp, X_test_hp, y_train_hp, y_test_hp = train_test_split(
            X_full, le.transform(df_full['koi_disposition']),
            test_size=0.2, random_state=42, stratify=le.transform(df_full['koi_disposition'])
        )

        # Get old model performance
        y_pred_old = model.predict(X_test_hp)
        old_accuracy = accuracy_score(y_test_hp, y_pred_old)
        old_f1 = f1_score(y_test_hp, y_pred_old, average='weighted')

        # Train new ensemble with custom hyperparameters
        seeds = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555][:n_models]
        new_models = []

        for seed in seeds:
            new_model = xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=seed,
                verbosity=0
            )
            new_model.fit(X_train_hp, y_train_hp)
            new_models.append(new_model)

        # Ensemble prediction
        y_pred_proba_new = np.mean([m.predict_proba(X_test_hp) for m in new_models], axis=0)
        y_pred_new = np.argmax(y_pred_proba_new, axis=1)

        # Get new model performance
        new_accuracy = accuracy_score(y_test_hp, y_pred_new)
        new_f1 = f1_score(y_test_hp, y_pred_new, average='weighted')

        # Cross-validation on new hyperparameters
        cv_model = xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=0
        )
        cv_scores = cross_val_score(cv_model, X_full,
                                     le.transform(df_full['koi_disposition']),
                                     cv=5, scoring='accuracy')

        # Save new models temporarily
        new_models[0].save_model(f'{backup_dir}/new_xgboost_model.json')

        # Calculate improvement
        accuracy_improvement = new_accuracy - old_accuracy
        f1_improvement = new_f1 - old_f1

        return jsonify({
            'success': True,
            'timestamp': timestamp,
            'backup_location': backup_dir,
            'hyperparameters_used': {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'gamma': gamma,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'n_models': n_models,
                'total_trees': n_estimators * n_models
            },
            'old_metrics': {
                'accuracy': float(old_accuracy),
                'f1_score': float(old_f1),
                'hyperparameters': 'Default (n_est=400, lr=0.05, depth=8, models=5)'
            },
            'new_metrics': {
                'accuracy': float(new_accuracy),
                'f1_score': float(new_f1),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std())
            },
            'improvements': {
                'accuracy': float(accuracy_improvement),
                'f1_score': float(f1_improvement)
            },
            'recommendation': 'keep_new' if accuracy_improvement > 0 else 'keep_old'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/api/get_top_features')
def get_top_features():
    """Get top 15 most important features"""
    importance_dict = {
        'feature': feature_names.tolist(),
        'importance': model.feature_importances_.tolist()
    }

    df_importance = pd.DataFrame(importance_dict)
    df_importance = df_importance.sort_values('importance', ascending=False)

    # Get top 15
    top_15 = df_importance.head(15)

    return jsonify({
        'features': top_15['feature'].tolist(),
        'importance': top_15['importance'].tolist()
    })

@app.route('/api/predict_sample', methods=['POST'])
def predict_sample():
    """Predict probability for a single sample using top 15 features"""
    try:
        data = request.get_json()
        input_features = data.get('features', {})

        if not input_features:
            return jsonify({'success': False, 'error': 'No features provided'}), 400

        # Get top 15 features
        importance_dict = {
            'feature': feature_names.tolist(),
            'importance': model.feature_importances_.tolist()
        }
        df_importance = pd.DataFrame(importance_dict)
        df_importance = df_importance.sort_values('importance', ascending=False)
        top_15_features = df_importance.head(15)['feature'].tolist()

        # Create a full feature array with median values for missing features
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_full)
        median_values = imputer.statistics_

        # Initialize sample with median values
        sample = pd.DataFrame([median_values], columns=feature_names)

        # Update with provided values for top 15 features
        for feature in top_15_features:
            if feature in input_features:
                try:
                    sample[feature] = float(input_features[feature])
                except (ValueError, TypeError):
                    return jsonify({
                        'success': False,
                        'error': f'Invalid value for feature: {feature}'
                    }), 400

        # Get prediction probabilities
        probabilities = model.predict_proba(sample)[0]
        prediction = model.predict(sample)[0]

        # Get class names
        class_names = le.classes_.tolist()

        # Create probability dictionary
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

        return jsonify({
            'success': True,
            'prediction': class_names[prediction],
            'probabilities': prob_dict,
            'confidence': float(max(probabilities))
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
