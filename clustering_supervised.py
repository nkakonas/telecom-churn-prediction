from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    balanced_accuracy_score, average_precision_score, brier_score_loss,
   log_loss, matthews_corrcoef
)
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



def preprocess_data(df):
    """
    Clean and preprocess the Telco churn dataset for modeling.
    - Removes metadata and irrelevant fields.
    - Prevents label leakage by removing any derived target columns.
    - Keeps cluster-based features only for models that include them.
    """
    
    # Create a working copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # -------------------------------------------------------------
    # 1) Remove metadata and unnecessary columns
    # -------------------------------------------------------------
    cols_to_drop = [
        'customer_id', 'count', 'lat_long', 'churn_label', 
        'churn_score', 'churn_reason', 'id'
    ]
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    
    # -------------------------------------------------------------
    # 2) Prevent Target Leakage
    #    - Only `churn_value` should be the true label.
    #    - Any other column derived from the churn label must be removed.
    #    - These include: churn_label_bin, churn_category, churn_flag, etc.
    # -------------------------------------------------------------
    leak_cols = [
        c for c in data.columns
        if c.startswith("churn_") and c != "churn_value"
    ]
    
    if leak_cols:
        print(f"[Leakage protection] Dropping: {leak_cols}")
        data = data.drop(columns=leak_cols, errors="ignore")
    
    # NOTE:
    # We do NOT remove clustering features.
    # - In `supervised.py`, they don't exist → baseline model.
    # - In `clustering_supervised.py`, they remain → enhanced model.
    
    # -------------------------------------------------------------
    # 3) Clean numeric columns (handle formatting / missing values)
    # -------------------------------------------------------------
    if 'total_charges' in data.columns:
        # Convert string values to numeric and handle missing values
        data['total_charges'] = pd.to_numeric(data['total_charges'], errors='coerce')
        data['total_charges'].fillna(data['total_charges'].median(), inplace=True)
    
    # -------------------------------------------------------------
    # 4) Create useful engineered features
    # -------------------------------------------------------------
    if 'monthly_charges' in data.columns and 'tenure_months' in data.columns:
        data['avg_monthly_charges'] = data['total_charges'] / (data['tenure_months'] + 1)
        data['charges_per_tenure'] = data['monthly_charges'] / (data['tenure_months'] + 1)
    
    # -------------------------------------------------------------
    # 5) Encode categorical variables using Label Encoding
    #    - Only applied to object dtype columns that are not the label.
    # -------------------------------------------------------------
    categorical_cols = data.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'churn_value']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    return data, label_encoders


def select_features_comparison(X, y, n_features=15):
    """
    Compare different feature selection methods
    """
    results = {}
    
    # Method 1: SelectKBest with f_classif
    selector_f = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
    selector_f.fit(X, y)
    scores_f = pd.DataFrame({
        'feature': X.columns,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)
    results['f_classif'] = scores_f.head(n_features)['feature'].tolist()
    
    # Method 2: SelectKBest with mutual_info
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X.shape[1]))
    selector_mi.fit(X, y)
    scores_mi = pd.DataFrame({
        'feature': X.columns,
        'mi_score': selector_mi.scores_
    }).sort_values('mi_score', ascending=False)
    results['mutual_info'] = scores_mi.head(n_features)['feature'].tolist()
    
    # Method 3: Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    results['random_forest'] = importance_df.head(n_features)['feature'].tolist()
    
    # Get consensus features (appearing in at least 2 methods)
    all_features = results['f_classif'] + results['mutual_info'] + results['random_forest']
    feature_counts = pd.Series(all_features).value_counts()
    consensus_features = feature_counts[feature_counts >= 2].index.tolist()
    
    # If not enough consensus, take top from RF (usually most reliable)
    if len(consensus_features) < n_features:
        consensus_features = results['random_forest'][:n_features]
    
    print("\n=== Feature Selection Results ===")
    print(f"\nTop {n_features} features by F-statistic:")
    print(scores_f.head(n_features))
    print(f"\nTop {n_features} features by Mutual Information:")
    print(scores_mi.head(n_features))
    print(f"\nTop {n_features} features by Random Forest Importance:")
    print(importance_df.head(n_features))
    print(f"\nConsensus features selected: {len(consensus_features)}")
    
    return consensus_features, importance_df


def compare_models(X_train, y_train):
    """
    Compare multiple classification models
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
    }
    
    results = {}
    
    print("\n=== Model Comparison ===")
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"\n{name}:")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Select best model based on CV score
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    print(f"\n*** Best Model: {best_model_name} ***")
    
    return results, best_model_name


def tune_best_model(X_train, y_train, model_name):
    """
    Hyperparameter tuning for the best model
    """
    print(f"\n=== Tuning {model_name} ===")
    
    if model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', None]
        }
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    elif model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        }
        model = GradientBoostingClassifier(random_state=42)

    elif model_name == 'CatBoost':
        param_grid = {
            'iterations': [100, 200],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1],
            'l2_leaf_reg': [1, 3, 5]
        }
        model = CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
    
    else:  # Logistic Regression
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'class_weight': ['balanced', None]
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def plot_results(y_test, y_pred, y_pred_proba, feature_importance, top_n=15):
    """
    Visualize model results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature Importance
    top_features = feature_importance.head(top_n)
    axes[1, 0].barh(range(len(top_features)), top_features['importance'])
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels(top_features['feature'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title(f'Top {top_n} Feature Importances')
    axes[1, 0].invert_yaxis()
    
    # Prediction Distribution
    axes[1, 1].hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
                    bins=30, label=['No Churn', 'Churn'], alpha=0.7)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main(df):
    """
    Main pipeline for churn prediction
    """
    print("=== Customer Churn Prediction Pipeline ===\n")
    
    # Preprocess data
    print("Step 1: Preprocessing data...")
    data, label_encoders = preprocess_data(df)
    
    # Separate features and target
    X = data.drop('churn_value', axis=1)
    y = data['churn_value']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Churn rate: {y.mean():.2%}")
    
    # Split data into train,test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature selection
    print("\nStep 2: Feature selection...")
    selected_features, feature_importance = select_features_comparison(X_train, y_train, n_features=15)
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Scale features
    print("\nStep 3: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)
    
    # Compare models
    print("\nStep 4: Comparing models...")
    model_results, best_model_name = compare_models(
        X_train_scaled,  y_train
    )
    
    # Tune best model
    print("\nStep 5: Hyperparameter tuning...")
    best_model = tune_best_model(X_train_scaled, y_train, best_model_name)
    
    # Final predictions
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluation
    print("\n=== Final Model Evaluation ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    # Existing: ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    # Added metrics (top 5 useful + PR-AUC)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred_proba)  # PR AUC / Average Precision
    brier = brier_score_loss(y_test, y_pred_proba)
    ll = log_loss(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print("\nAdditional metrics:")
    print(f"  PR AUC (Average Precision): {pr_auc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  Matthews Corr Coef (MCC): {mcc:.4f}")

    # Top-decile lift (business metric)
    results_df = pd.DataFrame({'y_true': y_test.values, 'y_proba': y_pred_proba})
    results_df = results_df.sort_values('y_proba', ascending=False)
    top_n = int(np.ceil(0.10 * len(results_df)))
    top_group = results_df.head(top_n)
    top_rate = top_group['y_true'].mean()
    base_rate = results_df['y_true'].mean()
    top_decile_lift = (top_rate / base_rate) if base_rate > 0 else np.nan
    print(f"\nTop-decile lift: {top_decile_lift:.4f} (top10% rate={top_rate:.4f} vs base rate={base_rate:.4f})")

    
    # Plot results
    if hasattr(best_model, 'feature_importances_'):
        final_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        final_importance = feature_importance.loc[
            feature_importance['feature'].isin(selected_features)
        ]
    
    plot_results(y_test, y_pred, y_pred_proba, final_importance)
    
    return best_model, scaler, selected_features, label_encoders



# Load data WITH cluster_improved from hierarchical clustering
path = Path("data") / "clustered_data_improved.csv"
df = pd.read_csv(path)

assert 'cluster_improved' in df.columns, "cluster_improved column is missing!"

best_model, scaler, selected_features, label_encoders = main(df)
