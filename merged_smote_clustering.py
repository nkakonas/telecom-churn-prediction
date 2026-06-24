from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    balanced_accuracy_score, average_precision_score, brier_score_loss,
    log_loss, matthews_corrcoef
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df, use_cluster=True):
    """
    Clean and preprocess the Telco churn dataset for modeling.
    """
    data = df.copy()
    
    # Remove metadata and unnecessary columns
    cols_to_drop = [
        'customer_id', 'count', 'lat_long', 'churn_label', 
        'churn_score', 'churn_reason', 'id'
    ]
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    
    # Prevent Target Leakage
    leak_cols = [
        c for c in data.columns
        if c.startswith("churn_") and c != "churn_value"
    ]
    
    if leak_cols:
        print(f"[Leakage protection] Dropping: {leak_cols}")
        data = data.drop(columns=leak_cols, errors="ignore")
    
    # Remove cluster feature if requested
    if not use_cluster:
        cluster_cols = [c for c in data.columns if 'cluster' in c.lower()]
        if cluster_cols:
            print(f"[No cluster version] Dropping: {cluster_cols}")
            data = data.drop(columns=cluster_cols)
    
    # Clean numeric columns
    if 'total_charges' in data.columns:
        data['total_charges'] = pd.to_numeric(data['total_charges'], errors='coerce')
        data['total_charges'].fillna(data['total_charges'].median(), inplace=True)
    
    # Feature engineering
    if 'monthly_charges' in data.columns and 'tenure_months' in data.columns:
        data['avg_monthly_charges'] = data['total_charges'] / (data['tenure_months'] + 1)
        data['charges_per_tenure'] = data['monthly_charges'] / (data['tenure_months'] + 1)
    
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'churn_value']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    return data, label_encoders


def get_all_models():
    """
    Return dictionary of all models to test
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', use_label_encoder=False),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced'),
        'SVM': SVC(random_state=42, probability=True, class_weight='balanced')
    }
    return models


def compare_all_configurations(df_with_cluster, df_without_cluster):
    """
    Compare all 28 model configurations:
    - 7 models × 2 feature versions × 2 sampling strategies
    """
    results = []
    
    configurations = [
        ('Baseline', df_without_cluster, False),
        ('Baseline', df_without_cluster, True),
        ('Enhanced', df_with_cluster, False),
        ('Enhanced', df_with_cluster, True),
    ]
    
    config_names = {
        (False, False): 'Baseline_NoSMOTE',
        (False, True): 'Baseline_SMOTE',
        (True, False): 'Enhanced_NoSMOTE',
        (True, True): 'Enhanced_SMOTE'
    }
    
    for use_cluster in [False, True]:
        for use_smote in [False, True]:
            config_name = config_names[(use_cluster, use_smote)]
            df = df_with_cluster if use_cluster else df_without_cluster
            
            print(f"\n{'='*60}")
            print(f"Configuration: {config_name}")
            print(f"{'='*60}")
            
            # Preprocess
            data, _ = preprocess_data(df, use_cluster=use_cluster)
            X = data.drop('churn_value', axis=1)
            y = data['churn_value']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=min(15, X_train.shape[1]))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            selected_features = X_train.columns[selector.get_support()].tolist()
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Apply SMOTE if requested
            if use_smote:
                smote = SMOTE(random_state=42)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                print(f"After SMOTE - Train size: {len(y_train)}, Churn rate: {y_train.mean():.2%}")
            
            # Test all models
            models = get_all_models()
            
            for model_name, model in models.items():
                print(f"\nTesting {model_name}...")
                
                try:
                    # Cross-validation
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                               cv=cv, scoring='roc_auc', n_jobs=-1)
                    
                    # Train on full training set
                    model.fit(X_train_scaled, y_train)
                    
                    # Test set evaluation
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    # Calculate metrics
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    pr_auc = average_precision_score(y_test, y_pred_proba)
                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    mcc = matthews_corrcoef(y_test, y_pred)
                    
                    results.append({
                        'Configuration': config_name,
                        'Model': model_name,
                        'Use_Cluster': use_cluster,
                        'Use_SMOTE': use_smote,
                        'CV_ROC_AUC_mean': cv_scores.mean(),
                        'CV_ROC_AUC_std': cv_scores.std(),
                        'Test_ROC_AUC': roc_auc,
                        'Test_PR_AUC': pr_auc,
                        'Test_Balanced_Acc': bal_acc,
                        'Test_MCC': mcc
                    })
                    
                    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                    print(f"  Test ROC-AUC: {roc_auc:.4f}")
                    
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue
    
    return pd.DataFrame(results)


def plot_comparison_results(results_df, save_path='model_comparison_plot.png'):
    """
    Visualize comparison results
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Heatmap: Model × Configuration
    pivot_cv = results_df.pivot_table(
        values='CV_ROC_AUC_mean', 
        index='Model', 
        columns='Configuration'
    )
    sns.heatmap(pivot_cv, annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[0, 0], 
                vmin=0.7, vmax=0.9, cbar_kws={'label': 'CV ROC-AUC'})
    axes[0, 0].set_title('Cross-Validation ROC-AUC Heatmap', fontsize=14, fontweight='bold')
    
    # 2. Bar plot: Top 10 configurations
    top_results = results_df.nlargest(10, 'CV_ROC_AUC_mean').copy()
    top_results['Label'] = top_results['Model'] + '\n' + top_results['Configuration']
    
    bars = axes[0, 1].barh(range(len(top_results)), top_results['CV_ROC_AUC_mean'])
    axes[0, 1].set_yticks(range(len(top_results)))
    axes[0, 1].set_yticklabels(top_results['Label'], fontsize=9)
    axes[0, 1].set_xlabel('CV ROC-AUC', fontsize=12)
    axes[0, 1].set_title('Top 10 Model Configurations', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    
    # Color bars
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 3. Effect of SMOTE
    smote_comparison = results_df.groupby(['Model', 'Use_SMOTE'])['CV_ROC_AUC_mean'].mean().reset_index()
    smote_pivot = smote_comparison.pivot(index='Model', columns='Use_SMOTE', values='CV_ROC_AUC_mean')
    smote_pivot.plot(kind='bar', ax=axes[1, 0], color=['#ff7f0e', '#2ca02c'])
    axes[1, 0].set_title('Effect of SMOTE on Model Performance', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Model', fontsize=12)
    axes[1, 0].set_ylabel('Average CV ROC-AUC', fontsize=12)
    axes[1, 0].legend(['No SMOTE', 'With SMOTE'], loc='lower right')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Effect of Cluster Feature
    cluster_comparison = results_df.groupby(['Model', 'Use_Cluster'])['CV_ROC_AUC_mean'].mean().reset_index()
    cluster_pivot = cluster_comparison.pivot(index='Model', columns='Use_Cluster', values='CV_ROC_AUC_mean')
    cluster_pivot.plot(kind='bar', ax=axes[1, 1], color=['#1f77b4', '#d62728'])
    axes[1, 1].set_title('Effect of Cluster Feature on Model Performance', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Model', fontsize=12)
    axes[1, 1].set_ylabel('Average CV ROC-AUC', fontsize=12)
    axes[1, 1].legend(['Baseline', 'With Cluster'], loc='lower right')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {save_path}")
    plt.close()


def tune_best_model(X_train, y_train, model_name):
    """
    Hyperparameter tuning for the best model
    """
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning: {model_name}")
    print(f"{'='*60}")
    
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 0.9, 1.0]
        },
        'CatBoost': {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'min_child_samples': [20, 30, 50]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf']
        }
    }
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', use_label_encoder=False),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced'),
        'SVM': SVC(random_state=42, probability=True, class_weight='balanced')
    }
    
    model = models[model_name]
    param_grid = param_grids[model_name]
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='roc_auc', 
        n_jobs=-1, verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def plot_final_evaluation(y_test, y_pred, y_pred_proba, feature_importance=None, 
                         save_path='best_model_evaluation.png'):
    """
    Visualize final model evaluation
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
                cbar_kws={'label': 'Count'})
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j+0.5, i+0.7, f'({cm_norm[i, j]:.1%})', 
                          ha='center', va='center', fontsize=10, color='gray')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance (if available)
    if feature_importance is not None and len(feature_importance) > 0:
        top_n = min(15, len(feature_importance))
        top_features = feature_importance.head(top_n)
        
        bars = axes[1, 0].barh(range(len(top_features)), top_features['importance'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'], fontsize=10)
        axes[1, 0].set_xlabel('Importance', fontsize=12)
        axes[1, 0].set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    else:
        axes[1, 0].text(0.5, 0.5, 'Feature importance not available\nfor this model', 
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
    
    # 4. Prediction Distribution
    axes[1, 1].hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
                    bins=30, label=['No Churn', 'Churn'], alpha=0.7, color=['#1f77b4', '#ff7f0e'])
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[1, 1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Final evaluation plot saved to: {save_path}")
    plt.close()


def main():
    """
    Main pipeline
    """
    print("="*80)
    print("COMPREHENSIVE CHURN PREDICTION MODEL COMPARISON")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading data...")
    path_with_cluster = Path("data") / "clustered_data_improved.csv"
    path_without_cluster = Path("data") / "merged_telco_preprocessed.csv"
    
    df_with_cluster = pd.read_csv(path_with_cluster)
    df_without_cluster = pd.read_csv(path_without_cluster)
    
    print(f"✓ Data with cluster: {df_with_cluster.shape}")
    print(f"✓ Data without cluster: {df_without_cluster.shape}")
    
    # Compare all configurations
    print("\n[2/5] Comparing 28 model configurations...")
    print("This will take some time... ☕")
    results_df = compare_all_configurations(df_with_cluster, df_without_cluster)
    
    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print(f"\n✓ Results saved to: model_comparison_results.csv")
    
    # Plot comparison
    print("\n[3/5] Creating comparison visualization...")
    plot_comparison_results(results_df)
    
    # Select best model
    best_config = results_df.loc[results_df['CV_ROC_AUC_mean'].idxmax()]
    print(f"\n{'='*80}")
    print("BEST MODEL CONFIGURATION:")
    print(f"{'='*80}")
    print(f"Model: {best_config['Model']}")
    print(f"Configuration: {best_config['Configuration']}")
    print(f"Use Cluster: {best_config['Use_Cluster']}")
    print(f"Use SMOTE: {best_config['Use_SMOTE']}")
    print(f"CV ROC-AUC: {best_config['CV_ROC_AUC_mean']:.4f} (+/- {best_config['CV_ROC_AUC_std']:.4f})")
    print(f"Test ROC-AUC: {best_config['Test_ROC_AUC']:.4f}")
    
    # Retrain best configuration with tuning
    print(f"\n[4/5] Retraining best model with hyperparameter tuning...")
    
    df = df_with_cluster if best_config['Use_Cluster'] else df_without_cluster
    data, _ = preprocess_data(df, use_cluster=best_config['Use_Cluster'])
    
    X = data.drop('churn_value', axis=1)
    y = data['churn_value']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(15, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # SMOTE
    if best_config['Use_SMOTE']:
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    
    # Tune
    best_model = tune_best_model(X_train_scaled, y_train, best_config['Model'])
    
    # Final evaluation
    print(f"\n[5/5] Final model evaluation...")
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE")
    print("="*80)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred_proba)
    ll = log_loss(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\nKey Metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  Matthews Corr Coef: {mcc:.4f}")
    
    # Top-decile lift
    results_test = pd.DataFrame({'y_true': y_test.values, 'y_proba': y_pred_proba})
    results_test = results_test.sort_values('y_proba', ascending=False)
    top_n = int(np.ceil(0.10 * len(results_test)))
    top_rate = results_test.head(top_n)['y_true'].mean()
    base_rate = results_test['y_true'].mean()
    lift = (top_rate / base_rate) if base_rate > 0 else np.nan
    print(f"\nBusiness Metric:")
    print(f"  Top-decile lift: {lift:.4f} (top 10% rate={top_rate:.4f} vs base={base_rate:.4f})")
    
    # Feature importance
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    # Plot final evaluation
    plot_final_evaluation(y_test, y_pred, y_pred_proba, feature_importance)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED! 🎉")
    print("="*80)
    print("\nGenerated files:")
    print("  1. model_comparison_results.csv")
    print("  2. model_comparison_plot.png")
    print("  3. best_model_evaluation.png")
    
    return best_model, scaler, selected_features, results_df


if __name__ == "__main__":
    best_model, scaler, selected_features, results_df = main()