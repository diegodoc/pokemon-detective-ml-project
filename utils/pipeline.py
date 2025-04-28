import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
import seaborn as sns
import shap


RANDOM_STATE = 42
TARGET_COL = "Team Rocket"

def prepare_dataset(raw_data_path: str, processed_path: str, input_path: str, test_size: float = 0.2) -> tuple:

    df = pd.read_csv(raw_data_path)
    df = df.copy()
    
    df[TARGET_COL] = df[TARGET_COL].str.strip().str.title()
    
    bool_cols = ["Is Pokemon Champion", "Rare Item Holder", "Charity Participation"]
    for col in bool_cols:
        df[col] = df[col].map({False: 0, True: 1})
    
    status_encoder = OrdinalEncoder(categories=[["Low", "Middle", "High"]])
    df["Economic Status"] = status_encoder.fit_transform(df[["Economic Status"]])
    
    os.makedirs(processed_path, exist_ok=True)
    df.to_csv(f"{processed_path}/processed_dataset.csv", index=False)
    
    df_labeled = df[df[TARGET_COL].notna()].copy()
    df_unlabeled = df[df[TARGET_COL].isna()].copy()
    
    df_labeled[TARGET_COL] = df_labeled[TARGET_COL].map({"No": 0, "Yes": 1})
    
    train_df, test_df = train_test_split(
        df_labeled, 
        test_size=test_size, 
        stratify=df_labeled[TARGET_COL],
        random_state=RANDOM_STATE
    )
    
    os.makedirs(input_path, exist_ok=True)
    train_df.to_csv(f"{input_path}/train.csv", index=False)
    test_df.to_csv(f"{input_path}/test.csv", index=False)
    df_unlabeled.to_csv(f"{input_path}/unlabeled.csv", index=False)
    
    return train_df, test_df, df_unlabeled

def prepare_features(df: pd.DataFrame, is_train: bool = True, fitted_columns: pd.Index = None) -> tuple:
    df = df.copy()
    
    cat_cols = ["Profession", "Most Used Pokemon Type", "PokéBall Usage", "Battle Strategy", "City"]
    df = pd.get_dummies(df, columns=cat_cols)
    
    if is_train:
        y = df[TARGET_COL]
        X = df.drop(TARGET_COL, axis=1)
        return X, y, X.columns
    else:
        df = df.reindex(columns=fitted_columns, fill_value=0)
        return df

def train_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                y_train: pd.Series, y_test: pd.Series, 
                use_smote: bool = False,
                use_class_weight: bool = False, n_splits: int = 5) -> tuple:
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Apply SMOTE if requested
        if use_smote:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_fold_train, y_fold_train = smote.fit_resample(X_fold_train, y_fold_train)
        
        # Apply class weights if requested
        if use_class_weight:
            class_counts = y_fold_train.value_counts()
            total_samples = len(y_fold_train)
            weights = {
                0: total_samples / (2 * class_counts[0]),
                1: total_samples / (2 * class_counts[1])
            }
            model.set_params(class_weight=weights)
        
        # Train model on fold
        model.fit(X_fold_train, y_fold_train)
        
        # Evaluate on validation fold
        y_fold_prob = model.predict_proba(X_fold_val)[:, 1]
        fold_score = roc_auc_score(y_fold_val, y_fold_prob)
        cv_scores.append(fold_score)
        
        print(f"Fold {fold} AUC-ROC: {fold_score:.3f}")
    
    print(f"\nMean CV AUC-ROC: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    
    # Final training on full training set
    if use_smote:
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n📋 Final Test Set Results:")
    print(classification_report(y_test, y_pred))
    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n🎯 Test Set AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
    
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.show()
    
    return model, np.mean(cv_scores)

def predict_unlabeled(model: object, unlabeled_df: pd.DataFrame, 
                     fitted_columns: pd.Index, output_path: str) -> None:

    X_unlabeled = prepare_features(
        unlabeled_df,
        is_train=False,
        fitted_columns=fitted_columns
    )
    
    probabilities = model.predict_proba(X_unlabeled)[:, 1]
    predictions = model.predict(X_unlabeled)
    
    unlabeled_df["Team_Rocket_Probability"] = probabilities
    unlabeled_df["Team_Rocket_Prediction"] = predictions
    
    os.makedirs(output_path, exist_ok=True)
    unlabeled_df.to_csv(f"{output_path}/predictions.csv", index=False)
    print("✅ Predictions saved to 'predictions.csv'")

def plot_metric_by_strategy(df):

    def plot_metric(metric_prefix, metric_name, palette="tab10"):
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(
            x="Strategy", 
            y=f"{metric_prefix}_{metric_name}", 
            hue="Strategy",
            data=df,
            palette=palette
        )
        plt.title(f"{metric_prefix} - {metric_name.capitalize()} by Strategy", fontsize=14)
        plt.xlabel("Balancing Strategy")
        plt.ylabel(metric_name.capitalize())
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend().remove()

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=10)

        plt.tight_layout()
        plt.show()

    # Plot metrics for class 0 and 1
    for metric in ['precision', 'recall', 'f1']:
        plot_metric('0', metric)
        plot_metric('1', metric)

    # Plot mean AUC-ROC
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x="Strategy", 
        y="mean_AUC_ROC", 
        hue="Strategy",
        data=df,
        palette="Set2"
    )
    plt.title("Mean AUC-ROC by Strategy", fontsize=14)
    plt.xlabel("Balancing Strategy")
    plt.ylabel("AUC-ROC")
    plt.ylim(0.99, 1.001)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend().remove()

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3, fontsize=7)

    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X_train: pd.DataFrame) -> None:
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    print("\n🎯 Top 10 Most Important Features:")
    print(importances.head(10))