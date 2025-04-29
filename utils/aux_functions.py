import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

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


def plot_feature_importance(model, X_train: pd.DataFrame) -> None:

    # Get feature importances
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importances.head(15),  # Show top 15 features
        x='importance',
        y='feature',
        palette='viridis'
    )
    
    plt.title('Top 15 Most Important Features (Random Forest)', fontsize=12)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    # Print top 10 features
    print("\n🎯 Top 10 Most Important Features:")
    print(importances.head(10))
    
    print("\n📝 Note: Higher importance scores indicate stronger influence on predictions.")

def plot_logistic_regression_importance(model: LogisticRegression, X_train: pd.DataFrame) -> None:

    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importances.head(15),
        x='importance',
        y='feature',
        palette='viridis'
    )
    
    plt.title('Top 15 Most Important Features (Logistic Regression)', fontsize=12)
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    print("\n🎯 Top 10 Most Important Features:")
    print(importances.head(10))
    
    print("\n📝 Note: Higher absolute coefficient values indicate stronger influence on predictions.")

def plot_model_metrics(df: pd.DataFrame) -> None:
    metrics_dict = {
        'AUC-ROC': df['mean_AUC_ROC'].values,
        'Precision': df['1_precision'].values,
        'Recall': df['1_recall'].values,
        'F1': df['1_f1'].values
    }
    
    plot_data = []
    for metric, values in metrics_dict.items():
        for model, value in zip(df['model'], values):
            plot_data.append({
                'Model': model,
                'Metric': metric,
                'Score': value
            })
        
    plot_df = pd.DataFrame(plot_data)
        
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=plot_df,
        x='Metric',
        y='Score',
        hue='Model',
        palette='viridis'
    )

    plt.title('Model Performance Metrics Comparison', fontsize=14)
    plt.xlabel('Metrics')
    plt.ylabel('Score')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.ylim(0, 1.1)
    plt.legend(title='')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def compare_model_predictions(lr_predictions: pd.DataFrame, rf_predictions: pd.DataFrame) -> None:

    # Create figure with a specific backend that ensures toolbar visibility
    plt.ioff()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    fig.suptitle('Logistic Regression vs Random Forest Predictions Comparison', fontsize=14)
    
    # 1. Probability Distribution with KDE and counts
    hist_lr = np.histogram(lr_predictions['Team_Rocket_Probability'], bins=50)
    hist_rf = np.histogram(rf_predictions['Team_Rocket_Probability'], bins=50)
    
    sns.histplot(
        data=lr_predictions, 
        x='Team_Rocket_Probability',
        bins=50,
        ax=axes[0,0],
        color='blue',
        alpha=0.5,
        label='Logistic Regression',
        stat='density'
    )
    sns.histplot(
        data=rf_predictions,
        x='Team_Rocket_Probability',
        bins=50,
        ax=axes[0,0],
        color='green',
        alpha=0.5,
        label='Random Forest',
        stat='density'
    )
    
    # Add max counts as annotations
    axes[0,0].text(0.02, axes[0,0].get_ylim()[1]*0.95, 
                   f'LR max count: {max(hist_lr[0])}\nRF max count: {max(hist_rf[0])}',
                   bbox=dict(facecolor='white', alpha=0.8))
    
    axes[0,0].set_title('Probability Distribution')
    axes[0,0].set_xlabel('Probability of Team Rocket Membership')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()
    
    # 2. Prediction Counts with values
    prediction_counts = pd.DataFrame({
        'LR': lr_predictions['Team_Rocket_Prediction'].value_counts(),
        'RF': rf_predictions['Team_Rocket_Prediction'].value_counts()
    })
    ax = prediction_counts.plot(kind='bar', ax=axes[0,1])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3)
    
    axes[0,1].set_title('Prediction Counts by Model')
    axes[0,1].set_xlabel('Predicted Class')
    axes[0,1].set_ylabel('Count')
    
    # 3. Probability Scatter Plot with correlation coefficient
    scatter = axes[1,0].scatter(
        lr_predictions['Team_Rocket_Probability'],
        rf_predictions['Team_Rocket_Probability'],
        alpha=0.5
    )
    axes[1,0].plot([0, 1], [0, 1], 'r--') 
    
    # Add correlation coefficient
    corr = np.corrcoef(lr_predictions['Team_Rocket_Probability'], 
                       rf_predictions['Team_Rocket_Probability'])[0,1]
    axes[1,0].text(0.02, 0.95, f'Correlation: {corr:.3f}', 
                   transform=axes[1,0].transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    axes[1,0].set_title('Probability Correlation')
    axes[1,0].set_xlabel('Logistic Regression Probability')
    axes[1,0].set_ylabel('Random Forest Probability')
    
    # 4. Disagreement Analysis with detailed stats
    disagreements = lr_predictions['Team_Rocket_Prediction'] != rf_predictions['Team_Rocket_Prediction']
    disagreement_count = disagreements.sum()
    agreement_rate = (1 - disagreements.mean()) * 100
    
    # Add more detailed statistics
    stats_text = (
        f"Model Agreement Rate: {agreement_rate:.2f}%\n"
        f"Number of Disagreements: {disagreement_count}\n"
        f"Total Predictions: {len(lr_predictions)}\n\n"
        f"Mean Probabilities:\n"
        f"LR: {lr_predictions['Team_Rocket_Probability'].mean():.3f}\n"
        f"RF: {rf_predictions['Team_Rocket_Probability'].mean():.3f}\n\n"
        f"Std Probabilities:\n"
        f"LR: {lr_predictions['Team_Rocket_Probability'].std():.3f}\n"
        f"RF: {rf_predictions['Team_Rocket_Probability'].std():.3f}"
    )
    
    axes[1,1].text(0.5, 0.5, 
                   stats_text,
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=axes[1,1].transAxes,
                   fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.8))
    axes[1,1].axis('off')
    
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return fig

def analyze_model_disagreements(lr_predictions: pd.DataFrame, rf_predictions: pd.DataFrame) -> None:
    # Find disagreements
    disagreements = lr_predictions['Team_Rocket_Prediction'] != rf_predictions['Team_Rocket_Prediction']
    disagreement_cases = lr_predictions[disagreements].copy()
    
    # Merge with RF predictions for comparison
    disagreement_cases['RF_Probability'] = rf_predictions[disagreements]['Team_Rocket_Probability']
    disagreement_cases['RF_Prediction'] = rf_predictions[disagreements]['Team_Rocket_Prediction']
    
    # Select relevant columns for analysis
    analysis_cols = [
        'ID', 'Age', 'City', 'Economic Status', 'Profession', 
        'Most Used Pokemon Type', 'Average Pokemon Level', 'Criminal Record',
        'PokéBall Usage', 'Win Ratio', 'Number of Gym Badges',
        'Battle Strategy', 'Number of Migrations', 'Rare Item Holder',
        'Debt to Kanto', 'Charity Participation',
        'Team_Rocket_Probability', 'Team_Rocket_Prediction',
        'RF_Probability', 'RF_Prediction'
    ]
    
    analysis_df = disagreement_cases[analysis_cols]

    return analysis_df