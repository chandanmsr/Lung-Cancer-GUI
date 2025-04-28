import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA

def plot_class_distribution(y):
    """Plot the distribution of cancer vs non-cancer cases."""
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(['No Cancer', 'Cancer'], counts, color=['dodgerblue', 'crimson'])
    plt.title('Class Distribution')
    plt.ylabel('Count')
    for i, c in enumerate(counts):
        plt.text(i, c+2, str(c), ha='center')
    plt.show()

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical features."""
    num = df.select_dtypes(include=['int64', 'float64'])
    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap='coolwarm', square=True, cbar_kws={'shrink': .8})
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_pca(X, y, scaler):
    """Plot PCA projection of the data."""
    Xf = scaler.transform(X)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(Xf)
    hue = y.map({0: 'No Cancer', 1: 'Cancer'})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=comps[:, 0], y=comps[:, 1],
                    hue=hue, palette=['dodgerblue', 'crimson'],
                    alpha=0.7, s=80)
    plt.title("PCA Projection (2 Components)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Status")
    plt.show()

def plot_model_comparison(perf_df):
    """Plot model performance comparison."""
    # Bar chart
    melt = perf_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melt, x='Model', y='Score', hue='Metric')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'PR AUC']
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], metrics)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ["0.2", "0.4", "0.6", "0.8", "1.0"],
               color="grey", size=7)
    plt.ylim(0, 1)

    for _, row in perf_df.iterrows():
        vals = row[metrics].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, marker='o', label=row['Model'])

    plt.title('Performance Radar Chart', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

def plot_feature_importance(models, features):
    """Plot feature importance for tree-based models."""
    tree_models = {
        name: m for name, m in models.items()
        if hasattr(m, 'feature_importances_')
    }
    if not tree_models:
        raise ValueError("No tree models available for feature importance")
    
    n = len(tree_models)
    rows = int(np.ceil(n/2))
    cols = 2 if n > 1 else 1
    plt.figure(figsize=(12, 5*rows))
    for i, (name, m) in enumerate(tree_models.items(), start=1):
        imp = m.feature_importances_
        idx = np.argsort(imp)[-10:]
        labs = [features[j] for j in idx]
        ax = plt.subplot(rows, cols, i)
        ax.barh(range(len(idx)), imp[idx], color='dodgerblue')
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(labs)
        ax.set_title(f'{name} Top 10 Features')
    plt.tight_layout()
    plt.show()

def plot_calibration_curves(models, X_test, y_test):
    """Plot calibration curves for probabilistic models."""
    plt.figure(figsize=(8, 8))
    for name, m in models.items():
        if hasattr(m, 'predict_proba'):
            prob = m.predict_proba(X_test)[:, 1]
            frac, mean = calibration_curve(y_test, prob, n_bins=10)
            plt.plot(mean, frac, 's-', label=name)
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()