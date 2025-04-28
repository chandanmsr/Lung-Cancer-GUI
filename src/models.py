from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

def get_models(y_train):
    """Initialize machine learning models with appropriate parameters."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=5,
            class_weight='balanced', random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        "SVM": SVC(
            probability=True, kernel='rbf',
            class_weight='balanced', random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3,
            scale_pos_weight=sum(y_train == 0)/sum(y_train == 1),
            random_state=42, eval_metric='logloss'
        ),
        "Naive Bayes": GaussianNB()
    }

def train_all(models, X_train, y_train):
    """Train all models on the given dataset."""
    for m in models.values():
        m.fit(X_train, y_train)
    return models