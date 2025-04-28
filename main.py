from src.data_preprocessing import load_and_prepare
from src.models import get_models, train_all
from src.gui import LungCancerGUI
from sklearn.metrics import accuracy_score

def main():
    """Main function to run the lung cancer prediction application."""
    # Load and prepare data
    df, Xtr, Xte, ytr, yte, features, scaler = load_and_prepare()
    
    # Get and train models
    models = train_all(get_models(ytr), Xtr, ytr)
    
    # Launch GUI
    app = LungCancerGUI(models, df, Xtr, Xte, ytr, yte, features, scaler)
    app.mainloop()

if __name__ == "__main__":
    main()