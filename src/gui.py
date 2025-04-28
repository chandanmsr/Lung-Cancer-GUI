import tkinter as tk
from tkinter import ttk, messagebox
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_curve, 
    auc, 
    average_precision_score
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from src.visualization import (
    plot_class_distribution, plot_correlation_heatmap,
    plot_pca, plot_model_comparison, plot_feature_importance,
    plot_calibration_curves
)

class LungCancerGUI(tk.Tk):
    def __init__(self, models, raw_df, X_train, X_test, y_train, y_test, features, scaler):
        super().__init__()
        self.title("Lung Cancer Prediction")
        
        # Store references to data and models
        self.models = models
        self.raw_df = raw_df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        self.scaler = scaler
        
        # Precompute test accuracies
        self.accuracies = {
            name: accuracy_score(self.y_test, m.predict(self.X_test))
            for name, m in self.models.items()
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        # Name entry
        ttk.Label(frm, text="Your Name:").grid(row=0, column=0, sticky='w')
        self.name_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.name_var).grid(
            row=0, column=1, sticky='ew', pady=2
        )

        # Model selector
        ttk.Label(frm, text="Choose model:").grid(row=1, column=0, sticky='w')
        self.model_var = tk.StringVar(value=list(self.models.keys())[0])
        ttk.OptionMenu(
            frm, self.model_var,
            self.model_var.get(), *self.models.keys()
        ).grid(row=1, column=1, sticky='ew')

        # Accuracy display
        self.acc_var = tk.StringVar()
        self.acc_var.set(
            f"Model accuracy (test): {self.accuracies[self.model_var.get()]:.1%}"
        )
        ttk.Label(
            frm, textvariable=self.acc_var,
            font=("Arial", 10, "italic")
        ).grid(row=2, column=0, columnspan=2, sticky='w', pady=2)
        self.model_var.trace('w', lambda *a: self.acc_var.set(
            f"Model accuracy (test): {self.accuracies[self.model_var.get()]:.1%}"
        ))

        # Feature inputs
        self.entry_vars, self.combo_vars, self.check_vars = {}, {}, {}
        row = 3
        for feat in self.features:
            ttk.Label(frm, text=feat.replace('_', ' ')) \
                .grid(row=row, column=0, sticky='w', pady=2)
            if feat == 'AGE':
                var = tk.StringVar()
                ttk.Entry(frm, textvariable=var).grid(
                    row=row, column=1, sticky='ew', pady=2
                )
                self.entry_vars[feat] = var
            elif feat == 'GENDER':
                var = tk.StringVar(value='M')
                ttk.OptionMenu(frm, var, 'M', 'M', 'F').grid(
                    row=row, column=1, sticky='ew', pady=2
                )
                self.combo_vars[feat] = var
            else:
                var = tk.IntVar(value=0)
                ttk.Checkbutton(frm, variable=var).grid(
                    row=row, column=1, sticky='w', pady=2
                )
                self.check_vars[feat] = var
            row += 1

        # Predict button
        ttk.Button(frm, text="Predict", command=self.predict).grid(
            row=row, column=0, columnspan=2, pady=(10, 20)
        )
        row += 1

        # Visualization buttons
        viz = [
            ("Class Distribution", self.show_class_distribution),
            ("Correlation Heatmap", self.show_corr_heatmap),
            ("PCA Scatter Plot", self.show_pca_plot),
            ("Model Comparison", self.show_model_comparison),
            ("Feature Importance", self.show_feature_importance),
            ("Calibration Curves", self.show_calibration_curves),
        ]
        for text, cmd in viz:
            ttk.Button(frm, text=text, command=cmd).grid(
                row=row, column=1, sticky='ew', pady=2
            )
            row += 1

        # View History button
        ttk.Button(frm, text="View History", command=self.show_history).grid(
            row=row, column=1, sticky='ew', pady=(10, 2)
        )

    def predict(self):
        try:
            vals = []
            for feat in self.features:
                if feat == 'AGE':
                    v = float(self.entry_vars[feat].get())
                elif feat == 'GENDER':
                    v = 1 if self.combo_vars[feat].get() == 'M' else 0
                else:
                    v = self.check_vars[feat].get()
                vals.append(v)
        except ValueError:
            return messagebox.showerror("Input error", "Please enter valid inputs.")

        # Run prediction to get conclusion
        arr = np.array(vals).reshape(1, -1)
        arr_s = self.scaler.transform(arr)
        model = self.models[self.model_var.get()]
        pred = model.predict(arr_s)[0]
        proba = model.predict_proba(arr_s)[0][1] if hasattr(model, 'predict_proba') else None

        conclusion = 'Cancer' if pred == 1 else 'No Cancer'

        # Log inputs + conclusion
        record = {'name': self.name_var.get(), 'model': self.model_var.get(), 'conclusion': conclusion}
        record.update({feat: vals[i] for i, feat in enumerate(self.features)})
        with open("data/inputs.json", "a") as f:
            json.dump(record, f)
            f.write("\n")

        # Show results
        acc = self.accuracies[self.model_var.get()]
        rec = self._get_recommendation(pred, proba)
        self.show_result_window(pred, proba, acc, rec)

    def show_history(self):
        try:
            with open("data/inputs.json") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return messagebox.showinfo("No History", "No input history found.")

        records = []
        for ln in lines:
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        if not records:
            return messagebox.showinfo("No History", "No valid records.")

        cols = ["name"] + self.features + ["model", "conclusion"]
        win = tk.Toplevel(self)
        win.title("Input History")
        win.geometry("900x400")

        tree = ttk.Treeview(win, columns=cols, show="headings")
        tree.pack(side="left", fill="both", expand=True)
        vs = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        vs.pack(side="right", fill="y")
        hs = ttk.Scrollbar(win, orient="horizontal", command=tree.xview)
        hs.pack(side="bottom", fill="x")
        tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)

        for c in cols:
            tree.heading(c, text=c.replace("_", " ").title())
            tree.column(c, width=100, anchor="center")

        for r in records:
            tree.insert("", "end", values=[r.get(c, "") for c in cols])

        win.grab_set()

    def _get_recommendation(self, pred, proba):
        if pred == 1:
            if proba and proba > 0.7:
                return "Immediate consult & CT scan recommended"
            elif proba and proba > 0.5:
                return "Consult a doctor soon; consider CT scan"
            else:
                return "Monitor symptoms; no CT scan needed"
        else:
            if proba and proba > 0.3:
                return "Consider follow-up consultation"
            return "Low risk; no CT scan needed"

    def show_result_window(self, pred, proba, acc, rec):
        win = tk.Toplevel(self)
        win.title("Prediction Result")
        win.geometry("400x350")
        tk.Label(win, text="Lung Cancer Prediction",
                 font=("Helvetica", 16, "bold")).pack(pady=(10, 5))
        color = 'crimson' if pred == 1 else 'dodgerblue'
        tk.Label(win, text=f"Result: {'Cancer' if pred else 'No Cancer'}",
                 font=("Helvetica", 14), fg=color).pack(pady=2)
        if proba is not None:
            tk.Label(win, text=f"Probability: {proba:.1%}",
                     font=("Helvetica", 12)).pack(pady=2)
        tk.Label(win, text=f"Model Accuracy: {acc:.1%}",
                 font=("Helvetica", 12, "italic")).pack(pady=2)
        tk.Label(win, text="Recommendation:",
                 font=("Helvetica", 12, "underline")).pack(pady=(10, 2))
        tk.Label(win, text=rec, font=("Helvetica", 12),
                 wraplength=350, justify="center").pack(pady=(0, 10))
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 10))
        win.grab_set()

    def show_class_distribution(self):
        plot_class_distribution(self.raw_df['LUNG_CANCER'])

    def show_corr_heatmap(self):
        plot_correlation_heatmap(self.raw_df)

    def show_pca_plot(self):
        plot_pca(self.raw_df.drop('LUNG_CANCER', axis=1),
                self.raw_df['LUNG_CANCER'],
                self.scaler)

    def show_model_comparison(self):
        data = []
        for name, m in self.models.items():
            yp = m.predict(self.X_test)
            pb = m.predict_proba(self.X_test)[:, 1] if hasattr(m, 'predict_proba') else None
            fpr, tpr, _ = (roc_curve(self.y_test, pb) if pb is not None else (None, None, None))
            data.append({
                'Model': name,
                'Accuracy': accuracy_score(self.y_test, yp),
                'Precision': precision_score(self.y_test, yp),
                'Recall': recall_score(self.y_test, yp),
                'F1': f1_score(self.y_test, yp),
                'ROC AUC': auc(fpr, tpr) if fpr is not None else np.nan,
                'PR AUC': average_precision_score(self.y_test, pb) if pb is not None else np.nan
            })
        perf_df = pd.DataFrame(data)
        plot_model_comparison(perf_df)

    def show_feature_importance(self):
        try:
            plot_feature_importance(self.models, self.features)
        except ValueError as e:
            messagebox.showinfo("Info", str(e))

    def show_calibration_curves(self):
        plot_calibration_curves(self.models, self.X_test, self.y_test)