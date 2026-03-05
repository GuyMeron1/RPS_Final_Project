import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os

PICKLE_DIR = "Data/pickles"
MODELS_DIR = "Models"
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")

def best_model_ever_trained_path():
    MODELS_DIR = "Models_Trained"
    BEST_MODEL = "best_model_DT_60P_3F_20Each.pkl"
    return os.path.join(MODELS_DIR, BEST_MODEL)
def best_model_path():
    MODELS_DIR = "Models"
    return os.path.join(MODELS_DIR, "best_model.pkl")

BEST_MODEL_PATH = best_model_path()

def get_best_model_path():
    if os.path.exists(BEST_MODEL_PATH):
        return BEST_MODEL_PATH
    return None
def load_data(file_prefix):
    try:
        with open(os.path.join(PICKLE_DIR, f"X_{file_prefix}.pkl"), "rb") as f:
            X = pickle.load(f)
        with open(os.path.join(PICKLE_DIR, f"y_{file_prefix}.pkl"), "rb") as f:
            y = pickle.load(f)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        return X, y
    except Exception as e:
        print(f"Error loading {file_prefix} data: {e}")
        return None, None
def plot_knn_history_from_train():
    print("Plotting KNN history from training")
    history_path = os.path.join(PICKLE_DIR, "knn_training_history.pkl")

    if not os.path.exists(history_path):
        print("No training history found.")
        return

    with open(history_path, "rb") as f:
        history = pickle.load(f)
    df = pd.DataFrame(history)

    selected_k, selected_p = None, None
    if os.path.exists(KNN_MODEL_PATH):
        with open(KNN_MODEL_PATH, "rb") as f:
            m = pickle.load(f)
            if hasattr(m, 'n_neighbors'):
                selected_k = m.n_neighbors
                selected_p = m.p

    if selected_k is None:
        best_idx = df['score'].idxmax()
        selected_k = df.loc[best_idx, 'k']
        selected_p = df.loc[best_idx, 'p']

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    p_titles = {1: "Manhattan Distance (p=1)", 2: "Euclidean Distance (p=2)"}
    p_colors = {1: 'blue', 2: 'green'}

    for i, p_val in enumerate([1, 2]):
        subset = df[df['p'] == p_val]
        ax = axes[i]
        ax.plot(subset['k'], subset['score'], marker='o', color=p_colors[p_val], linewidth=2)

        if p_val == selected_p:
            best_score = subset[subset['k'] == selected_k]['score'].values[0]
            ax.axvline(x=selected_k, color='red', linestyle='--', alpha=0.7, label=f'SELECTED: K={selected_k}')
            ax.scatter(selected_k, best_score, color='red', s=100, zorder=5)
            ax.annotate(f'Score: {best_score:.4f}',
                        xy=(selected_k, best_score),
                        xytext=(selected_k + 0.2, best_score),
                        color='black', fontweight='bold')

        ax.set_title(p_titles[p_val], fontsize=14)
        ax.set_ylabel("Validation F1-Score")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.xlabel("K (Number of Neighbors)", fontsize=12)
    plt.xticks(df['k'].unique())
    plt.suptitle(f"KNN Hyperparameter Analysis\n(Final Model Choice: K={selected_k}, p={selected_p})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
def plot_confusion_matrix_now():
    print("Generating Confusion Matrix")
    X_test, y_test = load_data("test")
    model_path = get_best_model_path()
    if not model_path: return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    if y_pred.ndim > 1: y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Rock', 'Paper', 'Scissors'],
                yticklabels=['Rock', 'Paper', 'Scissors'])
    plt.title(f"Confusion Matrix (Best Overall Model)")
    plt.ylabel('Actual Label (Ground Truth)')
    plt.xlabel('Predicted Label (Model Classification)')
    plt.show()
def plot_classification_report_heat():
    print("Generating Visual Classification Report")
    X_test, y_test = load_data("test")
    model_path = get_best_model_path()
    if not model_path: return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    if y_pred.ndim > 1: y_pred = np.argmax(y_pred, axis=1)

    report = classification_report(y_test, y_pred,
        target_names=['Rock', 'Paper', 'Scissors'],
        output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :3].T
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_report, annot=True, cmap="YlGnBu", cbar=False)
    plt.title("Classification Metrics per Class")
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix_now()
    plot_knn_history_from_train()
    plot_classification_report_heat()