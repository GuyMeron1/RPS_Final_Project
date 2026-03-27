"""import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os

PICKLE_DIR = "Data/pickles"
MODELS_DIR = "Models"
REPORTS_DIR = "Reports"
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

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
            ax.annotate(f'Score: {best_score:.4f}', xy=(selected_k, best_score), xytext=(selected_k + 0.2, best_score), color='black', fontweight='bold')

        ax.set_title(p_titles[p_val], fontsize=14)
        ax.set_ylabel("Validation F1-Score")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.xlabel("K (Number of Neighbors)", fontsize=12)
    plt.xticks(df['k'].unique())
    plt.suptitle(f"KNN Hyperparameter Analysis\n(Final Model Choice: K={selected_k}, p={selected_p})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORTS_DIR, "knn_optimization_plot.png"))
    plt.show()
def plot_confusion_matrix():
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

    # שמירה
    plt.savefig(os.path.join(REPORTS_DIR, "best_model_confusion_matrix.png"))
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

    # שמירה
    plt.savefig(os.path.join(REPORTS_DIR, "best_model_classification_report.png"))
    plt.show()


def plot_nn_training_history():
    print("Plotting Neural Network Training History (MLPClassifier)")
    history_path = os.path.join(PICKLE_DIR, "nn_training_history.pkl")

    if not os.path.exists(history_path):
        print("No NN training history found. Run train_nn() first.")
        return

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    # יצירת גרף עם שני צירים
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. גרף ה-Loss (Loss Curve)
    ax1.plot(history['loss'], color='blue', linewidth=2)
    ax1.set_title('Training Loss Curve', fontsize=14)
    ax1.set_ylabel('Loss (Log Loss)')
    ax1.set_xlabel('Iterations (Epochs)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. גרף ה-Validation Accuracy (Validation Scores)
    if history['validation_scores']:
        ax2.plot(history['validation_scores'], color='green', marker='o', markersize=4, linewidth=2)
        ax2.set_title('Validation Accuracy during Training', fontsize=14)
        ax2.set_ylabel('Accuracy Score')
        ax2.set_xlabel('Iterations (Epochs)')
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, 'No Validation Scores\n(early_stopping was False)',
                 ha='center', va='center', fontsize=12)

    plt.suptitle("Neural Network Learning Process", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # שמירה לתיקיית הדוחות
    save_path = os.path.join(REPORTS_DIR, "nn_learning_curves.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.show()


def save_hyperparameters_table_image():
    print("Generating Hyperparameters Table Image with adjusted widths...")
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]

    data = []
    important_params = {
        'MLPClassifier': ['hidden_layer_sizes', 'alpha', 'learning_rate'],
        'KNeighborsClassifier': ['n_neighbors', 'p', 'weights'],
        'SVC': ['C', 'kernel', 'gamma'],
        'DecisionTreeClassifier': ['criterion', 'max_depth'],
        'RandomForestClassifier': ['n_estimators', 'max_depth']
    }

    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            model_type = type(model).__name__
            params = model.get_params()
            relevant = {k: params[k] for k in important_params.get(model_type, []) if k in params}

            data.append([
                model_file.replace(".pkl", ""),
                model_type.replace("Classifier", ""),
                str(relevant).replace("{", "").replace("}", "").replace("'", "")
            ])
        except:
            continue

    if not data: return

    # יצירת הטבלה
    fig, ax = plt.subplots(figsize=(14, len(data) * 0.7 + 2))
    ax.axis('off')

    # הגדרת רוחב עמודות: עמודה 1 ו-2 קצרות (15% כל אחת), עמודה 3 רחבה (70%)
    col_widths = [0.15, 0.15, 0.7]

    table = ax.table(
        cellText=data,
        colLabels=["Model Name", "Type", "Key Hyperparameters"],
        cellLoc='left',
        loc='center',
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)  # הגדלת גובה התאים כדי שהטקסט הארוך לא ייחתך

    # עיצוב כותרות
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white', ha='center')
            cell.set_facecolor('#2c3e50')
        else:
            cell.set_edgecolor('#bdc3c7')

    plt.title("Model Hyperparameters Summary", fontsize=18, pad=30, weight='bold')

    save_path = os.path.join(REPORTS_DIR, "models_hyperparameters_table.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Table saved with optimized widths to: {save_path}")



if __name__ == "__main__":
    #plot_confusion_matrix()
    #plot_knn_history_from_train()
    #plot_classification_report_heat()
    #plot_nn_training_history()
    save_hyperparameters_table_image()"""
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os

# --- Configuration ---
PICKLE_DIR = "Data/pickles"
MODELS_DIR = "Models"
REPORTS_DIR = "Reports"
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")

# Ensure reports directory exists
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)


# --- Path Helpers ---
def best_model_ever_trained_path():
    MODELS_DIR_TRAINED = "Models_Trained"
    BEST_MODEL = "best_model_DT_60P_3F_20Each.pkl"
    return os.path.join(MODELS_DIR_TRAINED, BEST_MODEL)
def best_model_path():
    return os.path.join(MODELS_DIR, "best_model.pkl")

BEST_MODEL_PATH = best_model_path()

def get_best_model_path():
    if os.path.exists(BEST_MODEL_PATH):
        return BEST_MODEL_PATH
    return None
# --- Core Data Loading ---
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

# --- Visualization Functions ---
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
            ax.annotate(f'Score: {best_score:.4f}', xy=(selected_k, best_score), xytext=(selected_k + 0.2, best_score),
                        color='black', fontweight='bold')

        ax.set_title(p_titles[p_val], fontsize=14)
        ax.set_ylabel("Validation F1-Score")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.xlabel("K (Number of Neighbors)", fontsize=12)
    plt.xticks(df['k'].unique())
    plt.suptitle(f"KNN Hyperparameter Analysis\n(Final Model Choice: K={selected_k}, p={selected_p})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORTS_DIR, "knn_optimization_plot.png"))
    plt.show()
def plot_confusion_matrix():
    print("Generating Confusion Matrix for Best Model")
    X_test, y_test = load_data("test")
    model_path = get_best_model_path()
    if not model_path:
        print("Best model not found at path.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Rock', 'Paper', 'Scissors'],
                yticklabels=['Rock', 'Paper', 'Scissors'])
    plt.title(f"Confusion Matrix (Best Overall Model)")
    plt.ylabel('Actual Label (Ground Truth)')
    plt.xlabel('Predicted Label (Model Classification)')
    plt.savefig(os.path.join(REPORTS_DIR, "best_model_confusion_matrix.png"))
    plt.show()
def plot_nn_training_history():
    print("Plotting Neural Network Training History")
    history_path = os.path.join(PICKLE_DIR, "nn_training_history.pkl")

    if not os.path.exists(history_path):
        print("No NN training history found.")
        return

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(history['loss'], color='blue', linewidth=2)
    ax1.set_title('Training Loss Curve', fontsize=14)
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Iterations')
    ax1.grid(True, linestyle='--', alpha=0.6)

    if history.get('validation_scores'):
        ax2.plot(history['validation_scores'], color='green', marker='o', markersize=4, linewidth=2)
        ax2.set_title('Validation Accuracy', fontsize=14)
        ax2.set_ylabel('Accuracy Score')
        ax2.set_xlabel('Iterations')
        ax2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Neural Network Learning Process", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORTS_DIR, "nn_learning_curves.png"))
    plt.show()
def save_hyperparameters_table_image():
    print("Generating Hyperparameters Summary Table...")
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]

    data = []
    important_params = {
        'MLPClassifier': ['hidden_layer_sizes', 'alpha', 'learning_rate'],
        'KNeighborsClassifier': ['n_neighbors', 'p', 'weights'],
        'SVC': ['C', 'kernel', 'gamma'],
        'DecisionTreeClassifier': ['criterion', 'max_depth'],
        'RandomForestClassifier': ['n_estimators', 'max_depth']
    }

    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            model_type = type(model).__name__
            params = model.get_params()
            relevant = {k: params[k] for k in important_params.get(model_type, []) if k in params}
            data.append([model_file.replace(".pkl", ""), model_type.replace("Classifier", ""),
                         str(relevant).replace("{", "").replace("}", "").replace("'", "")])
        except:
            continue

    if not data: return

    fig, ax = plt.subplots(figsize=(14, len(data) * 0.7 + 2))
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=["Model Name", "Type", "Key Hyperparameters"], cellLoc='left',
                     loc='center', colWidths=[0.15, 0.15, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white', ha='center')
            cell.set_facecolor('#2c3e50')

    plt.title("Model Hyperparameters Summary", fontsize=18, pad=30, weight='bold')
    plt.savefig(os.path.join(REPORTS_DIR, "models_hyperparameters_table.png"), bbox_inches='tight', dpi=300)
    plt.show()
def save_all_classification_reports():
    """Generates and saves classification reports for EVERY .pkl model in the Models directory."""
    print("Generating and saving classification reports for all models in folder...")
    X_test, y_test = load_data("test")
    if X_test is None: return

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]

    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        model_name = model_file.replace(".pkl", "")

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            y_pred = model.predict(X_test)
            if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
                y_pred = np.argmax(y_pred, axis=1)

            report = classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors'], output_dict=True)
            df_report = pd.DataFrame(report).iloc[:-1, :3].T

            plt.figure(figsize=(10, 5))
            sns.heatmap(df_report, annot=True, cmap="YlGnBu", cbar=False, fmt=".2f")
            plt.title(f"Classification Report: {model_name}")

            save_path = os.path.join(REPORTS_DIR, f"report_{model_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved report for: {model_name}")

        except Exception as e:
            print(f"Could not process {model_file}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    plot_confusion_matrix()
    plot_knn_history_from_train()
    plot_nn_training_history()
    save_hyperparameters_table_image()
    save_all_classification_reports()