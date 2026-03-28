import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os

# System Configuration
# Define directory structure for data retrieval and report generation
PICKLE_DIR = "Data/pickles"
MODELS_DIR = "Models"
REPORTS_DIR = "Reports"

# Central reference to the main trained model file
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")

# Ensure output directory exists to avoid Input/Output errors during saving
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)


# Data Utility Functions
def get_best_model_path():
    if os.path.exists(BEST_MODEL_PATH):
        return BEST_MODEL_PATH
    return None

def load_data(file_prefix):
    """
    Loads and prepares the data for the models.
    - Changes the input (X) from 3D to 2D so the models can read it easily.
    - Simplifies the labels (y) into a basic list of numbers (0, 1, or 2).
    """
    try:
        with open(os.path.join(PICKLE_DIR, f"X_{file_prefix}.pkl"), "rb") as f:
            X = pickle.load(f)
        with open(os.path.join(PICKLE_DIR, f"y_{file_prefix}.pkl"), "rb") as f:
            y = pickle.load(f)

        # Reshape: (samples, frames, keypoints) -> (samples, frames * keypoints)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        return X, y

    except Exception as e:
        print(f"Error loading {file_prefix} data: {e}")
        return None, None


# Statistical Visualization Functions
def plot_knn_history_from_train():
    """
    Creates a graph to show how the best settings for the KNN model founded.
    - Compares accuracy for different numbers of neighbors (K).
    - Compares two different ways to measure distance (p=1 vs p=2).
    """
    print("Plotting KNN history from training...")
    history_path = os.path.join(PICKLE_DIR, "knn_training_history.pkl")

    if not os.path.exists(history_path):
        print("No KNN training history found.")
        return

    with open(history_path, "rb") as f:
        history = pickle.load(f)
    df = pd.DataFrame(history)

    # Extract chosen hyperparameters from the saved model for visual highlighting
    selected_k, selected_p = None, None
    if os.path.exists(KNN_MODEL_PATH):
        with open(KNN_MODEL_PATH, "rb") as f:
            m = pickle.load(f)
            if hasattr(m, 'n_neighbors'):
                selected_k = m.n_neighbors
                selected_p = m.p

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    p_titles = {1: "Manhattan Distance (p=1)", 2: "Euclidean Distance (p=2)"}

    for i, p_val in enumerate([1, 2]):
        subset = df[df['p'] == p_val]
        ax = axes[i]
        ax.plot(subset['k'], subset['score'], marker='o', linewidth=2, color='tab:blue')

        # Visual indicator for the optimal parameter set
        if p_val == selected_p and selected_k is not None:
            ax.axvline(x=selected_k, color='red', linestyle='--', label=f'Chosen: K={selected_k}')

        ax.set_title(p_titles[p_val])
        ax.set_ylabel("Validation F1-Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("KNN Hyperparameter Optimization Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORTS_DIR, "knn_optimization_plot.png"))
    plt.show()

def plot_confusion_matrix():
    """
    Creates a heatmap to show how well the model predicted each gesture.
    - Shows how many times the model was correct.
    - Shows exactly which gestures the model got mixed up.
    - Helps to see the accuracy for each specific action.
    """
    print("Generating Confusion Matrix...")
    X_test, y_test = load_data("test")
    model_path = get_best_model_path()

    if not model_path or X_test is None:
        print("Model or test data missing.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Rock', 'Paper', 'Scissors'],
                yticklabels=['Rock', 'Paper', 'Scissors'])
    plt.title("Confusion Matrix: Model Performance on Test Set")
    plt.ylabel('True Label (Actual)')
    plt.xlabel('Predicted Label (Model)')
    plt.savefig(os.path.join(REPORTS_DIR, "best_model_confusion_matrix.png"))
    plt.show()

def plot_nn_training_history():
    """
    Plots the progress of the Neural Network during training.
    - Loss Curve: Shows how the model's mistakes decreased over time.
    - Accuracy Curve: Shows how the model's success rate improved.
    - Helps us see if the model actually learned from the data.
    """
    print("Plotting Neural Network Training History...")
    history_path = os.path.join(PICKLE_DIR, "nn_training_history.pkl")

    if not os.path.exists(history_path):
        print("No NN history found.")
        return

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss Curve Analysis
    ax1.plot(history['loss'], color='tab:red', linewidth=2)
    ax1.set_title('Training Loss (Error Reduction)')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, alpha=0.3)

    # Validation Performance Tracking
    if history.get('validation_scores'):
        ax2.plot(history['validation_scores'], color='tab:green', linewidth=2)
        ax2.set_title('Validation Accuracy Progress')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Accuracy Score')
        ax2.grid(True, alpha=0.3)

    plt.savefig(os.path.join(REPORTS_DIR, "nn_learning_curves.png"))
    plt.show()

def save_hyperparameters_table_image():
    """
    Finds the specific settings used for each trained model.
    - Collects details like hidden layers, tree depth, or neighbors.
    - Saves a clean table image to compare all models in the report.
    """
    print("Generating Models Hyperparameters Table...")
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]

    data = []
    # Dictionary of which settings are important for each model type
    important_params = {
        'KNeighborsClassifier': ['n_neighbors', 'p', 'weights'],  # KNN
        'GaussianNB': ['var_smoothing'],  # NB
        'DecisionTreeClassifier': ['criterion', 'max_depth'], # DT
        'SVC': ['C', 'kernel', 'gamma'],  # SVM
        'MLPClassifier': ['hidden_layer_sizes', 'alpha', 'learning_rate']  # NN
    }

    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            model_type = type(model).__name__
            params = model.get_params()
            # Keep only the important settings for the table
            relevant = {k: params[k] for k in important_params.get(model_type, []) if k in params}
            data.append([model_file.replace(".pkl", ""), model_type.replace("Classifier", ""),
                         str(relevant).replace("{", "").replace("}", "").replace("'", "")])
        except:
            continue

    if not data: return

    # Creating the visual table using Matplotlib
    fig, ax = plt.subplots(figsize=(14, len(data) * 0.7 + 2))
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=["Model Name", "Type", "Key Settings"], cellLoc='left', loc='center', colWidths=[0.15, 0.15, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    plt.title("Model Settings Summary", fontsize=18, pad=30, weight='bold')
    plt.savefig(os.path.join(REPORTS_DIR, "models_hyperparameters_table.png"), bbox_inches='tight', dpi=300)
    plt.show()

def save_all_classification_reports():
    """
    Saves a detailed performance report for every model in the folder.
    - Shows how accurate the model is for Rock, Paper, and Scissors.
    - Saves the results as a heatmap image.
    """
    print("Generating classification reports for all models...")
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

            # Generate the text report and convert it to a visual heatmap
            report = classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors'], output_dict=True)
            df_report = pd.DataFrame(report).iloc[:-1, :3].T

            plt.figure(figsize=(10, 5))
            sns.heatmap(df_report, annot=True, cmap="YlGnBu", cbar=False, fmt=".2f")
            plt.title(f"Performance Report: {model_name}")

            save_path = os.path.join(REPORTS_DIR, f"report_{model_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved report for: {model_name}")

        except Exception as e:
            print(f"Could not process {model_file}: {e}")


# Main Execution Engine
if __name__ == "__main__":
    # Execute all visualization routines to populate the Reports directory
    plot_confusion_matrix()
    plot_knn_history_from_train()
    plot_nn_training_history()
    save_hyperparameters_table_image()
    save_all_classification_reports()