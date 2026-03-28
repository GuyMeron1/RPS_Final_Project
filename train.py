import os
import pickle
import warnings
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning

# Ignore warnings about models not converging (happens sometimes with small data)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Paths for data and where to save the trained models
DATA_PICKLE_DIR = os.path.join("Data", "pickles")
MODEL_DIR = "Models"
os.makedirs(MODEL_DIR, exist_ok=True)

# LOAD DATA
# Loading the split datasets
with open(f"{DATA_PICKLE_DIR}/X_train.pkl", "rb") as f: X_train = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/y_train.pkl", "rb") as f: y_train = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/X_valid.pkl", "rb") as f: X_valid = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/y_valid.pkl", "rb") as f: y_valid = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/X_test.pkl", "rb") as f: X_test = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/y_test.pkl", "rb") as f: y_test = pickle.load(f)

# FLATTEN DATA
# Classic ML models expect a 2D array (Samples, Features).
# We flatten the sequences into one long row of numbers.
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# MODEL 1: K-NEAREST NEIGHBORS (KNN)
def train_knn():
    # Grid of parameters to test (Number of neighbors and Distance metric)
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21], 'p': [1, 2]}
    metrics = {}
    knn_history = []

    for k in param_grid['n_neighbors']:
        for p in param_grid['p']:
            knn = KNeighborsClassifier(n_neighbors=k, p=p)
            knn.fit(X_train, y_train)
            y_val_pred = knn.predict(X_valid)
            # F1-score measures the balance between precision and recall
            score = f1_score(y_valid, y_val_pred, average='macro')
            metrics[(k, p)] = score
            knn_history.append({'k': k, 'p': p, 'score': score})

    # Save training history for visualization later
    with open(os.path.join(DATA_PICKLE_DIR, "knn_training_history.pkl"), "wb") as f:
        pickle.dump(knn_history, f)

    # Pick the best parameters and retrain on combined Train+Valid data
    best_params = max(metrics, key=metrics.get)
    best_knn = KNeighborsClassifier(n_neighbors=best_params[0], p=best_params[1])
    best_knn.fit(np.vstack([X_train, X_valid]), np.hstack([y_train, y_valid]))

    with open(os.path.join(MODEL_DIR, "knn_model.pkl"), "wb") as f:
        pickle.dump(best_knn, f)

    print(f"KNN - Best params: {best_params}")
    return best_knn

# MODEL 2: NAIVE BAYES (NB)
def train_nb():
    # Testing different smoothing parameters to avoid zero-probability issues
    param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
    metrics = {}
    for vs in param_grid['var_smoothing']:
        nb = GaussianNB(var_smoothing=vs)
        nb.fit(X_train, y_train)
        y_val_pred = nb.predict(X_valid)
        metrics[vs] = f1_score(y_valid, y_val_pred, average='macro')

    best_vs = max(metrics, key=metrics.get)
    best_nb = GaussianNB(var_smoothing=best_vs)
    best_nb.fit(np.vstack([X_train, X_valid]), np.hstack([y_train, y_valid]))

    with open(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"), "wb") as f:
        pickle.dump(best_nb, f)
    return best_nb

# MODEL 3: DECISION TREE (DT)
def train_dt():
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }
    metrics = {}
    # Nested loops to find the best tree structure
    for c in param_grid['criterion']:
        for md in param_grid['max_depth']:
            for mss in param_grid['min_samples_split']:
                for msl in param_grid['min_samples_leaf']:
                    dt = DecisionTreeClassifier(criterion=c, max_depth=md, min_samples_split=mss, min_samples_leaf=msl)
                    dt.fit(X_train, y_train)
                    y_val_pred = dt.predict(X_valid)
                    metrics[(c, md, mss, msl)] = f1_score(y_valid, y_val_pred, average='macro')

    best_params = max(metrics, key=metrics.get)
    best_dt = DecisionTreeClassifier(criterion=best_params[0], max_depth=best_params[1],
                                     min_samples_split=best_params[2], min_samples_leaf=best_params[3])
    best_dt.fit(np.vstack([X_train, X_valid]), np.hstack([y_train, y_valid]))

    with open(os.path.join(MODEL_DIR, "decision_tree_model.pkl"), "wb") as f:
        pickle.dump(best_dt, f)
    return best_dt

# MODEL 4: SUPPORT VECTOR MACHINE (SVM)
def train_svm():
    # SVM finds the best boundary between classes
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01], 'kernel': ['linear', 'rbf']}
    metrics = {}
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            for kernel in param_grid['kernel']:
                svm = SVC(C=C, gamma=gamma, kernel=kernel, probability=True)
                svm.fit(X_train, y_train)
                y_val_pred = svm.predict(X_valid)
                metrics[(C, gamma, kernel)] = f1_score(y_valid, y_val_pred, average='macro')

    best_params = max(metrics, key=metrics.get)
    best_svm = SVC(C=best_params[0], gamma=best_params[1], kernel=best_params[2], probability=True)
    best_svm.fit(np.vstack([X_train, X_valid]), np.hstack([y_train, y_valid]))

    with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "wb") as f:
        pickle.dump(best_svm, f)
    return best_svm

# MODEL 5: NEURAL NETWORK (NN)
def train_nn():
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],  # Brain size (number of neurons)
        'alpha': [0.0001, 0.001],  # Regularization to prevent overfitting
        'learning_rate': ['constant', 'adaptive']
    }
    metrics = {}

    # Find best parameters
    for hls in param_grid['hidden_layer_sizes']:
        for alpha in param_grid['alpha']:
            for lr in param_grid['learning_rate']:
                nn = MLPClassifier(hidden_layer_sizes=hls, alpha=alpha, learning_rate=lr, max_iter=500, random_state=42)
                nn.fit(X_train, y_train)
                y_val_pred = nn.predict(X_valid)
                metrics[(hls, alpha, lr)] = f1_score(y_valid, y_val_pred, average='macro')

    best_params = max(metrics, key=metrics.get)

    # Train final model with early stopping to record history
    best_nn = MLPClassifier(hidden_layer_sizes=best_params[0], alpha=best_params[1], learning_rate=best_params[2], max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)

    best_nn.fit(np.vstack([X_train, X_valid]), np.hstack([y_train, y_valid]))

    with open(os.path.join(MODEL_DIR, "neural_net_model.pkl"), "wb") as f:
        pickle.dump(best_nn, f)

    # Save loss curve
    training_history = {
        'loss': best_nn.loss_curve_,
        'validation_scores': best_nn.validation_scores_ if hasattr(best_nn, 'validation_scores_') else []
    }
    with open(os.path.join(DATA_PICKLE_DIR, "nn_training_history.pkl"), "wb") as f:
        pickle.dump(training_history, f)

    print(f"Neural Net - Best params: {best_params}")
    return best_nn

# FIND THE WINNER
models = {
    'knn': train_knn(),
    'naive_bayes': train_nb(),
    'decision_tree': train_dt(),
    'svm': train_svm(),
    'neural_net': train_nn()
}

best_score = 0
best_model_name = ""
best_model = None

# Evaluate all models on the test set
for name, model in models.items():
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    print(f"{name}: Test F1-score = {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

# Save the winner
with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

print(f"\nWinner: {best_model_name} with Test F1-score = {best_score:.4f}")