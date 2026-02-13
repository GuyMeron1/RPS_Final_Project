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

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_PICKLE_DIR = os.path.join("Data", "pickles")
MODEL_DIR = "Models"
os.makedirs(MODEL_DIR, exist_ok=True)

with open(f"{DATA_PICKLE_DIR}/X_train.pkl", "rb") as f: X_train = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/y_train.pkl", "rb") as f: y_train = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/X_valid.pkl", "rb") as f: X_valid = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/y_valid.pkl", "rb") as f: y_valid = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/X_test.pkl", "rb") as f: X_test = pickle.load(f)
with open(f"{DATA_PICKLE_DIR}/y_test.pkl", "rb") as f: y_test = pickle.load(f)

# Flatten sequences
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test  = X_test.reshape(X_test.shape[0], -1)

y_train = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train
y_valid = np.argmax(y_valid, axis=1) if y_valid.ndim > 1 else y_valid
y_test  = np.argmax(y_test, axis=1)  if y_test.ndim > 1 else y_test

def train_knn():
    param_grid = {'n_neighbors':[1,3,5,7], 'p':[1,2]}
    metrics = {}
    for k in param_grid['n_neighbors']:
        for p in param_grid['p']:
            knn = KNeighborsClassifier(n_neighbors=k, p=p)
            knn.fit(X_train, y_train)
            y_val_pred = knn.predict(X_valid)
            metrics[(k,p)] = f1_score(y_valid, y_val_pred, average='macro')
    best_params = max(metrics, key=metrics.get)
    best_knn = KNeighborsClassifier(n_neighbors=best_params[0], p=best_params[1])
    best_knn.fit(np.vstack([X_train, X_valid]), np.hstack([y_train, y_valid]))
    with open(os.path.join(MODEL_DIR, "knn_model.pkl"), "wb") as f: pickle.dump(best_knn, f)
    print(f"KNN - Best params: {best_params}")
    return best_knn
def train_nb():
    param_grid = {'var_smoothing':[1e-9,1e-8,1e-7]}
    metrics = {}
    for vs in param_grid['var_smoothing']:
        nb = GaussianNB(var_smoothing=vs)
        nb.fit(X_train, y_train)
        y_val_pred = nb.predict(X_valid)
        metrics[vs] = f1_score(y_valid, y_val_pred, average='macro')
    best_vs = max(metrics, key=metrics.get)
    best_nb = GaussianNB(var_smoothing=best_vs)
    best_nb.fit(np.vstack([X_train,X_valid]), np.hstack([y_train,y_valid]))
    with open(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"), "wb") as f: pickle.dump(best_nb, f)
    print(f"Naive Bayes - Best var_smoothing: {best_vs}")
    return best_nb
def train_dt():
    param_grid = {'criterion':['gini','entropy'], 'max_depth':[None,3,5],
                  'min_samples_split':[2,4], 'min_samples_leaf':[1,2]}
    metrics = {}
    for c in param_grid['criterion']:
        for md in param_grid['max_depth']:
            for mss in param_grid['min_samples_split']:
                for msl in param_grid['min_samples_leaf']:
                    dt = DecisionTreeClassifier(criterion=c, max_depth=md,
                                                min_samples_split=mss, min_samples_leaf=msl)
                    dt.fit(X_train, y_train)
                    y_val_pred = dt.predict(X_valid)
                    metrics[(c,md,mss,msl)] = f1_score(y_valid, y_val_pred, average='macro')
    best_params = max(metrics, key=metrics.get)
    best_dt = DecisionTreeClassifier(criterion=best_params[0], max_depth=best_params[1],
                                     min_samples_split=best_params[2], min_samples_leaf=best_params[3])
    best_dt.fit(np.vstack([X_train,X_valid]), np.hstack([y_train,y_valid]))
    with open(os.path.join(MODEL_DIR, "decision_tree_model.pkl"), "wb") as f: pickle.dump(best_dt, f)
    print(f"Decision Tree - Best params: {best_params}")
    return best_dt
def train_svm():
    param_grid = {'C':[0.1,1,10], 'gamma':[0.1,0.01], 'kernel':['linear','rbf']}
    metrics = {}
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            for kernel in param_grid['kernel']:
                svm = SVC(C=C,gamma=gamma,kernel=kernel,probability=True)
                svm.fit(X_train, y_train)
                y_val_pred = svm.predict(X_valid)
                metrics[(C,gamma,kernel)] = f1_score(y_valid, y_val_pred, average='macro')
    best_params = max(metrics, key=metrics.get)
    best_svm = SVC(C=best_params[0], gamma=best_params[1], kernel=best_params[2], probability=True)
    best_svm.fit(np.vstack([X_train,X_valid]), np.hstack([y_train,y_valid]))
    with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "wb") as f: pickle.dump(best_svm, f)
    print(f"SVM - Best params: {best_params}")
    return best_svm
def train_nn():
    param_grid = {'hidden_layer_sizes': [(50,),(100,)], 'alpha':[0.0001,0.001], 'learning_rate':['constant','adaptive']}
    metrics = {}
    for hls in param_grid['hidden_layer_sizes']:
        for alpha in param_grid['alpha']:
            for lr in param_grid['learning_rate']:
                nn = MLPClassifier(hidden_layer_sizes=hls, alpha=alpha, learning_rate=lr,
                                   max_iter=500, random_state=42)
                nn.fit(X_train, y_train)
                y_val_pred = nn.predict(X_valid)
                metrics[(hls,alpha,lr)] = f1_score(y_valid, y_val_pred, average='macro')
    best_params = max(metrics, key=metrics.get)
    best_nn = MLPClassifier(hidden_layer_sizes=best_params[0], alpha=best_params[1],
                            learning_rate=best_params[2], max_iter=500, random_state=42)
    best_nn.fit(np.vstack([X_train,X_valid]), np.hstack([y_train,y_valid]))
    with open(os.path.join(MODEL_DIR, "neural_net_model.pkl"), "wb") as f: pickle.dump(best_nn, f)
    print(f"Neural Net - Best params: {best_params}")
    return best_nn

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
for name, model in models.items():
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    print(f"{name}: Test F1-score = {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model: {best_model_name} with Test F1-score = {best_score:.4f}")