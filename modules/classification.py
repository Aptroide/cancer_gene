import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import logging
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("optuna").disabled = True
from rich import print

import tensorflow as tf

def print_fancy_title(method: str):
    title = f"{method.upper()} "
    box_width = len(title) + 4  # Para ajustar el ancho del cuadro

    print("\n" + "═" * box_width)
    print(f"║{title.center(box_width - 2)}║")
    print("═" * box_width)


def preprocess_data(df: pd.DataFrame,
                     cluster_num: int,
                     test_size: float = 0.30,
                     val_size: float = 0.50,
                     random_state: int = 42):
    """
    Preprocesses the DataFrame for modeling:
    - Binarizes the 'Cluster' column into `label`.
    - Drops non-numeric 'Cancer' column if present.
    - Splits into train, test, and validation sets.
    - Applies MinMax scaling followed by standard scaling.
    Returns scaled splits and labels.
    """
    df_copy = df.copy()
    # Binary label for the specified cluster
    df_copy['label'] = df_copy['Cluster'].apply(lambda x: 1 if x == cluster_num else 0)
    df_copy = df_copy.drop(columns=['Cluster'])

    if 'Cancer' in df_copy.columns:
        df_copy = df_copy.drop(columns=['Cancer'])

    X = df_copy.drop(columns=['label'])
    y = df_copy['label']

    # Split into train, temp; then split temp into test and validation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    # Scaling pipeline: MinMax then Standard
    mm_scaler = MinMaxScaler()
    X_train_mm = mm_scaler.fit_transform(X_train)
    X_test_mm = mm_scaler.transform(X_test)
    X_val_mm = mm_scaler.transform(X_val)

    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train_mm)
    X_test_scaled = std_scaler.transform(X_test_mm)
    X_val_scaled = std_scaler.transform(X_val_mm)

    # Wrap back to DataFrame if needed elsewhere
    return (
        pd.DataFrame(X_train_scaled, columns=X.columns),
        pd.DataFrame(X_test_scaled, columns=X.columns),
        pd.DataFrame(X_val_scaled, columns=X.columns),
        y_train, y_test, y_val
    )


def train_classical_models(X_train, y_train, X_test, y_test, X_val, y_val):
    """
    Trains a suite of classical ML models and evaluates on test and validation sets.
    Returns two DataFrames (test results, validation results) sorted by F1 score,
    plus the name of the best validation model.
    """
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': LinearSVC(max_iter=10000),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(verbosity=0),
        'LightGBM': LGBMClassifier(verbose=-1),
        'CatBoost': CatBoostClassifier(verbose=0)
    }

    results_test, results_val = [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_val = model.predict(X_val)

        test_scores = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred_test),
            'Precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        }
        val_scores = {
            'Model': name,
            'Accuracy': accuracy_score(y_val, y_pred_val),
            'Precision': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
            'Recall': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
        }

        results_test.append(test_scores)
        results_val.append(val_scores)

    df_test = pd.DataFrame(results_test).sort_values(by='F1 Score', ascending=False)
    df_val = pd.DataFrame(results_val).sort_values(by='F1 Score', ascending=False)
    best_name = df_val.iloc[0]['Model']
    return df_test, df_val, best_name


def tune_neural_network(X_train, y_train, X_val, y_val, n_trials: int = 50, epochs: int = 50):
    """
    Runs Optuna hyperparameter tuning for a simple feed-forward neural network.
    Returns the best trial object.
    """
    def create_model(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        model = tf.keras.Sequential()
        for i in range(n_layers):
            n_units = trial.suggest_int(f'n_units_l{i}', 4, 128)
            model.add(tf.keras.layers.Dense(n_units, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def objective(trial):
        model = create_model(trial)
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            validation_data=(X_val, y_val),
            verbose=0
        )
        val_acc = max(model.history.history['val_accuracy'])
        return val_acc

    study = optuna.create_study(direction='maximize', study_name='nn_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_trial


def train_best_neural(best_trial, X_train, y_train, X_val, y_val, epochs: int = 50):
    """
    Builds and trains the neural network with the best hyperparameters.
    Returns the trained Keras model.
    """
    # Reconstruct the model architecture
    model = tf.keras.Sequential()
    for i in range(best_trial.params['n_layers']):
        units = best_trial.params[f'n_units_l{i}']
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_trial.params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=0
    )
    return model


def evaluate_model(model, X, y, set_name: str = ''):
    """
    Evaluates the model and prints metrics.
    Returns a dictionary of metrics.
    """
    preds = model.predict(X)
    if preds.ndim > 1:
        preds = preds.squeeze()
    y_pred = (preds > 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"Metrics for {set_name} set:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}


def run_pipeline(df: pd.DataFrame, cluster_num: int):
    """
    Runs the full pipeline:
      1. Preprocessing
      2. Classical models training + evaluation
      3. Neural net tuning and training
      4. Model evaluations and saving
    """

    # 1. Preprocess
    X_train, X_test, X_val, y_train, y_test, y_val = preprocess_data(df, cluster_num)

    # 2. Classical models
    df_test, df_val, best_classical = train_classical_models(
        X_train, y_train, X_test, y_test, X_val, y_val
    )
    print("\n=== Classical Models Test Results ===")
    print(df_test)
    print("\n=== Classical Models Validation Results ===")
    print(df_val)
    print(f"\nBest classical model: {best_classical}")

    print_fancy_title("Neural Network Tuning")
    print("Tuning neural network hyperparameters can take a while...")
    # 3. Neural network tuning
    best_trial = tune_neural_network(X_train, y_train, X_val, y_val)
    print(f"\nBest NN validation accuracy: {best_trial.value:.4f}")
    print("Best NN hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    # 4. Train best NN and evaluate
    best_nn = train_best_neural(best_trial, X_train, y_train, X_val, y_val)
    print()
    test_metrics = evaluate_model(best_nn, X_test, y_test, set_name='test')
    val_metrics = evaluate_model(best_nn, X_val, y_val, set_name='validation')

    # Return results if needed
    return {
        'classical_test': df_test,
        'classical_val': df_val,
        'best_classical': best_classical,
        'nn_best_trial': best_trial,
        'nn_test_metrics': test_metrics,
        'nn_val_metrics': val_metrics
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ML pipeline on clustered data.")
    parser.add_argument('data_path', type=str, help='Path to CSV file')
    parser.add_argument('cluster_num', type=int, help='Cluster number to binarize label')
    parser.add_argument('--output_model', type=str, default='best_optuna_model.keras',
                        help='Filename to save the best neural network')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    run_pipeline(df, args.cluster_num, output_model_path=args.output_model)
