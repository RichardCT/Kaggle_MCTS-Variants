import pandas as pd
import numpy as np
import optuna
import joblib
import yaml
import logging
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm.callback import record_evaluation
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from itertools import chain
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def load_data_preprocess():
    # Step 1: Load the data
    df = pd.read_csv('train.csv')

    # Drop non-feature columns
    df = df.drop(columns=['Id'])

    # Select relevant columns and fill NaN values
    X_selected = df.iloc[:, 3:807]
    X_selected = X_selected.fillna(X_selected.mean())

    # Debug: Check initial NaN count
    logging.info("NaN count before processing:", X_selected.isna().sum().sum())

    # Fill missing values, handle all-NaN columns and infinities
    X_selected = X_selected.fillna(X_selected.mean())  # Fill NaNs with column means
    X_selected = X_selected.fillna(0)  # Fill remaining NaNs with 0
    X_selected = X_selected.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    X_selected = X_selected.fillna(0)  # Handle infinities turned into NaNs

    # Confirm no NaN values
    assert X_selected.isna().sum().sum() == 0, "NaN values still present after fillna."

    # Step 2: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Step 3: Apply PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    joblib.dump(pca, 'pca_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    """
    # Load preprocessing objects and model
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_model.pkl')

    model = xgb.Booster()
    model.load_model('xgboost_model_train.json')

    # Standardize the test data
    test_features_scaled = scaler.transform(test_features)

    # Apply PCA transformation
    test_features_pca = pca.transform(test_features_scaled)

    # Convert to DMatrix for XGBoost
    test_data = xgb.DMatrix(test_features_pca)

    # Make predictions
    predictions = model.predict(test_data)
    """

    X_pca_df = pd.DataFrame(X_pca, columns=[f"pca_{i+1}" for i in range(X_pca.shape[1])])
    #test_pca_df = pd.DataFrame(test_scaled, columns=[f"pca_{i+1}" for i in range(X_pca.shape[1])])

    # Align the index with the original DataFrame (if needed)
    X_pca_df.index = X_scaled.index if hasattr(X_scaled, 'index') else range(len(X_pca_df))
    #test_pca_df.index = test_scaled.index if hasattr(test_scaled, 'index') else range(len(test_pca_df))

    y = df['utility_agent1']
    y_wins = df['num_wins_agent1']
    y_losses = df['num_losses_agent1']

    agents_train = df[['agent1', 'agent2']]
    pca_df = pd.concat([agents_train, X_pca_df], axis=1)

    return pca_df, y

def categorize_agents(pca_df):
    def create_agent_map(row):
        selection_map = {
            'UCB1': 1,
            'UCB1GRAVE': 2,
            'ProgressiveHistory': 3,
            'UCB1Tuned': 4
        }
        
        playout_map = {
            'Random200': 1,
            'MAST': 2,
            'NST': 3
        }

        parts = row.split('-')

        selection = parts[1]
        exploration_const = float(parts[2])
        playout = parts[3]
        score_bounds = parts[4] == 'true'

        # Convert categorical values to numeric values
        selection_value = selection_map.get(selection)
        playout_value = playout_map.get(playout)
        score_bounds_value = 1 if score_bounds else 0

        # Create the embedding map
        return [selection_value, exploration_const, playout_value, score_bounds_value]

    # Apply the function to each row in the DataFrame
    pca_df[['agent1_emb1', 'agent1_emb2', 'agent1_emb3', 'agent1_emb4']] = pd.DataFrame(
        agents_train['agent1'].apply(create_agent_map).tolist(), index=pca_df.index
    )
    pca_df[['agent2_emb1', 'agent2_emb2', 'agent2_emb3', 'agent2_emb4']] = pd.DataFrame(
        agents_train['agent2'].apply(create_agent_map).tolist(), index=pca_df.index
    )
    pca_df = pca_df.drop(columns=['agent1', 'agent2'])

    return pca_df

# Step 3: Training models (simplified)
# Example training loop for strategy combinations
def train_model_gbm(params_or_trial, num_boost_round, X, y, is_optuna=False):
    """
    Trains a LightGBM model using fixed parameters or Optuna trial parameters.

    Args:
        params_or_trial (dict or optuna.trial.Trial): Fixed parameters dictionary or Optuna trial.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.ndarray): Target labels.
        is_optuna (bool): Flag to indicate if using Optuna for hyperparameter optimization.

    Returns:
        For Optuna: Validation score (e.g., RMSE) for the trial.
        For regular training: Tuple (y_pred, y_test, lgb_evals_result).
    """
    # Step 1: Split into training and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # Step 2: Further split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=40)

    # Step 3: Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Initialize empty evaluation result storage
    lgb_evals_result = {}

    # Determine parameters
    if is_optuna:  # If using Optuna
        trial = params_or_trial  # The 'params_or_trial' is a trial object
        params = {
            'objective': 'regression',  # Regression task
            'metric': 'rmse',           # Evaluation metric
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 7, 255),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        }
        num_boost_round = trial.suggest_int('num_boost_round', 50, 500)
    else:  # If using fixed parameters
        params = params_or_trial

    # Train the model
    lgbm_model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'eval'],
        callbacks=[lgb.early_stopping(stopping_rounds=10), record_evaluation(lgb_evals_result)]
    )

    # Predict on test set
    y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)

    if is_optuna:  # If using Optuna, return the evaluation score
        # Assume RMSE is the metric
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return rmse  # Objective to minimize
    else:  # For regular training, return predictions and evaluation results
        lgbm_model.save_model('lgbm_model_weights.json')
        logging.info("LGBM Full model weights saved!")
        return y_pred, y_test, lgb_evals_result

def train_model_xg(params_or_trial, num_boost_round, X, y, is_optuna=False):
    """
    Trains a XGBoost model using fixed parameters or Optuna trial parameters.

    Args:
        params_or_trial (dict or optuna.trial.Trial): Fixed parameters dictionary or Optuna trial.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.ndarray): Target labels.
        is_optuna (bool): Flag to indicate if using Optuna for hyperparameter optimization.

    Returns:
        For Optuna: Validation score (e.g., RMSE) for the trial.
        For regular training: Tuple (y_pred, y_test, lgb_evals_result).
    """
    # Split data into training and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # Further split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=40)

    # Create DMatrix for XGBoost
    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)
    test_data = xgb.DMatrix(X_test, label=y_test)

    # Initialize empty evaluation result storage
    xgb_evals_result = {}

    # Determine parameters
    if is_optuna:  # If using Optuna
        trial = params_or_trial  # The 'params_or_trial' is a trial object
            param = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': trial.suggest_loguniform('eta', 0.01, 0.3),  # Learning rate
                'max_depth': trial.suggest_int('max_depth', 3, 10),  # Tree depth
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Subsample ratio
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Column subsampling
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),  # L2 regularization
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0)  # L1 regularization
            }
        num_boost_round = trial.suggest_int('num_boost_round', 50, 500)
    else:  # If using fixed parameters
        params = params_or_trial

    # Train the model
    evals = [(train_data, 'train'), (test_data, 'eval')]
    xgb_model = xgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round, 
        evals=evals,
        evals_result=xgb_evals_result,
        verbose_eval=False
    )

    # Predict on test set
    y_pred = xgb_model.predict(X_test, num_iteration=lgbm_model.best_iteration)

    if is_optuna:  # If using Optuna, return the evaluation score
        # Assume RMSE is the metric
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return rmse  # Objective to minimize
    else:  # For regular training, return predictions and evaluation results
        xgb_model.save_model('xgb_model_weights.json')
        logging.info("XGB Full model weights saved!")
        return y_pred, y_test, xgb_evals_result

def train_model_cat(params_or_trial, num_boost_round, X, y, is_optuna=False):
    """
    Trains a CatBoost model using fixed parameters or Optuna trial parameters.

    Args:
        params_or_trial (dict or optuna.trial.Trial): Fixed parameters dictionary or Optuna trial.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.ndarray): Target labels.
        is_optuna (bool): Flag to indicate if using Optuna for hyperparameter optimization.

    Returns:
        For Optuna: Validation score (e.g., RMSE) for the trial.
        For regular training: Tuple (y_pred, y_test, lgb_evals_result).
    """
    # Split data into training and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # Further split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=40)

    # Determine parameters
    if is_optuna:  # If using Optuna
        trial = params_or_trial  # The 'params_or_trial' is a trial object
            params = {
                'loss_function': 'RMSE',  # Regression task
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),  # Equivalent to 'eta'
                'depth': trial.suggest_int('depth', 3, 10),  # Tree depth
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),  # L2 regularization (similar to `lambda`)
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),  # Controls subsampling randomness
                'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),  # Noise added to tree splits
                'border_count': trial.suggest_int('border_count', 32, 255),  # Number of splits for numerical features
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),  # Tree growth strategy
                'iterations': trial.suggest_int('num_boost_round', 50, 500)  # Number of boosting rounds
            }
    else:  # If using fixed parameters
        params = params_or_trial

    # Train the model
    cat_model = CatBoostRegressor(**params, verbose=0)  # Pass the parameters to the model
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    # Predict on test set
    y_pred = cat_model.predict(X_test)

    if is_optuna:  # If using Optuna, return the evaluation score
        # Assume RMSE is the metric
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return rmse  # Objective to minimize
    else:  # For regular training, return predictions and evaluation results
        cat_model.save_model('catboost_model_weights.json', format='json')
        logging.info("CatB Full model weights saved!")
        return y_pred, y_test, cat_model

def objective_functions(trial, model):
    # Map model types to their respective training functions
    mymodels = {
        'lightgbm': train_model_gbm,
        'xgboost': train_model_xg,
        'catboost': train_model_cat
    }

    # Check if the model exists in the dictionary
    if model not in mymodels:
        raise ValueError(f"Unsupported model type: {model}")

    # Call the corresponding function dynamically
    return mymodels[model](trial, X, y, is_optuna=True)

def load_params(file_path, model_type):
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    if model_type not in params:
        raise ValueError(f"Model type '{model_type}' not found in parameters file.")
    return params[model_type]

def print_rmse(y_pred, y_test, model):
    mse = mean_squared_error(y_pred, y_test_gbm)
    rmse = np.sqrt(mse)
    logging.info(f"{model} RMSE: ", rmse)

def run_initial_test(params, model_name, train_func, X, y):
    y_pred, y_test, results = train_func(params, X, y, is_optuna=False)
    print_rmse(y_pred, y_test, model_name)
    return y_pred, y_test, results

def finetune_model(model_name, X, y, n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_functions(trial, model=model_name), n_trials=n_trials)
    logging.info(f"Best parameters for {model_name}: {study.best_params}")
    return study.best_params

def plot_rmse(results_gbm, results_xg, results_cat, tag):
    xgb_train_rmse = results_xg['train']['rmse']
    xgb_eval_rmse = results_xg['eval']['rmse']

    lgb_train_rmse = results_gbm['train']['rmse']
    lgb_eval_rmse = results_gbm['eval']['rmse']

    cat_train_rmse = cat_model.get_evals_result()['learn']['RMSE']
    cat_eval_rmse = cat_model.get_evals_result()['validation']['RMSE']

    # Plot training RMSE
    plt.plot(xgb_train_rmse, label='XGBoost')
    plt.plot(lgb_train_rmse, label='LightGBM')
    plt.plot(cat_train_rmse, label='CatBoost')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title(f'Training {tag} RMSE')
    plt.legend()
    plt.grid()
    plt.savefig(f'training_rmse_plot_{tag}.png')
    plt.close()

    # Plot evaluation RMSE
    plt.plot(xgb_eval_rmse, label='XGBoost')
    plt.plot(lgb_eval_rmse, label='LightGBM')
    plt.plot(cat_eval_rmse, label='CatBoost')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title(f'Evaluation {tag} RMSE')
    plt.legend()
    plt.grid()
    plt.savefig(f'evaluation_rmse_plot_{tag}.png') 
    plt.close()

def main():
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    params_file = "params.yaml"
    gbm_params = load_params(file_path, "lightgbm")
    xg_params = load_params(file_path, "xgboost")
    cat_params = load_params(file_path, "catboost")

    X, y = load_data_preprocess

    # Run initial test with default params
    y_pred_gbm, y_test_gbm, results_gbm = run_initial_test(gbm_params, "LightGBM", train_model_gbm, X, y)
    y_pred_xg, y_test_xg, results_xg = run_initial_test(xg_params, "XGBoost", train_model_xg, X, y)
    y_pred_cat, y_test_cat, results_cat = run_initial_test(cat_params, "CatBoost", train_model_cat, X, y)
    combined_pred = (y_pred_gbm + y_pred_xg + y_pred_cat)/3
    print_rmse(combined_pred, y_test_cat, 'Combined')
    plot_rmse(results_gbm, results_xg, results_cat, "initial")

    # Finetune parameters for all models
    gbm_best_params = finetune_model('lightgbm', X, y, n_trials=50)
    xg_best_params = finetune_model('xgboost', X, y, n_trials=50)
    cat_best_params = finetune_model('catboost', X, y, n_trials=50)

    # Run with optimized parameters
    y_pred_gbm, y_test_gbm, results_gbm = run_initial_test(gbm_params, "LightGBM", train_model_gbm, X, y)
    y_pred_xg, y_test_xg, results_xg = run_initial_test(xg_params, "XGBoost", train_model_xg, X, y)
    y_pred_cat, y_test_cat, results_cat = run_initial_test(cat_params, "CatBoost", train_model_cat, X, y)
    combined_pred = (y_pred_gbm + y_pred_xg + y_pred_cat)/3
    print_rmse(combined_pred, y_test_cat, 'Combined')
    plot_rmse(results_gbm, results_xg, results_cat, "optimized")

if __name__ == '__main__':
    main()
