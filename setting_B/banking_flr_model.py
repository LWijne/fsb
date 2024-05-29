############################# FAIR LOGISTIC REGRESSION #############################

#!/usr/bin/env python
# coding: utf-8

# Import essential packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import urllib.request
import joblib

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# AIF360
from aif360.sklearn.inprocessing.grid_search_reduction import GridSearchReduction

# HyperOpt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Path
sys.path.append('../')

# No warnings
pd.options.mode.chained_assignment = None

from warnings import filterwarnings
filterwarnings('ignore')

############################# Data pre-processing and feature selection functions #############################

def read_data():
    '''
    Reads the file and filters the relevant information.

            Parameters:


            Returns:
                    datasets (pandas.DataFrame): DataFrame containing the relevant data.
    '''

    datasets_url = "https://github.com/pereirabarataap/fair_tree_classifier/raw/main/datasets.pkl"    
    datasets = joblib.load(urllib.request.urlopen(datasets_url))
    datasets = datasets['bank_marketing']
    datasets = pd.concat([datasets["X"], datasets["y"].to_frame(), datasets["z"]["marital"].to_frame()], axis=1)
    return datasets

class MissIndicator():
    
    def __init__(self):
        self.is_fit = False
        
    def fit(self, X, y=None):
        self.mi = MissingIndicator(sparse=False, error_on_new=False)
        self.mi.fit(X)
        
    def transform(self, X, y=None):
        return np.concatenate([X, self.mi.transform(X)], axis=1)
    
    def fit_transform(self, X, y=None):
        self.mi = MissingIndicator(sparse=False, error_on_new=False)
        self.mi.fit(X)
        return np.concatenate([X, self.mi.transform(X)], axis=1)


class Clamper():
    
    def __init__(self):
        self.is_fit = False
        self.values_to_keep = {}
        
    def _get_values_to_keep_from_value_counts(self, value_counts):
        values = value_counts.keys()
        counts = value_counts.values.astype(int)
        count_p = counts / sum(counts)
        min_p_increase = 1/len(values)
        index_to_keep = np.argmin(abs(count_p - min_p_increase))
        values_to_keep = values[:index_to_keep]

        return values_to_keep
    
    def fit_transform(self, X, y=None):
        transformed_X = copy.deepcopy(X)
        for column in X.columns:
            self.values_to_keep[column] = self._get_values_to_keep_from_value_counts(
                X[column].value_counts()
            )
            transformed_X[column].loc[
                ~(transformed_X[column].isin(self.values_to_keep[column]))
            ] = "other"
        self.is_fit = True
        return transformed_X
    
    def fit(self, X, y=None):
        for column in X.columns:
            self.values_to_keep[column] = self._get_values_to_keep_from_value_counts(
                X[column].value_counts()
            )
        self.is_fit = True
        
    def transform(self, X, y=None):
        transformed_X = copy.deepcopy(X)
        for column in X.columns:
            transformed_X[column].loc[
                ~(transformed_X[column].isin(self.values_to_keep[column]))
            ] = "other"
        
        return transformed_X

def data_pre_processing(bank_marketing):
    '''
    Missing value imputation and converting the sensitive attribute into a binary attribute.

            Parameters:
                    bank_marketing (pandas.DataFrame): DataFrame containing the data.

            Returns:
                    bank_marketing (pandas.DataFrame): DataFrame containing the preprocessed data.
    '''

    bank_marketing["marital"][bank_marketing["marital"] == "married"] = 1 # Married
    bank_marketing["marital"][bank_marketing["marital"] == "non-married"] = 0 # Not married
        
    # Replace NaN's with 'missing' for string columns
    for x in cat_columns:
        bank_marketing[x] = bank_marketing[x].fillna('missing') 
        # Also replace values with "unknown" to missing
        bank_marketing[x][bank_marketing[x].apply(str.lower).str.contains("unknown")] = 'missing'

    return bank_marketing


def data_prep(df, K, predictors, target_col):
    '''
    Prepares a dictionary of X, y and folds.

            Parameters:
                    df (pandas.DataFrame): DataFrame containing the data.
                    K (int): Number of cross validation folds.
                    predictors (list): List of predictor columns.
                    target_col (str): The target column.

            Returns:
                    data_prep_dict (dict): Dictionary with X, y and folds.
    '''
    # Select targets from development data
    targets = df[target_col].reset_index(drop=True)
    
    # Select predictors from data
    df = df[predictors].reset_index(drop=True)
    
    # Create K-fold cross validation folds
    splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    
    # Create result dictionary
    data_prep_dict = {}
    data_prep_dict["X"] = df
    data_prep_dict["y"] = targets
    data_prep_dict['folds'] = splitter
    
    return data_prep_dict  


############################# Parameters #############################

K = 10 # K-fold CV

hyperopt_evals = 100 # Max number of evaluations for HPO

target_col = "y" # Target

sensitive_col = "marital" # Sensitive attribute

random_state = 42 # Seed to be used for reproducibility 

standard_threshold = 0.5 # Classification threshold

thresholds = np.arange(0.05, 1.0, 0.05) # Thresholds for experiments

theta = 0.0 # Performance (0) - fairness (1)

theta_list = np.arange(0.0, 1.1, 0.1) # Thetas for experiments

# Define list of predictors to use
predictors = [
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "job",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "marital"
]

# Specify which predictors are numerical
num_columns = [
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

# Specify which predictors are categorical and need to be one-hot-encoded
cat_columns = [
    "job",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]

num_transformer = Pipeline([
    ('scaler', RobustScaler()),
    ("mindic", MissIndicator()),
    ('imputer', SimpleImputer())
])
cat_transformer = Pipeline([
    ("clamper", Clamper()),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

ct = ColumnTransformer([
    ('num_transformer', num_transformer, num_columns),
    ('cat_transformer', cat_transformer, cat_columns)
],
    remainder='passthrough'
)

bank_marketing = read_data()
bank_marketing = data_pre_processing(bank_marketing)

# Prepare the data 
bank_marketing = data_prep(df=bank_marketing,
                   K=K,
                   predictors=predictors,
                   target_col=target_col)


def strong_demographic_parity_score(s, y_prob):
    '''
    Returns the strong demographic parity score.

            Parameters:
                    s (array-like): The sensitive features over which strong demographic parity should be assessed.
                    y_prob (array-like): The predicted probabilities returned by the classifier.

            Returns:
                    sdp (float): The strong demographic parity score.
    '''
    y_prob = np.array(y_prob)
    s = np.array(s)
    if len(s.shape)==1:
        s = s.reshape(-1,1)
    
    sensitive_aucs = []
    for s_column in range(s.shape[1]):
        if len(np.unique(s[:, s_column]))==1:
            sensitive_aucs.append(1) 
        else:
            sens_auc = 0
            for s_unique in np.unique(s[:, s_column]):
                s_bool = (s[:, s_column]==s_unique)
                auc = roc_auc_score(s_bool, y_prob)
                auc = max(1-auc, auc)
                sens_auc = max(sens_auc, auc)
            sensitive_aucs.append(sens_auc)
    
    s_auc = sensitive_aucs[0] if len(sensitive_aucs)==1 else sensitive_aucs
    sdp = abs(2*s_auc-1)
    return sdp


############################# HPO #############################

def cross_val_score_custom(model, X, y, cv=10):
    '''
    Evaluate the ROC AUC score by cross-validation.

            Parameters:
                    model (GridSearchReduction object): The model.
                    X (array-like): The training data.
                    y (array-like): The labels.
                    cv (int): Number of folds.

            Returns:
                    auc_perf (float): The ROC AUC score of the predictions and the labels.
                    auc_fair (float): The ROC AUC score of the predictions and the sensitive attribute.
    '''
    
    # Create K-fold cross validation folds
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    auc_perf_list = []
    auc_fair_list = []

    s = X[sensitive_col]
    splitter_y = y.astype(str) + s.astype(str)

    # Looping over the folds
    for trainset, testset in splitter.split(X,splitter_y):

        # Splitting and reparing the data, targets and sensitive attributes
        X_train = X[X.index.isin(trainset)]
        y_train = y[y.index.isin(trainset)]
        X_test = X[X.index.isin(testset)]
        y_test = y[y.index.isin(testset)]
        s_test = X_test[sensitive_col].astype(int)
        
        X_train = ct.fit_transform(X_train)

        columns = list(ct.transformers_[0][1][2].get_feature_names_out())+list(ct.transformers_[1][1][1].get_feature_names_out())+['marital']

        X_train = pd.DataFrame(X_train, columns=columns)
        X_test = pd.DataFrame(ct.transform(X_test), columns=columns)

        # Initializing and fitting the classifier
        cv = model
        cv.fit(X_train, y_train)

        # Final predictions
        y_pred_probs = cv.predict_proba(X_test).T[1]
        y_true = y_test

        auc_perf_list.append(roc_auc_score(y_true,y_pred_probs))
        auc_fair_list.append(0.5 + abs(0.5 - roc_auc_score(s_test, y_pred_probs)))


    # Final results
    auc_perf_list = np.array(auc_perf_list)
    auc_perf = np.nanmean(auc_perf_list, axis=0)
    auc_fair_list = np.array(auc_fair_list)
    auc_fair = np.nanmean(auc_fair_list, axis=0)
    return auc_perf, auc_fair

def best_model(trials):
    '''
    Retrieve the best model.

            Parameters:
                    trials (Trials object): Trials object.

            Returns:
                    trained_model (dict): The best model parameters.
    '''
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    trained_model = best_trial_obj['result']['trained_model']
    return trained_model

def objective(params):
    '''
    Retrieve the loss for a model created by certain parameters.

            Parameters:
                    params (dict): The parameters to create the model.

            Returns:
                    (dict): The loss, status and trained model parameters.
    '''
    model = GridSearchReduction(
      prot_attr=sensitive_col,
      estimator=LogisticRegression(random_state=random_state, 
                                   penalty=params['penalty'], 
                                   tol=params['tol'], 
                                   C=params['C'], 
                                   fit_intercept=params['fit_intercept'], 
                                   class_weight=params['class_weight'], 
                                   solver='saga', 
                                   max_iter=params['max_iter'], 
                                   l1_ratio=params['l1_ratio']),
      constraints="DemographicParity",
      constraint_weight=params['constraint_weight'],
      grid_size=params['grid_size'],
      grid_limit=params['grid_limit'],
      drop_prot_attr=True,
      loss=params['loss']
    )
    roc_auc_y, roc_auc_s = cross_val_score_custom(
      model,
      X_train_df,
      y_train_df,
      cv=K
    )
    goal = (1-theta) * roc_auc_y - theta * roc_auc_s
    
    return {'loss': -goal, 'status': STATUS_OK, 'trained_model': params}


############################# Training the classifier, predictions and outcomes #############################

def fair_logistic_regression_(best_flr_model_params):
    '''
    Computes the average and std of AUC and SDP over K folds.

            Parameters:
                    best_flr_model_params (dict): The parameters of the best model.

            Returns:
                    roc_auc (np.array): The average of the ROC AUC list for each theta.
                    strong_dp (np.array): The average of the strong demographic parity list for each theta.
                    std_auc (np.array): The standard deviation of the ROC AUC list for each theta.
                    std_sdp (np.array): The standard deviation of the strong demographic parity list for each theta.
    '''

    roc_auc_list_2d = np.array([])
    strong_dp_list_2d = np.array([])
    
    y = bank_marketing["y"]
    s = bank_marketing["X"][sensitive_col]
    splitter_y = y.astype(str) + s.astype(str)

    # Looping over the folds
    for trainset, testset in bank_marketing["folds"].split(bank_marketing["X"],splitter_y):

        roc_auc_list_1d = np.array([])
        strong_dp_list_1d = np.array([])
        
        global X_train_df
        global y_train_df

        # Splitting and preparing the data, targets and sensitive attributes
        X_train_df = bank_marketing["X"][bank_marketing["X"].index.isin(trainset)]
        y_train_df = bank_marketing["y"][bank_marketing["y"].index.isin(trainset)]
        X_test_df = bank_marketing["X"][bank_marketing["X"].index.isin(testset)]
        y_test_df = bank_marketing["y"][bank_marketing["y"].index.isin(testset)]
        
        s_test = X_test_df[sensitive_col].astype(int)
        
        X_train_df = ct.fit_transform(X_train_df)
        
        columns = list(ct.transformers_[0][1][2].get_feature_names_out())+list(ct.transformers_[1][1][1].get_feature_names_out())+['marital']

        X_train_df = pd.DataFrame(X_train_df, columns=columns)
        X_test_df = pd.DataFrame(ct.transform(X_test_df), columns=columns)

        for th in theta_list:
            # Initializing and fitting the classifier
            
            cv = GridSearchReduction(
            prot_attr=sensitive_col,
            estimator=LogisticRegression(random_state=random_state, 
                                        penalty=best_flr_model_params['penalty'], 
                                        tol=best_flr_model_params['tol'], 
                                        C=best_flr_model_params['C'], 
                                        fit_intercept=best_flr_model_params['fit_intercept'], 
                                        class_weight=best_flr_model_params['class_weight'], 
                                        solver='saga', 
                                        max_iter=best_flr_model_params['max_iter'], 
                                        l1_ratio=best_flr_model_params['l1_ratio']),
            constraints="DemographicParity",
            constraint_weight=th,
            grid_size=best_flr_model_params['grid_size'],
            grid_limit=best_flr_model_params['grid_limit'],
            drop_prot_attr=True,
            loss=best_flr_model_params['loss']
            )

            cv.fit(X_train_df, y_train_df)

            # Final predictions
            y_pred_probs = cv.predict_proba(X_test_df).T[1]
            y_true = y_test_df

            roc_auc_list_1d = np.append(roc_auc_list_1d, roc_auc_score(y_true, y_pred_probs))
            strong_dp_list_1d = np.append(strong_dp_list_1d, strong_demographic_parity_score(s_test, y_pred_probs))
        
        roc_auc_list_2d = np.vstack([roc_auc_list_2d, roc_auc_list_1d]) if roc_auc_list_2d.size else roc_auc_list_1d
        strong_dp_list_2d = np.vstack([strong_dp_list_2d, strong_dp_list_1d]) if strong_dp_list_2d.size else strong_dp_list_1d
    
        print("Completed a fold")

    
    return np.mean(roc_auc_list_2d, axis=0), np.mean(strong_dp_list_2d, axis=0), np.std(roc_auc_list_2d, axis=0), np.std(strong_dp_list_2d, axis=0)

y = bank_marketing["y"]
s = bank_marketing["X"][sensitive_col]
splitter_y = y.astype(str) + s.astype(str)

best_auc = 0.0
best_flr_model_params = None

# Looping over the folds
for trainset, testset in bank_marketing["folds"].split(bank_marketing["X"],splitter_y):
    
    global X_train_df
    global y_train_df

    # Splitting and preparing the data, targets and sensitive attributes
    X_train_df = bank_marketing["X"][bank_marketing["X"].index.isin(trainset)]
    y_train_df = bank_marketing["y"][bank_marketing["y"].index.isin(trainset)]
    X_test_df = bank_marketing["X"][bank_marketing["X"].index.isin(testset)]
    y_test_df = bank_marketing["y"][bank_marketing["y"].index.isin(testset)]
    
    params = {
        'penalty': hp.choice('penalty', ["l1", "l2", "elasticnet", None]),
        'constraint_weight': hp.choice('constraint_weight', [0.0]),
        'grid_size': hp.uniformint('grid_size', 2, 50, q=1.0),
        'grid_limit': hp.uniform('grid_limit', 0.4, 10.0),
        'loss': hp.choice('loss', ["ZeroOne", "Square", "Absolute"]),
        'tol': hp.uniform('tol', 0.00001, 0.001),
        'C': hp.uniform('C', 0.01, 10.0),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'max_iter': hp.uniformint('max_iter', 10, 1000, q=1.0),
        'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0)
    }

    trials = Trials()

    opt = fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        max_evals=hyperopt_evals,
        trials=trials
    )

    model_params = best_model(trials)
    
    s_test = X_test_df[sensitive_col].astype(int)
    
    X_train_df = ct.fit_transform(X_train_df)
    
    columns = list(ct.transformers_[0][1][2].get_feature_names_out())+list(ct.transformers_[1][1][1].get_feature_names_out())+['marital']

    X_train_df = pd.DataFrame(X_train_df, columns=columns)
    X_test_df = pd.DataFrame(ct.transform(X_test_df), columns=columns)

    cv = GridSearchReduction(
    prot_attr=sensitive_col,
    estimator=LogisticRegression(random_state=random_state, 
                                penalty=model_params['penalty'], 
                                tol=model_params['tol'], 
                                C=model_params['C'], 
                                fit_intercept=model_params['fit_intercept'], 
                                class_weight=model_params['class_weight'], 
                                solver='saga', 
                                max_iter=model_params['max_iter'], 
                                l1_ratio=model_params['l1_ratio']),
    constraints="DemographicParity",
    constraint_weight=model_params['constraint_weight'],
    grid_size=model_params['grid_size'],
    grid_limit=model_params['grid_limit'],
    drop_prot_attr=True,
    loss=model_params['loss']
    )

    cv.fit(X_train_df, y_train_df)

    # Final predictions
    y_pred_probs = cv.predict_proba(X_test_df).T[1]
    y_true = y_test_df

    roc_auc = roc_auc_score(y_true, y_pred_probs)
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_flr_model_params = model_params


auc_list, sdp_list, std_auc_list, std_sdp_list = fair_logistic_regression_(best_flr_model_params)

############################# Plot: AUC and SDP trade-off #############################

print("auc_flr_setB_bank =", auc_list.tolist())
print("sdp_flr_setB_bank =", sdp_list.tolist())
print("std_auc_flr_setB_bank =", std_auc_list.tolist())
print("std_sdp_flr_setB_bank =", std_sdp_list.tolist())

