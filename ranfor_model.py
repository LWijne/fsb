############################# RANDOM FOREST #############################

#!/usr/bin/env python
# coding: utf-8

# Import essential packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob
import copy

# Sklearn
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

# Fairlearn
from fairlearn.metrics import demographic_parity_ratio

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
                    sloopschepen (pandas.DataFrame): DataFrame containing the relevant data.
    '''

    # Get the file
    sb_file = "sloopschepen_2024-01-22.csv"
    # Read
    sloopschepen = pd.read_csv(sb_file).drop("Unnamed: 0",axis=1)
    # Filter out active ships
    sloopschepen = sloopschepen[sloopschepen.dismantled == 1]
    # Get the relevant columns
    sloopschepen = sloopschepen[predictors+[target_col]].reset_index(drop=True)
    return sloopschepen

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

def data_pre_processing(sloopschepen):
    '''
    Missing value imputation and converting the sensitive attribute into a binary attribute.

            Parameters:
                    sloopschepen (pandas.DataFrame): DataFrame containing the data.

            Returns:
                    sloopschepen (pandas.DataFrame): DataFrame containing the preprocessed data.
    '''

    EOL_FOC_list = ["KNA", "COM", "PLW", "TUV", "TGO", "TZA", "VCT", "SLE"]
    for x in ["country_current_flag", "country_previous_flag"]:
        sloopschepen[x][~sloopschepen[x].isin(EOL_FOC_list)] = 0 # non-FOC for ship-breaking
        sloopschepen[x][sloopschepen[x].isin(EOL_FOC_list)] = 1 # FOC for ship-breaking acc to NGO SP
        
    # Replace NaN's with 'missing' for string columns
    for x in cat_columns:
        sloopschepen[x] = sloopschepen[x].fillna('missing')
        # Also replace values with "unknown" or similar to missing
        sloopschepen[x][sloopschepen[x].apply(str.lower).str.contains("unknown|unspecified")] = 'missing'
        sloopschepen[x][sloopschepen[x].apply(str.lower) == "unk"] = 'missing' 

    return sloopschepen


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

target_col = "beached" # Target

sensitive_col = "country_current_flag" # Sensitive attribute

random_state = 42 # Seed to be used for reproducibility 

standard_threshold = 0.5 # Classification threshold

thresholds = np.arange(0.05, 1.0, 0.05) # Thresholds for experiments

theta = 0.0 # Performance (0) - fairness (1)

# Define list of predictors to use
predictors = [
    "vessel_type",
    "gross_tonnage",
    "port_of_registry",
    "country_current_flag",
    "country_previous_flag",
    "years_since_final_flag_swap",
    "pop_current_flag",
    "gdpcap_current_flag",
    "speed",
    "age_in_months"
]

# Specify which predictors are numerical
num_columns = [
    "gross_tonnage",
    "years_since_final_flag_swap",
    "speed",
    "age_in_months",
    "pop_current_flag",
    "gdpcap_current_flag"
]

# Specify which predictors are categorical and need to be one-hot-encoded
cat_columns = [
    "vessel_type",
    "port_of_registry"
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

sloopschepen = read_data()
sloopschepen = data_pre_processing(sloopschepen)

# Prepare the data 
sloopschepen = data_prep(df=sloopschepen,
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

def cross_val_score_custom(model, X, y, s, cv=10):
    '''
    Evaluate the ROC AUC score by cross-validation.

            Parameters:
                    model (GridSearchReduction object): The model.
                    X (array-like): The training data.
                    y (array-like): The labels.
                    s (array-like): The sensitive attribute.
                    cv (int): Number of folds.

            Returns:
                    roc_auc (float): The ROC AUC score.
    '''
    
    # Create K-fold cross validation folds
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    auc_list = []
    sdp_list = []

    # Looping over the folds
    for trainset, testset in splitter.split(X,y):

        # Splitting and reparing the data, targets and sensitive attributes
        X_train_df = X[X.index.isin(trainset)]
        y_train_df = y[y.index.isin(trainset)]
        X_test_df = X[X.index.isin(testset)]
        y_test_df = y[y.index.isin(testset)]
        s_test = s[s.index.isin(testset)].astype(int)
        

        # Initializing and fitting the classifier
        cv = model
        cv.fit(X_train_df, y_train_df)

        # Final predictions
        y_pred_probs = cv.predict_proba(X_test_df).T[1]
        y_true = y_test_df

        auc_list.append(roc_auc_score(y_true,y_pred_probs))
        sdp_list.append(0.5 + abs(0.5 - roc_auc_score(s_test, y_pred_probs)))


    # Final results
    auc_list = np.array(auc_list)
    roc_auc = np.nanmean(auc_list, axis=0)
    sdp_list = np.array(sdp_list)
    strong_dem_par = np.nanmean(sdp_list, axis=0)
    return roc_auc, strong_dem_par

def best_model(trials):
    '''
    Retrieve the best model.

            Parameters:
                    trials (Trials object): Trials object.

            Returns:
                    trained_model (RandomForestClassifier object): The best model.
    '''
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    trained_model = best_trial_obj['result']['trained_model']
    return trained_model

def objective(params):
    model = RandomForestClassifier(
      n_estimators=params['n_estimators'],
      criterion=params['criterion'],
      max_depth=params['max_depth'],
      min_samples_split=params['min_samples_split'],
      min_samples_leaf=params['min_samples_leaf'],
      max_features=params['max_features'],
      bootstrap=params['bootstrap'],
      random_state=random_state,
      class_weight=params['class_weight']
    )
    pipeline = Pipeline([('column_transformer', ct), ('classifier', model)])
    roc_auc_y, roc_auc_s = cross_val_score_custom(
      pipeline,
      X_train_df,
      y_train_df,
      s_train,
      cv=K,
    )
    goal = (1-theta) * roc_auc_y - theta * roc_auc_s

    return {'loss': -goal, 'status': STATUS_OK, 'trained_model': model}


############################# Training the classifier, predictions and outcomes #############################

auc_plot_list, f1_plot_list, dpr_plot_list, sdp_plot_list = [], [], [], []

# Looping over the folds
for trainset, testset in sloopschepen["folds"].split(sloopschepen["X"],sloopschepen["y"]):

    # Splitting and reparing the data, targets and sensitive attributes
    X_train_df = sloopschepen["X"][sloopschepen["X"].index.isin(trainset)]
    y_train_df = sloopschepen["y"][sloopschepen["y"].index.isin(trainset)]
    X_test_df = sloopschepen["X"][sloopschepen["X"].index.isin(testset)]
    y_test_df = sloopschepen["y"][sloopschepen["y"].index.isin(testset)]
    s_train = X_train_df[sensitive_col]
    s_test = X_test_df[sensitive_col]
    X_train_df = X_train_df.drop(columns=[sensitive_col])
    X_test_df = X_test_df.drop(columns=[sensitive_col])
    
    params = {
        'n_estimators': hp.uniformint('n_estimators', 10, 1000, q=1.0),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('max_depth', [None, hp.uniformint('max_depth_int', 2, 200, q=1.0)]),
        'min_samples_split': hp.uniformint('min_samples_split', 2, 10, q=1.0),
        'min_samples_leaf': hp.uniformint('min_samples_leaf', 1, 10, q=1.0),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'bootstrap': hp.choice('bootstrap', [True, False]),
        'class_weight': hp.choice('class_weight', [None, 'balanced', 'balanced_subsample'])
    }

    trials = Trials()

    opt = fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials
    )

    # Initializing and fitting the classifier
    clf = best_model(trials)
    pipeline = Pipeline([('column_transformer', ct), ('classifier', clf)])
    pipeline.fit(X_train_df, y_train_df)

    # Final predictions
    y_pred_probs = pipeline.predict_proba(X_test_df).T[1]
    y_true = y_test_df

    auc_plot, f1_plot, dpr_plot, eor_plot, sdp_plot = [], [], [], [], []

    # Looping over the thresholds
    for t in thresholds:
        y_pred = [1 if pred>=t else 0 for pred in y_pred_probs] # Predictions for t

        # Adding all scores for this fold, for this threshold
        auc_plot.append(roc_auc_score(y_true,y_pred_probs))
        f1_plot.append(f1_score(y_true,y_pred))
        dpr_plot.append(demographic_parity_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=s_test))
        sdp_plot.append(strong_demographic_parity_score(s_test, y_pred_probs))

    # Final results for this fold (list for all thresholds)
    auc_plot_list.append(auc_plot)
    f1_plot_list.append(f1_plot)
    dpr_plot_list.append(dpr_plot)
    sdp_plot_list.append(sdp_plot)


# Final results
auc_plot_list = np.array(auc_plot_list)
auc_plot = np.nanmean(auc_plot_list, axis=0)
auc_std = np.nanstd(auc_plot_list, axis=0)

f1_plot_list = np.array(f1_plot_list)
f1_plot = np.nanmean(f1_plot_list, axis=0)
f1_std = np.nanstd(f1_plot_list, axis=0)

dpr_plot_list = np.array(dpr_plot_list)
dpr_plot = np.nanmean(dpr_plot_list, axis=0)
dpr_std = np.nanstd(dpr_plot_list, axis=0)

sdp_plot_list = np.array(sdp_plot_list)
sdp_plot = np.nanmean(sdp_plot_list, axis=0)
sdp_std = np.nanstd(sdp_plot_list, axis=0)
rev_sdp_plot = 1-sdp_plot


############################# Plot: performance and fairness measures against thresholds #############################

plt.plot(thresholds, auc_plot)
plt.fill_between(thresholds, auc_plot - auc_std, auc_plot + auc_std, alpha=0.2)

plt.plot(thresholds, f1_plot)
plt.fill_between(thresholds, f1_plot - f1_std, f1_plot + f1_std, alpha=0.2)

plt.plot(thresholds, dpr_plot)
plt.fill_between(thresholds, dpr_plot - dpr_std, dpr_plot + dpr_std, alpha=0.2)

plt.plot(thresholds, rev_sdp_plot)
plt.fill_between(thresholds, rev_sdp_plot - sdp_std, rev_sdp_plot + sdp_std, alpha=0.2)

plt.title("Performance and fairness measures (with sensitive attribute = 'country_current_flag')\nof Random Forest plotted against thresholds.")
plt.xlabel("Threshold")
plt.legend(["AUC", "F1", "DPR", "1-SDP"])
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.xticks(np.arange(0.0, 1.1, 0.1));

print("auc_rf =", [auc_plot[0]])
print("sdp_rf =", [sdp_plot[0]])
print("std_auc_rf =", [auc_std[0]])
print("std_sdp_rf =", [sdp_std[0]])


############################# Plot: AUC and SDP trade-off #############################

