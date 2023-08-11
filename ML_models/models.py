import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

# Defines all the models except the AttentiveFP
def linear_regression(x_train, x_test, y_train, y_test):
    l_reg = LinearRegression()
    folds = KFold(n_splits=10, shuffle=True, random_state=100)
    rmse_score = cross_val_score(l_reg, x_train, y_train, scoring='neg_mean_squared_error', cv=folds)
    param = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    # define model
    model = Ridge(tol=0.1)
    search = GridSearchCV(model, param, scoring='neg_mean_squared_error', n_jobs=-1, cv=folds, refit=True)
    search.fit(x_train, y_train)
    train_result = mean_squared_error(search.predict(x_train), y_train)
    test_result = mean_squared_error(search.predict(x_test), y_test)
    search.predict(x_test)
    return rmse_score, train_result, test_result, search.best_params_, search.cv_results_


def svm(x_train, x_test, y_train, y_test):
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 0.01, 10, 0.001],
        'degree': [1, 2, 3],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    }
    svm_reg = SVR()
    grid_fit = RandomizedSearchCV(estimator=svm_reg, scoring='neg_mean_squared_error', param_distributions=param_grid, n_jobs=-2,
                            refit=True, verbose=40)
    grid_fit.fit(x_train, y_train)
    train_result = mean_squared_error(grid_fit.predict(x_train), y_train)
    test_result = mean_squared_error(grid_fit.predict(x_test), y_test)
    return train_result, test_result, grid_fit.best_params_, grid_fit.cv_results_


def random_forest(x_train, x_test, y_train, y_test):
    n_estimators = [5, 10, 20, 30, 40, 50, 100]  # number of trees in the random forest
    max_features = ['sqrt']  # number of features in consideration at every split
    max_depth = [int(x) for x in np.linspace(10, 120, num=12)]  # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 4, 6, 10, 20, 40, 60]  # minimum sample number to split a node
    min_samples_leaf = [1, 3, 5]  # minimum sample number that can be stored in a leaf node
    bootstrap = [True, False]  # method used to sample data points

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=5, verbose=2, random_state=35, n_jobs=-1, refit=True)
    rf_random.fit(x_train, y_train)
    train_result = mean_squared_error(rf_random.predict(x_train), y_train)
    test_result = mean_squared_error(rf_random.predict(x_test), y_test)
    return train_result, test_result, rf_random.best_params_, rf_random.cv_results_


def light_gbm(x_train, x_test, y_train, y_test):
    params = {'learning_rate': np.linspace(0.0001, 0.5, 10),
              'boosting_type': ['gbdt', 'dart', 'goss'],
              'num_leaves': np.linspace(20, 300, 5,dtype='int64'),
              #'min_data_in_leaf': np.linspace(10, 100, 5,dtype='int64'),
              'max_depth': np.linspace(5, 300, 5, dtype='int64')}  # initialize parameters

    # This parameter defines the number of HP points to be tested
    n_HP_points_to_test = 100
    clf = LGBMRegressor(random_state=314, silent=True, n_jobs=-2,objective='regression')
    gs = RandomizedSearchCV(
        estimator=clf, param_distributions=params,
        n_iter=n_HP_points_to_test,
        cv=3,
        refit=True,
        random_state=314,
        verbose=40)
    gs.fit(x_train, y_train)
    train_result = mean_squared_error(gs.predict(x_train), y_train)
    test_result = mean_squared_error(gs.predict(x_test), y_test)
    return train_result, test_result, gs.best_params_, gs.cv_results_
