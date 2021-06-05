#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from dms.models import AveragingModels
# import xgboost as xgb
# import lightgbm as lgb


def rmsle_cv(model, train, y_train):
    n_folds = 10
    seed = 42
    kf = KFold(n_folds, shuffle=True, random_state=seed).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train,
                                   scoring="neg_mean_squared_error", cv = kf))

    return rmse


def reading_processing():
    path = '/home/jinma/dms_data'
    dms_data = 'dms_r4j012_fixed.dat.txt'
    file = os.path.join(path, dms_data)
    df = pd.read_table(file, low_memory=False, sep='\t', thousands=',')

    df.replace(-999, np.NaN, inplace=True)
    print df.columns
    df.drop(df[df['swDMS'].isnull()].index, inplace=True)
    df_sub = df.loc[df['DMSPt'].isnull() == False, :]
    print df_sub.info()

    # np.log1p = log(1+x)
    print df_sub['swDMS'].min()

    y = np.log10(df_sub['swDMS'] + 1.0)

    df_sub.drop(['swDMS', 'ContributionNumber', 'DateTime', 'Lat', 'Lon', 'flag'], axis=1, inplace=True)
    x = df_sub.copy()

    print x.mean()
    x.fillna(x.median(), inplace=True)

    print y.max(), type(y)

    return x, y

def testing_all_models(x, y):
    n_folds = 10

    seed = 43

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=10, random_state=seed))
    lasso_std = make_pipeline(StandardScaler(), Lasso(alpha=10, random_state=seed))


    ridge = make_pipeline(RobustScaler(), Ridge(alpha=10, random_state=seed))
    ridge_std = make_pipeline(StandardScaler(), Ridge(alpha=10, random_state=seed))

    linear = make_pipeline(RobustScaler(), LinearRegression(n_jobs=12))
    linear_std = make_pipeline(StandardScaler(), LinearRegression(n_jobs=12))


    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=seed))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, subsample=0.8,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=seed)

    random_forest = RandomForestRegressor(n_estimators=1000, max_depth=4, n_jobs=12)

    # our models to run the prediction
    models = [lasso, lasso_std, ridge, ridge_std, linear, linear_std, ENet, GBoost, random_forest]
    models_name = ['lasso', 'lasso_std', 'ridge', 'ridge_std', 'linear', 'linear_std', 'ENet', 'GBoost', 'random_forest']
    assert len(models) == len(models_name)

    for i, model in enumerate(models):
        score = rmsle_cv(model, x, y)
        print("{:s} score: {:.4f} ({:.4f})\n".format(models_name[i], score.mean(), score.std()))
        # y_pred = model.predict(x)
        # corr = np.corrcoef(y, y_pred)
        # print 'correlation of y', corr


    averaged_models = AveragingModels(models=(ENet, GBoost, lasso, linear, random_forest))
    score = rmsle_cv(averaged_models, x, y)
    print("Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    # y_pred = averaged_models.predict(x)
    # corr = np.corrcoef(y, y_pred)
    # print 'average model correlation of y', corr

def testing_linear_regression(x, y):
    n_folds = 10
    seed = 42
    linear = make_pipeline(RobustScaler(), LinearRegression(n_jobs=12))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_pred = np.zeros((len(y)))
    MSE = np.zeros(n_folds)
    corr = np.zeros(n_folds)
    for fold_no, [trn_id, val_id] in enumerate(folds.split(x, y)):
        x_trn, x_val = x.values[trn_id], x.values[val_id]
        y_trn, y_val = y.values[trn_id], y.values[val_id]
        linear.fit(x_trn, y_trn)
        y_pred[val_id] = linear.predict(x_val)
        MSE[fold_no] = mean_squared_error(y_true=y_val, y_pred=y_pred[val_id])
        corr[fold_no] = np.corrcoef(y_val, y_pred[val_id])[0, 1]
        print 'fold, MSE and corr:', fold_no, MSE[fold_no], corr[fold_no]

    print 'mean MSE and corr:', np.mean(MSE), np.mean(corr)


if __name__ == '__main__':
    x, y = reading_processing()
    # testing_all_models(x, y)
    print 'just test git'
    print 'test a commit'

    testing_linear_regression(x, y)

