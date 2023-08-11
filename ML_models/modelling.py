import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from models import linear_regression, svm, random_forest, light_gbm


# Trains the different models and outputs csv file of the best parameters
class Model:
    def __init__(self):
        data = pd.read_csv('raw/features.csv')
        y = data['logS']
        le = LabelEncoder()
        kingdomlabels = le.fit_transform(data.kingdom)
        X = data.iloc[:, [*range(5, 213),*range(222,247)]]
        print(X.columns.to_list())
        X_scales = MinMaxScaler().fit_transform(X=X.values)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.column_stack((X_scales,kingdomlabels)), np.asarray(y),
                                                                                train_size=0.8,
                                                                                shuffle=True)

    def train(self):
        lin_reg = linear_regression(self.x_train, self.x_test, self.y_train, self.y_test)
        print('Linear Ridge Regression: Train score:', lin_reg[1], ' Test score:', lin_reg[2])
        print('Best parameter:', lin_reg[-2])
        lin_result = pd.DataFrame.from_dict(lin_reg[-1])
        lin_result.to_csv('Linear Regression_QSPR_full_classification')
        svc = svm(self.x_train, self.x_test, self.y_train, self.y_test)
        print('SVM Train score:', svc[0], ' Test score:', svc[1])
        print('Best parameter:', svc[-2])
        svc_result = pd.DataFrame.from_dict(svc[-1])
        svc_result.to_csv('SVC_QSPR_full_classification')
        rf = random_forest(self.x_train, self.x_test, self.y_train, self.y_test)
        print('Random Forest - Train score:', rf[0], ' Test score:', rf[1])
        print('Best parameter:', rf[-2])
        rf_result = pd.DataFrame.from_dict(rf[-1])
        rf_result.to_csv('Random_Forest_QSPR_full_classification')
        lgb = light_gbm(self.x_train, self.x_test, self.y_train, self.y_test)
        print('Light Gradient - Train score:', lgb[0], ' Test score:', lgb[1])
        print('Best parameter:', lgb[-2])
        lgb_result = pd.DataFrame.from_dict(lgb[-1])
        lgb_result.to_csv('Light-Gradient-Boosting_QSPR_butina_kingdom')


if __name__ == '__main__':
    Model().train()
