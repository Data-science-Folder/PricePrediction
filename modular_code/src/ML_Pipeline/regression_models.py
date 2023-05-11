from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib 

model_dict = {
    'lin_reg': LinearRegression,
    'rf_reg':RandomForestRegressor,
    'xgb_reg':XGBRegressor
}

def regression(X_train, X_test, y_train, y_test, model_name, model_path):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param model_name:
    :param model_path:
    :return: tuple
        Tuple with R2 score, MAPE, RMSE
    """
    if model_name not in model_dict :
        raise ValueError(
            f'Only these options for model_name are allowed : {list(model_dict.keys())}')
    pipe = Pipeline([('scaler', StandardScaler()), (model_name, model_dict[model_name]())])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe[model_name], model_path)
    print(f'model saved in {model_path}')
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mae, mse
