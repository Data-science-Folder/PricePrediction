from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def arima_model(data, best_order, train_size=0.66):
    X = data.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    for i in range(len(test)):
        model = ARIMA(history, order=best_order)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[i]
        history.append(obs)

    r2 = r2_score(test, predictions)
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    return r2, mae, mse