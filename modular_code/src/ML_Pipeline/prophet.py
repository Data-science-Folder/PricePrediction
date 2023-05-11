from fbprophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def prophet(df_pr, y_col_name, date_col_name, pred_col_name, prediction_size=50):
    train_dataset = df_pr[:-prediction_size]
    test_dataset = df_pr[-prediction_size:]

    model = Prophet(weekly_seasonality=True, daily_seasonality=False)
    model.fit(train_dataset)
    future = model.make_future_dataframe(periods=prediction_size, freq='W')
    forecast = model.predict(future)

    r2 = r2_score(test_dataset[y_col_name], forecast[-prediction_size:]['yhat'])
    mae = mean_absolute_error(test_dataset[y_col_name], forecast[-prediction_size:]['yhat'])
    mse = mean_squared_error(test_dataset[y_col_name], forecast[-prediction_size:]['yhat'])
    return r2, mae, mse
