# PricePrediction

Description
--------------------
The goal is to predict Average price of avocado.

Data Loading and Description
--------------------

* The dataset consists of the information about HASS Avocado. Historical data on avocado prices and sales volume in multiple US markets. Various variables present in the dataset includes Date, AveragePrice,Total Volume, Total Bags,Year,Type etc.
* The dataset comprises of 18249 observations of 14 columns. Below is a table showing names of all the columns and their description.
* Data location:  https://www.kaggle.com/neuromusic/avocado-prices#avocado.csv


Columns 

| Feature      | Description                               |
|--------------|-------------------------------------------|
| Date         | date of the observation                   |
| AveragePrice | average price of a single avocado         |
| Type         | conventional / organic                    |
| Region       | region of the observation                                    |
| Total Volume | Total number of avocados sold                                   |
| 4046         | Total number of avocados with PLU 4046 sold                                   |
| 4225         | Total number of avocados with PLU 4225 sold                                   |
| 4770         | Total number of avocados with PLU 4770 sold                                    |
| Total Bags   | Total bags sol                                     |
| Size         | Total bags sold by size                                      |

Tech Stack
--------------------

*  Language used : Python
*  Libraries used : statmodels, pmdarima, fbprophet, scikit-learn

Approach
--------------------
| Steps                       | Sub parts                                                                                                      |
|-----------------------------|----------------------------------------------------------------------------------------------------------------|
| Data Preprocessing          | Check for missing values, Label Encoding, One hot encoding                                                     |
| Exploratory Data Analysis   | Identifying any overarching trend in data over time, Identifying any repetitive, seasonal patterns in the data |
| Feature Engineerin          | Creating new column                                                                                            |
| Building Forecast models    | Linear Regression, Random Forest Regressor, XGB Regressor, Facebook Prophet, ARIMA, SARIMAX                    |
| Evaluating Forecast models  | R-squared, MAPE, MAE, plots comprising the actual values, forecast and confidence intervals.                                                                                        |

Usage
--------------------
Start at: ../modular_code/src/engine.py
Avocado.ipynb


# Project tree

 * [PredictPrice](./tree-md)
 * [data](./dir2)
   * [avocado.csv](./dir2/file21.ext)

 * [modular_code](./dir2)
   * [input](./dir2/file11.ext)
     * [avocado.csv](./dir2/dir3/file11.ext)
   * [lib](./dir1/file12.ext)
     * [Avocado.ipynb](./dir1/dir2/file11.ext)
   * [output](./dir1/file12.ext)
     * [lin_reg.pkl](./dir1/dir2/file11.ext)
     * [rf_reg.pkl](./dir1/dir2/file11.ext)
     * [xbg_reg.pkl](./dir1/dir2/file11.ext)
   * [src](./dir1/file11.ext)
     * [engine.py](./dir1/dir2/file11.ext)
     * [ML_pipleine](./dir1/dir2/file11.ext)
       * [arima.py](./dir1/dir2/dir3/file11.ext)
       * [dataset.py](./dir1/dir2/dir3/file11.ext)
       * [encoding.py](./dir1/dir2/dir3/file11.ext)
       * [train_test_split.py](./dir1/dir2/dir3/file11.ext)
       * [regression_models.py](./dir1/dir2/dir3/file11.ext)
 * [README.md](./README.md)


