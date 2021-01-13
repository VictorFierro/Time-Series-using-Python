```python
#This file explains how to use SARIMAX to predict electricity prices on a monthly basis.

## First we import basic libraries
```


```python
#SARIMA Analysis for for monthly average prices of ERCOT Electricity Price
#Author: Victor Fierro
#Date: 25-Feb-2020

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
#First, we import montly prices of the ERCOT market, in South Houston, USA.
```


```python
df = pd.read_excel('Ercot_monthly.xlsx', sheet_name="Prices", skiprows=0)
#df.index = pd.to_datetime(df.index)
df = df.set_index('Delivery Date')
df.index = pd.to_datetime(df.index)
#df.set_index('Settlement Point Price').info()

#We print the data to validate we uploaded data correctly
df.count()

```




    Settlement Point Price    36
    dtype: int64




```python
#The following instructions help us to identify seasonality decomposing data in different periods.
```


```python
import matplotlib.pyplot as plt

decomposition = seasonal_decompose(df["Settlement Point Price"], period=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(16, 13)
```


    <Figure size 432x288 with 0 Axes>



![png](output_5_1.png)



```python
#Code:  https://medium.com/datadriveninvestor/time-series-prediction-using-sarimax-a6604f258c56


from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def test_adf(series, title=''):
    dfout={}
    dftest=sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')
    for key,val in dftest[4].items():
        dfout[f'critical value ({key})']=val
    if dftest[1]<=0.05:
        print("Strong evidence against Null Hypothesis")
        print("Reject Null Hypothesis - Data is Stationary")
        print("Data is Stationary", title)
    else:
        print("Strong evidence for  Null Hypothesis")
        print("Accept Null Hypothesis - Data is not Stationary")
        print("Data is NOT Stationary for", title)
```


```python
test_adf(df, "Ercot Prices")
```

    Strong evidence against Null Hypothesis
    Reject Null Hypothesis - Data is Stationary
    Data is Stationary Ercot Prices
    


```python
diff = df.diff()
test_adf(diff, "Diff Ercot Prices")
```

    Strong evidence against Null Hypothesis
    Reject Null Hypothesis - Data is Stationary
    Data is Stationary Diff Ercot Prices
    


```python
#The following instructions help us to identify seasonality in residuals
```


```python
#Code:  https://towardsdatascience.com/time-series-in-python-part-2-dealing-with-seasonal-data-397a65b74051

import statsmodels.api as sm
from statsmodels.api import OLS

x, y = np.arange(len(decomposition.trend.dropna())), decomposition.trend.dropna()
x = sm.add_constant(x)
model = OLS(y, x)
res = model.fit()
print(res.summary())
fig, ax = plt.subplots(1, 2, figsize=(12,6));
ax[0].plot(decomposition.trend.dropna().values, label='trend')
ax[0].plot([res.params.x1*i + res.params.const for i in np.arange(len(decomposition.trend.dropna()))])
ax[1].plot(res.resid.values);
ax[1].plot(np.abs(res.resid.values));
ax[1].hlines(0, 0, len(res.resid), color='r');
ax[0].set_title("Trend and Regression");
ax[1].set_title("Residuals");


#From the results we can see that the residuals increases. Even though, there is a trend, the R-Squared and Adj. R-squared are very low, we cannot use this regression.
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  trend   R-squared:                       0.694
    Model:                            OLS   Adj. R-squared:                  0.680
    Method:                 Least Squares   F-statistic:                     49.98
    Date:                Tue, 12 Jan 2021   Prob (F-statistic):           4.31e-07
    Time:                        21:01:44   Log-Likelihood:                -51.044
    No. Observations:                  24   AIC:                             106.1
    Df Residuals:                      22   BIC:                             108.4
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         26.5158      0.839     31.600      0.000      24.776      28.256
    x1             0.4420      0.063      7.070      0.000       0.312       0.572
    ==============================================================================
    Omnibus:                        5.440   Durbin-Watson:                   0.565
    Prob(Omnibus):                  0.066   Jarque-Bera (JB):                3.559
    Skew:                          -0.890   Prob(JB):                        0.169
    Kurtosis:                       3.628   Cond. No.                         26.1
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


![png](output_10_1.png)



```python
#The following instructions help us to identify stationarity in data and log data
```


```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


fig, ax = plt.subplots(2, sharex=True, figsize=(12,6))
ax[0].plot(df["Settlement Point Price"].values);
ax[0].set_title("Raw data");
ax[1].plot(np.log(df["Settlement Point Price"].values));
ax[1].set_title("Logged data (deflated)");
ax[1].set_ylim(0, 15);

fig, ax = plt.subplots(2, 2, figsize=(12,8))
first_diff = (np.log(df["Settlement Point Price"])- np.log(df["Settlement Point Price"]).shift()).dropna()
ax[0, 0] = plot_acf(np.log(df["Settlement Point Price"]), ax=ax[0, 0], lags=20, title="ACF - Logged data")
ax[1, 0] = plot_pacf(np.log(df["Settlement Point Price"]), ax=ax[1, 0], lags=20, title="PACF - Logged data")
ax[0, 1] = plot_acf(first_diff , ax=ax[0, 1], lags=20, title="ACF - Differenced Logged data")
ax[1, 1] = plot_pacf(first_diff, ax=ax[1, 1], lags=20, title="PACF - Differenced Logged data")
```


![png](output_12_0.png)



![png](output_12_1.png)



```python
## We test for stationarity using now the KPSS to confirm stationarity in log data.
# KPSS test for stationarity 
#code: https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788

from statsmodels.tsa.stattools import kpss

print(" > Is the first diff log data stationary ?")
dftest = kpss(first_diff, 'ct')
print("Test statistic = {:.3f}".format(dftest[0]))
print("P-value = {:.3f}".format(dftest[1]))
print("Critical values :")
for k, v in dftest[3].items():
    print("\t{}: {}".format(k, v))
#Comments: as the test_statistic is lower than the critical values, we accept the null hypothesis --> Our series is stationary
```

     > Is the first diff log data stationary ?
    Test statistic = 0.152
    P-value = 0.045
    Critical values :
    	10%: 0.119
    	5%: 0.146
    	2.5%: 0.176
    	1%: 0.216
    

    C:\Users\victor.fierro\AppData\Roaming\Python\Python37\site-packages\statsmodels\tsa\stattools.py:1661: FutureWarning: The behavior of using lags=None will change in the next release. Currently lags=None is the same as lags='legacy', and so a sample-size lag length is used. After the next release, the default will change to be the same as lags='auto' which uses an automatic lag length selection method. To silence this warning, either use 'auto' or 'legacy'
      warn(msg, FutureWarning)
    


```python
#Test if it is required to differentiate data or not
```


```python

from pmdarima.arima.stationarity import ADFTest

# Test whether we should difference at the alpha=0.05
# significance level
adf_test = ADFTest(alpha=0.05)
p_val, should_diff = adf_test.should_diff(df)  # (0.01, False)
print(p_val,should_diff)
#According with the ad_test, we should NOT differentiate
```

    0.04533988473625324 False
    


```python
# NOW we are ready to predict prices.
## We use pmdarima library and auto_arima function considering a max ACF and PACF of 4 periods.

## We first predict prices using part of the known prices as test and prediction periods.

```


```python
import  pmdarima  as pm
from pmdarima import auto_arima


stepwise_model1 = auto_arima(df, start_p=0, start_q=0,
                           max_p=12, max_q=12, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           start_Q=0,
                           max_P=12, max_Q=12, 
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model1.aic())
```

    Performing stepwise search to minimize aic
    Fit ARIMA: (0, 1, 0)x(0, 1, 0, 12) (constant=True); AIC=235.907, BIC=238.178, Time=0.030 seconds
    Fit ARIMA: (1, 1, 0)x(1, 1, 0, 12) (constant=True); AIC=233.840, BIC=238.381, Time=0.479 seconds
    Fit ARIMA: (0, 1, 1)x(0, 1, 1, 12) (constant=True); AIC=226.259, BIC=230.801, Time=0.799 seconds
    Near non-invertible roots for order (0, 1, 1)(0, 1, 1, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 1.000)
    Fit ARIMA: (0, 1, 0)x(0, 1, 0, 12) (constant=False); AIC=233.929, BIC=235.065, Time=0.035 seconds
    Fit ARIMA: (1, 1, 0)x(0, 1, 0, 12) (constant=True); AIC=234.259, BIC=237.666, Time=0.099 seconds
    Fit ARIMA: (1, 1, 0)x(2, 1, 0, 12) (constant=True); AIC=235.837, BIC=241.515, Time=2.939 seconds
    Fit ARIMA: (1, 1, 0)x(1, 1, 1, 12) (constant=True); AIC=235.840, BIC=241.518, Time=0.701 seconds
    Fit ARIMA: (1, 1, 0)x(0, 1, 1, 12) (constant=True); AIC=233.839, BIC=238.381, Time=0.486 seconds
    Fit ARIMA: (1, 1, 0)x(0, 1, 2, 12) (constant=True); AIC=235.839, BIC=241.516, Time=1.133 seconds
    Fit ARIMA: (1, 1, 0)x(1, 1, 2, 12) (constant=True); AIC=237.839, BIC=244.652, Time=3.082 seconds
    Fit ARIMA: (0, 1, 0)x(0, 1, 1, 12) (constant=True); AIC=234.093, BIC=237.500, Time=0.520 seconds
    Near non-invertible roots for order (0, 1, 0)(0, 1, 1, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 0.999)
    Fit ARIMA: (2, 1, 0)x(0, 1, 1, 12) (constant=True); AIC=233.577, BIC=239.255, Time=0.697 seconds
    Fit ARIMA: (2, 1, 0)x(0, 1, 0, 12) (constant=True); AIC=233.888, BIC=238.430, Time=0.256 seconds
    Fit ARIMA: (2, 1, 0)x(1, 1, 1, 12) (constant=True); AIC=235.578, BIC=242.391, Time=0.945 seconds
    Fit ARIMA: (2, 1, 0)x(0, 1, 2, 12) (constant=True); AIC=235.578, BIC=242.391, Time=2.583 seconds
    Near non-invertible roots for order (2, 1, 0)(0, 1, 2, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 0.990)
    Fit ARIMA: (2, 1, 0)x(1, 1, 0, 12) (constant=True); AIC=233.581, BIC=239.258, Time=0.669 seconds
    Fit ARIMA: (2, 1, 0)x(1, 1, 2, 12) (constant=True); AIC=237.565, BIC=245.514, Time=3.697 seconds
    Near non-invertible roots for order (2, 1, 0)(1, 1, 2, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 0.998)
    Fit ARIMA: (3, 1, 0)x(0, 1, 1, 12) (constant=True); AIC=234.301, BIC=241.114, Time=0.736 seconds
    Fit ARIMA: (2, 1, 1)x(0, 1, 1, 12) (constant=True); AIC=229.042, BIC=235.855, Time=1.334 seconds
    Near non-invertible roots for order (2, 1, 1)(0, 1, 1, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 1.000)
    Fit ARIMA: (1, 1, 1)x(0, 1, 1, 12) (constant=True); AIC=227.970, BIC=233.647, Time=0.833 seconds
    Near non-invertible roots for order (1, 1, 1)(0, 1, 1, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 1.000)
    Fit ARIMA: (3, 1, 1)x(0, 1, 1, 12) (constant=True); AIC=230.408, BIC=238.356, Time=0.916 seconds
    Near non-invertible roots for order (3, 1, 1)(0, 1, 1, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 0.997)
    Total fit time: 23.015 seconds
    226.25899012785817
    


```python
## We set the train period and the test period to finally conduct the prediction of prices.
#We can add as many periods as we prefer, but the number of predicted periods must be chosen considering the number of test periods.
```


```python
train, test = df[0:30], df[31:238]
stepwise_model1.fit(train)
future_forecast = stepwise_model1.predict(n_periods=5)

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=["Settlement Point Pric"])
#pd.concat([test,future_forecast],axis=1).iplot()


#future_forecast=pd.DataFrame(Arima_model.predict(n_periods=185), index=test.index)

future_forecast.columns = ['Predicted_Temperature']
plt.figure(figsize=(15,10))
plt.plot(train, label='Training')
plt.plot(test, label='Test')
plt.plot(future_forecast, label='Predicted')
plt.legend(loc = 'upper center')
plt.show()
```


![png](output_19_0.png)



```python
# We can now predict future prices.
##We add the number of periods into the dataset
```


```python
import datetime
from dateutil.relativedelta import relativedelta


start = datetime.datetime(2020 ,1, 1)
end = datetime.datetime(2021 , 8, 1)
future = pd.date_range(start, end, freq='M')


```


```python

future = pd.DataFrame(index=future, columns=df.columns)
print(future)

```

               Settlement Point Price
    2020-01-31                    NaN
    2020-02-29                    NaN
    2020-03-31                    NaN
    2020-04-30                    NaN
    2020-05-31                    NaN
    2020-06-30                    NaN
    2020-07-31                    NaN
    2020-08-31                    NaN
    2020-09-30                    NaN
    2020-10-31                    NaN
    2020-11-30                    NaN
    2020-12-31                    NaN
    2021-01-31                    NaN
    2021-02-28                    NaN
    2021-03-31                    NaN
    2021-04-30                    NaN
    2021-05-31                    NaN
    2021-06-30                    NaN
    2021-07-31                    NaN
    


```python
dfFuture = pd.concat([df, future])
print(dfFuture)
```

                Settlement Point Price
    2017-01-01               23.624812
    2017-02-01               21.079360
    2017-03-01               23.679735
    2017-04-01               25.609403
    2017-05-01               26.495013
    2017-06-01               29.654694
    2017-07-01               32.129462
    2017-08-01               28.315027
    2017-09-01               25.515097
    2017-10-01               25.563656
    2017-11-01               23.963499
    2017-12-01               21.997917
    2018-01-01               39.498145
    2018-02-01               22.306399
    2018-03-01               22.885993
    2018-04-01               24.471889
    2018-05-01               34.226075
    2018-06-01               30.068639
    2018-07-01               78.847379
    2018-08-01               31.667527
    2018-09-01               27.189806
    2018-10-01               29.772513
    2018-11-01               29.830704
    2018-12-01               30.089301
    2019-01-01               25.086062
    2019-02-01               22.125640
    2019-03-01               27.134315
    2019-04-01               22.463333
    2019-05-01               24.059946
    2019-06-01               23.904361
    2019-07-01               29.637500
    2019-08-01              125.936546
    2019-09-01               78.900472
    2019-10-01               29.468401
    2019-11-01               23.715527
    2019-12-01               18.952513
    2020-01-31                     NaN
    2020-02-29                     NaN
    2020-03-31                     NaN
    2020-04-30                     NaN
    2020-05-31                     NaN
    2020-06-30                     NaN
    2020-07-31                     NaN
    2020-08-31                     NaN
    2020-09-30                     NaN
    2020-10-31                     NaN
    2020-11-30                     NaN
    2020-12-31                     NaN
    2021-01-31                     NaN
    2021-02-28                     NaN
    2021-03-31                     NaN
    2021-04-30                     NaN
    2021-05-31                     NaN
    2021-06-30                     NaN
    2021-07-31                     NaN
    


```python
#We conduct the prection
```


```python
train, test = dfFuture[0:35], dfFuture[36:57]
stepwise_modelForecast.fit(train)
future_forecast = stepwise_modelForecast.predict(n_periods=19)

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=["Settlement Point Price"])
#pd.concat([test,future_forecast],axis=1).iplot()

print(future_forecast)
#future_forecast=pd.DataFrame(Arima_model.predict(n_periods=185), index=test.index)
#future_forecast.columns = ['Predicted_Temperature']
plt.figure(figsize=(13,5))
plt.plot(train, label='Training')
plt.plot(test, label='Test')
plt.plot(future_forecast, label='Predicted')
plt.legend(loc = 'upper center')
plt.title('Forecasting $USD per MWh')
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-33-fbe662765dd7> in <module>
          1 train, test = dfFuture[0:35], dfFuture[36:57]
    ----> 2 stepwise_modelForecast.fit(train)
          3 future_forecast = stepwise_modelForecast.predict(n_periods=19)
          4 
          5 future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=["Settlement Point Price"])
    

    NameError: name 'stepwise_modelForecast' is not defined



```python
#We can visualize in detail only predicted prices.

dfFutureVal = pd.concat([df, future_forecast])
print(dfFutureVal)
#future_forecast=pd.DataFrame(Arima_model.predict(n_periods=185), index=test.index)
#future_forecast.columns = ['Predicted_Temperature']
plt.figure(figsize=(15,10))
#plt.plot(train, label='Training')
plt.plot(dfFutureVal, label='Total')
#plt.plot(future_forecast, label='Predicted')
plt.legend(loc = 'upper center')
plt.show()
```

                Settlement Point Price
    2017-01-01               23.624812
    2017-02-01               21.079360
    2017-03-01               23.679735
    2017-04-01               25.609403
    2017-05-01               26.495013
    2017-06-01               29.654694
    2017-07-01               32.129462
    2017-08-01               28.315027
    2017-09-01               25.515097
    2017-10-01               25.563656
    2017-11-01               23.963499
    2017-12-01               21.997917
    2018-01-01               39.498145
    2018-02-01               22.306399
    2018-03-01               22.885993
    2018-04-01               24.471889
    2018-05-01               34.226075
    2018-06-01               30.068639
    2018-07-01               78.847379
    2018-08-01               31.667527
    2018-09-01               27.189806
    2018-10-01               29.772513
    2018-11-01               29.830704
    2018-12-01               30.089301
    2019-01-01               25.086062
    2019-02-01               22.125640
    2019-03-01               27.134315
    2019-04-01               22.463333
    2019-05-01               24.059946
    2019-06-01               23.904361
    2019-07-01               29.637500
    2019-08-01              125.936546
    2019-09-01               78.900472
    2019-10-01               29.468401
    2019-11-01               23.715527
    2019-12-01               18.952513
    2020-01-31               41.803812
    2020-02-29               43.910031
    2020-03-31               38.168882
    2020-04-30               42.472915
    2020-05-31               40.807412
    2020-06-30               45.003006
    2020-07-31               44.933270
    2020-08-31               61.553967
    2020-09-30              111.211193
    2020-10-31               80.624139
    2020-11-30               50.597484
    2020-12-31               47.447376
    2021-01-31               59.419354
    2021-02-28               61.979743
    2021-03-31               56.692764
    2021-04-30               61.450966
    2021-05-31               60.239634
    2021-06-30               64.889398
    2021-07-31               65.273832
    


![png](output_26_1.png)



```python
future_forecast['Settlement Point Price']
```




    2020-01-31     41.803812
    2020-02-29     43.910031
    2020-03-31     38.168882
    2020-04-30     42.472915
    2020-05-31     40.807412
    2020-06-30     45.003006
    2020-07-31     44.933270
    2020-08-31     61.553967
    2020-09-30    111.211193
    2020-10-31     80.624139
    2020-11-30     50.597484
    2020-12-31     47.447376
    2021-01-31     59.419354
    2021-02-28     61.979743
    2021-03-31     56.692764
    2021-04-30     61.450966
    2021-05-31     60.239634
    2021-06-30     64.889398
    2021-07-31     65.273832
    Name: Settlement Point Price, dtype: float64




```python
plt.figure(figsize=(15,10))
plt.plot(future_forecast, label='Predicted')
plt.legend(loc = 'upper center')
plt.show()
```


![png](output_28_0.png)

