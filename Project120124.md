## Polish stock market indicies analysis with Deep Learning models 


```python
# I begin with some standard imports
import math
import os
import sys
import time
import numpy as np
import pandas as pd
import quantstats as qs
import random as rn
import matplotlib.pyplot as plt
import scipy.stats as scs
%matplotlib inline
%config InLineBackend.figure_format = 'svg'
```


```python
import warnings
warnings.simplefilter('ignore')
```


```python
# Sci-kit learn imports (for shallow learning)
import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
```


```python
# Tensorflow and Keras imports (for deep learning)
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.initializers import random_normal
from keras.layers import Dense, Dropout, SimpleRNN, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
```


```python
# Some Matplotlib and Seaborn settings to standardize the figures in the notebook 
plt.rcParams["figure.dpi"] = 75
plt.style.use("seaborn-v0_8")
```


```python
# I define a seed function to set all seeds (starting values for Random Number Generator) for future reproducibility
def set_seeds(seed=100):
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)
set_seeds()
```


```python
# Setup a function and a destination folder to save the models
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deep"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figures", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# Before moving on I check the current versions of tech stack/libraries installed/utilized
print("Python version: ", sys.version[:5])
print("Scikit-learn version: ", sklearn.__version__)
print("Keras version: ", keras.__version__)
print("Tensorflow version: ", tf.__version__)
```

    Python version:  3.8.3
    Scikit-learn version:  1.3.2
    Keras version:  2.10.0
    Tensorflow version:  2.10.0


## Index level prediction

## EDA


```python
# Import raw data from a CSV file obtained from InfoStrefa (Polish equity market data provider) database
# Set first column as an index column
# Convert the 'date' objects to a 'DatetimeIndex' objects
raw = pd.read_csv('WIG.csv', index_col=0, parse_dates=True)
```


```python
# First, inspect the metadata
raw.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 7732 entries, 1991-04-16 to 2023-12-29
    Data columns (total 4 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   open    7732 non-null   float64
     1   high    7732 non-null   float64
     2   low     7732 non-null   float64
     3   close   7732 non-null   float64
    dtypes: float64(4)
    memory usage: 302.0 KB

The dataset consists of 7 732 non-null datapoints for Polish WIG index (each point with OHLC index levels)

```python
# Second, inspect the data
raw.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-12-21</th>
      <td>78341.36</td>
      <td>78840.93</td>
      <td>78237.66</td>
      <td>78723.40</td>
    </tr>
    <tr>
      <th>2023-12-22</th>
      <td>78560.60</td>
      <td>78814.86</td>
      <td>78409.07</td>
      <td>78667.79</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>78916.60</td>
      <td>79460.94</td>
      <td>78784.47</td>
      <td>79246.98</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>79451.43</td>
      <td>79548.69</td>
      <td>78935.53</td>
      <td>78937.48</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>78961.91</td>
      <td>79159.47</td>
      <td>78202.60</td>
      <td>78459.91</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Third, visualize the 'close' column
raw['close'].plot(title='WIG index');
```


![png](output_14_0.png)



```python
# Let's utilise quantstats library capabilities for some further EDA
qs.plots.snapshot(raw['close'], title='WIG index performance', fontname='sans-serif')
qs.plots.drawdowns_periods(raw['close'], title='WIG', fontname='sans-serif')
qs.plots.monthly_heatmap(raw['close'], fontname='sans-serif')
```


![png](output_15_0.png)



![png](output_15_1.png)



![png](output_15_2.png)


## Simple linear regression


```python
# Convert the raw data into a DataFrame object, drop all columns except for the 'close' column (closing price)
# Make a copy of the DataFrame
df = pd.DataFrame(raw).drop(['open', 'high', 'low'], axis=1)
df1 = df.copy()
```


```python
# Add some lagged values of the 'close' column to the DataFrame
# Drop any 'na' values created during the process 
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    df1[col] = df1['close'].shift(lag)
    cols.append(col)
df1.dropna(inplace=True)
```


```python
# Perform a linear regression (target = close, predictors = lagged close)
# Print the fitted regression parameters
lin_reg = np.linalg.lstsq(df1[cols], df1['close'], rcond=None)[0]
lin_reg.round(2)
```




    array([ 1.06, -0.08,  0.06, -0.02, -0.01])




```python
# Add the model's predictions to the DataFrame
df1['prediction'] = np.dot(df1[cols], lin_reg)
```


```python
# Plot the index level vs the model predicted level
df1[['close', 'prediction']].loc['2023-06-01':].plot();
```


![png](output_21_0.png)


## Index return prediction
Let's try to predict the index return instead of the index level

```python
# Make a copy of the raw data
# Calculate the daily log returns
# Drop the na value created in the process
df2 = df.copy()
df2['index_return'] = np.log(df2['close'] / df2['close'].shift(1))
df2.dropna(inplace=True)
```


```python
# Plot a histogram of daily log returns against a normal pdf curve
# The realized returns obviously do not follow a normal distribution
plt.hist(df2['index_return'], bins=300, density=True, label='log returns')
plt.xlabel('Log returns')
plt.ylabel('Probability density')
plt.title('WIG log returns distribution')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, df2['index_return'].mean(), df2['index_return'].std()), 'r', lw=2.0, label='pdf')
plt.legend();
```


![png](output_25_0.png)



```python
# Add 5 lagged values of the 'index_return' column to the DataFrame
# Drop any 'na' values created during the process 
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    df2[col] = df2['index_return'].shift(lag)
    cols.append(col)
df2.dropna(inplace=True)
```


```python
# Perform a linear regression (target = log daily return, predictors = lagged log daily returns)
# Print the fitted regression parameters
lin_reg = np.linalg.lstsq(df2[cols], df2['index_return'], rcond=None)[0]
lin_reg.round(2)
```




    array([ 0.23, -0.06,  0.03,  0.03,  0.01])




```python
# Add the model's predictions to the DataFrame
df2['prediction'] = np.dot(df2[cols], lin_reg)
```


```python
# Plot the index realized return vs the model predicted return
df2[['index_return', 'prediction']].plot();
```


![png](output_29_0.png)



```python
# Calculate a 'hits' ratio (% of correct index direction predictions)
hits = np.sign(df2['index_return'] * df2['prediction']).value_counts()
hits.values[0] / sum(hits)
```




    0.5267926482008801



## Index direction prediction (linear regression)
Let's try to predict the index return instead of the index level

```python
# Make a copy of the raw data
# Calculate the daily log returns
# Drop the na value created in the process
df3 = df.copy()
df3['index_return'] = np.log(df3['close'] / df3['close'].shift(1))
df3.dropna(inplace=True)
```


```python
# Add 5 lagged values of the 'index_return' column to the DataFrame
# Drop any 'na' values created during the process 
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    df3[col] = df3['index_return'].shift(lag)
    cols.append(col)
df3.dropna(inplace=True)
```


```python
# Build a linear regression model (target = sign of log daily return, predictors = lagged log daily returns)
# Print the fitted regression parameters
lin_reg = np.linalg.lstsq(df3[cols], np.sign(df3['index_return']), rcond=None)[0]
lin_reg.round(2)
```




    array([ 5.36, -2.08,  0.22,  1.06,  0.29])




```python
# Add the model's predictions to the DataFrame
df3['prediction'] = np.sign(np.dot(df3[cols], lin_reg))
df3['prediction'].value_counts()
```




    prediction
     1.0    3957
    -1.0    3769
    Name: count, dtype: int64




```python
# Calculate a 'hits' ratio (% of correct index direction predictions)
hits = np.sign(df3['index_return'] * df3['prediction']).value_counts()
hits.values[0] / sum(hits)
```




    0.529510742945897



## Vectorized backtest
Let's perform a vectorized backtest 

```python
# Calculate the 'strategy return' - return realized if investing in the index based on the model's predictions (no transaction costs) 
df3['strategy_return'] = df3['prediction'] * df3['index_return']
df3[['index_return', 'strategy_return']].sum().apply(np.exp)
```




    index_return       8.202813e+01
    strategy_return    2.844236e+06
    dtype: float64




```python
# Plot cumulative returns of the strategy vs the index return
df3[['index_return', 'strategy_return']].dropna().cumsum().apply(np.exp).plot();
```


![png](output_41_0.png)


## Index direction prediction (logistic regression)


```python
# Make a copy of the raw data
# Calculate the daily log returns
# Drop the na value created in the process
df4 = df.copy()
df4['index_return'] = np.log(df4['close'] / df4['close'].shift(1))
df4.dropna(inplace=True)
```


```python
# Add 5 lagged values of the 'index_return' column to the DataFrame
# Drop any 'na' values created during the process 
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    df4[col] = df4['index_return'].shift(lag)
    cols.append(col)
df4.dropna(inplace=True)
```


```python
# Instantiate a logistic regression model
log_reg = linear_model.LogisticRegression(C=1e7, solver='lbfgs', multi_class='auto', max_iter=1000)
```


```python
# Perform a logistic regression (target = log daily return, predictors = lagged log daily returns)
log_reg.fit(df4[cols], np.sign(df4['index_return']))
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=10000000.0, max_iter=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(C=10000000.0, max_iter=1000)</pre></div></div></div></div></div>




```python
# Add the model's predictions to the DataFrame
df4['prediction'] = log_reg.predict(df4[cols])
df4['prediction'].value_counts()
```




    prediction
     1.0    5770
    -1.0    1956
    Name: count, dtype: int64




```python
# Calculate a 'hits' ratio (% of correct index direction predictions)
hits = np.sign(df4['index_return'] * df4['prediction']).value_counts()
hits.values[0] / sum(hits)
```




    0.5358529640176029




```python
# Calculate accuracy_score metric
accuracy_score(df4['prediction'], np.sign(df4['index_return']))
```




    0.5358529640176029




```python
# Calculate the 'strategy return' - return realized if investing in the index based on the model's predictions (no transaction costs) 
df4['strategy_return'] = df4['prediction'] * df4['index_return']
df4[['index_return', 'strategy_return']].sum().apply(np.exp)
```




    index_return       8.202813e+01
    strategy_return    4.189164e+06
    dtype: float64




```python
# Plot a cumulative returns of the strategy vs the index
df4[['index_return', 'strategy_return']].dropna().cumsum().apply(np.exp).plot();
```


![png](output_51_0.png)


## Index direction prediction (deep neural network)


```python
# Make a copy of the raw data
# Calculate the daily log returns
# Drop the na value created in the process
df5 = df.copy()
df5['index_return'] = np.log(df5['close'] / df5['close'].shift(1))
df5.dropna(inplace=True)
```


```python
# Convert daily log returns into 0 (negative return) or 1 (positive return) 
df5['direction'] = np.where(df5['index_return'] > 0, 1, 0)
```


```python
# Add 5 lagged values of the 'index_return' column to the DataFrame
# Drop any 'na' values created during the process
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    df5[col] = df5['index_return'].shift(lag)
    cols.append(col)
df5.dropna(inplace=True)
```


```python
df5.shape
```




    (7726, 8)




```python
# Perform a train/val/test split
# Assume 80% of the dataset used for training, 20% of the dataset used for testing, 10% of the training dataset used for validation
x_train, x_test, y_train, y_test = train_test_split(df5, df5, test_size=0.2, random_state=100, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=100, shuffle=False)

print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)
print("Shape of x_val: ", x_val.shape)
print("Shape of y_val: ", y_val.shape)
```

    Shape of x_train:  (5562, 8)
    Shape of y_train:  (5562, 8)
    Shape of x_test:  (1546, 8)
    Shape of y_test:  (1546, 8)
    Shape of x_val:  (618, 8)
    Shape of y_val:  (618, 8)



```python
print(len(x_train) + len(x_val) + len(x_test))
print(len(y_train) + len(y_val) + len(y_test))
```

    7726
    7726


## A simple DNN


```python
# Let's start with 1 hidden layer, 32 neurons, ReLU activation
set_seeds()
model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(len(cols),)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=["accuracy"])
model.fit(x_train[cols], x_train['direction'], epochs = 25, verbose=False, shuffle=False)
```




    <keras.callbacks.History at 0x7f2a8fb50a60>




```python
res = pd.DataFrame(model.history.history)
res[['loss', 'accuracy']].plot(style='--');
```


![png](output_61_0.png)



```python
model.evaluate(x_train[cols], x_train['direction'])
```

    174/174 [==============================] - 1s 2ms/step - loss: 0.6851 - accuracy: 0.5502





    [0.6851018667221069, 0.5501618385314941]




```python
pred = np.where(model.predict(x_train[cols]) > 0.5, 1, 0)
```

    174/174 [==============================] - 0s 2ms/step



```python
x_train['prediction'] = np.where(pred > 0, 1, -1)
x_train['strategy_return'] = (x_train['prediction'] * x_train['index_return'])
x_train[['index_return', 'strategy_return']].cumsum().apply(np.exp).plot();
```


![png](output_64_0.png)



```python
model.evaluate(x_test[cols], x_test['direction'])
```

    49/49 [==============================] - 0s 3ms/step - loss: 0.6953 - accuracy: 0.5071





    [0.6953163146972656, 0.5071151256561279]




```python
test_scores = model.evaluate(x_test[cols], x_test['direction'], verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

    49/49 - 0s - loss: 0.6953 - accuracy: 0.5071 - 114ms/epoch - 2ms/step
    Test loss: 0.6953163146972656
    Test accuracy: 0.5071151256561279



```python
pred = np.where(model.predict(x_test[cols]) > 0.5, 1, 0)
```

    49/49 [==============================] - 0s 2ms/step



```python
x_test['prediction'] = np.where(pred > 0, 1, -1)
x_test['prediction'].value_counts()
```




    prediction
     1    1194
    -1     352
    Name: count, dtype: int64




```python
x_test['strategy_return'] = (x_test['prediction'] * x_test['index_return'])
x_test[['index_return', 'strategy_return']].sum().apply(np.exp)
```




    index_return       1.231425
    strategy_return    1.186141
    dtype: float64




```python
x_test[['index_return', 'strategy_return']].cumsum().apply(np.exp).plot();
```


![png](output_70_0.png)


## A simple DNN - additional features


```python
df5['mom'] = df5['index_return'].rolling(5).mean().shift(1)
df5['vol'] = df5['index_return'].rolling(20).std().shift(1)
df5['dis'] = (df5['close'] - df5['close'].rolling(50).mean()).shift(1)
df5.dropna(inplace=True)
cols.extend(['mom', 'vol', 'dis'])
```


```python
# Perform a train/val/test split
# Assume 80% of the dataset used for training, 20% of the dataset used for testing, 10% of the training dataset used for validation
x_train, x_test, y_train, y_test = train_test_split(df5, df5, test_size=0.2, random_state=100, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=100, shuffle=False)
```


```python
# 2 hidden layers, 32 neurons each, ReLU activation
set_seeds()
model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(len(cols),)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=["accuracy"])
model.fit(x_train[cols], x_train['direction'], epochs = 50, verbose=False, shuffle=False)
```




    <keras.callbacks.History at 0x7f2a8f6a83a0>




```python
res = pd.DataFrame(model.history.history)
res[['loss', 'accuracy']].plot(style='--');
```


![png](output_75_0.png)



```python
model.evaluate(x_train[cols], x_train['direction'])
```

    173/173 [==============================] - 0s 2ms/step - loss: 1.7044 - accuracy: 0.4900





    [1.7044377326965332, 0.49004703760147095]




```python
pred = np.where(model.predict(x_train[cols]) > 0.5, 1, 0)
```

    173/173 [==============================] - 0s 2ms/step



```python
x_train['prediction'] = np.where(pred > 0, 1, -1)
x_train['strategy_return'] = (x_train['prediction'] * x_train['index_return'])
x_train[['index_return', 'strategy_return']].cumsum().apply(np.exp).plot();
```


![png](output_78_0.png)



```python
model.evaluate(x_test[cols], x_test['direction'])
```

    48/48 [==============================] - 0s 2ms/step - loss: 3.0308 - accuracy: 0.4831





    [3.030824661254883, 0.4830729067325592]




```python
test_scores = model.evaluate(x_test[cols], x_test['direction'], verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

    48/48 - 0s - loss: 3.0308 - accuracy: 0.4831 - 101ms/epoch - 2ms/step
    Test loss: 3.030824661254883
    Test accuracy: 0.4830729067325592



```python
pred = np.where(model.predict(x_test[cols]) > 0.5, 1, 0)
```

    48/48 [==============================] - 0s 1ms/step



```python
x_test['prediction'] = np.where(pred > 0, 1, -1)
x_test['prediction'].value_counts()
```




    prediction
    -1    781
     1    755
    Name: count, dtype: int64




```python
x_test['strategy_return'] = (x_test['prediction'] * x_test['index_return'])
x_test[['index_return', 'strategy_return']].sum().apply(np.exp)
```




    index_return       1.204587
    strategy_return    0.487658
    dtype: float64




```python
x_test[['index_return', 'strategy_return']].cumsum().apply(np.exp).plot();
```


![png](output_84_0.png)

