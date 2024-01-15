## Polish stock market indices analysis


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
# To suppress some warnings
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
# To setup a function and a destination folder to save the models
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


## Exploratory Data Analysis (EDA)


```python
# Import raw data from a CSV file obtained from InfoStrefa database for 4 main Polish equity market indices
# Set first columns as index columns
# Convert the date (string) columns to a DatetimeIndex columns
raw = pd.read_csv('WIG.csv', index_col=0, parse_dates=True)

wig = pd.read_csv('WIG.csv', index_col=0, parse_dates=True)
wig20 = pd.read_csv('WIG20.csv', index_col=0, parse_dates=True)
mwig40 = pd.read_csv('mWIG40.csv', index_col=0, parse_dates=True)
swig80 = pd.read_csv('sWIG80.csv', index_col=0, parse_dates=True)
```


```python
# Let's inspect:
# 1. Metadata (info() method)
# 2. Data (tail() method)
# 3. Baisc statistics (describe() method, below for wig20 index only)
```


```python
wig20.info()
wig20.tail()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 7399 entries, 1994-04-18 to 2023-12-29
    Data columns (total 4 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   open_wig20   7399 non-null   float64
     1   high_wig20   7399 non-null   float64
     2   low_wig20    7399 non-null   float64
     3   close_wig20  7399 non-null   float64
    dtypes: float64(4)
    memory usage: 289.0 KB





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
      <th>open_wig20</th>
      <th>high_wig20</th>
      <th>low_wig20</th>
      <th>close_wig20</th>
    </tr>
    <tr>
      <th>Data</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-12-21</th>
      <td>2333.32</td>
      <td>2355.14</td>
      <td>2329.78</td>
      <td>2348.67</td>
    </tr>
    <tr>
      <th>2023-12-22</th>
      <td>2345.16</td>
      <td>2353.47</td>
      <td>2338.03</td>
      <td>2346.70</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>2348.85</td>
      <td>2378.32</td>
      <td>2348.85</td>
      <td>2371.28</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>2374.38</td>
      <td>2380.55</td>
      <td>2356.32</td>
      <td>2356.32</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>2356.84</td>
      <td>2367.01</td>
      <td>2334.30</td>
      <td>2342.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
wig20.describe().round(2)
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
      <th>open_wig20</th>
      <th>high_wig20</th>
      <th>low_wig20</th>
      <th>close_wig20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7399.00</td>
      <td>7399.00</td>
      <td>7399.00</td>
      <td>7399.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1994.83</td>
      <td>2010.19</td>
      <td>1978.30</td>
      <td>1994.45</td>
    </tr>
    <tr>
      <th>std</th>
      <td>617.82</td>
      <td>622.53</td>
      <td>611.55</td>
      <td>617.21</td>
    </tr>
    <tr>
      <th>min</th>
      <td>577.90</td>
      <td>577.90</td>
      <td>577.90</td>
      <td>577.90</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1583.18</td>
      <td>1598.26</td>
      <td>1568.66</td>
      <td>1582.80</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2003.90</td>
      <td>2020.04</td>
      <td>1983.44</td>
      <td>2000.90</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2371.74</td>
      <td>2387.70</td>
      <td>2354.78</td>
      <td>2370.56</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3935.47</td>
      <td>3940.53</td>
      <td>3910.96</td>
      <td>3917.87</td>
    </tr>
  </tbody>
</table>
</div>




```python
mwig40.info()
mwig40.tail()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 6509 entries, 1997-12-31 to 2023-12-29
    Data columns (total 4 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   open_mwig40   6509 non-null   float64
     1   high_mwig40   6509 non-null   float64
     2   low_mwig40    6509 non-null   float64
     3   close_mwig40  6509 non-null   float64
    dtypes: float64(4)
    memory usage: 254.3 KB





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
      <th>open_mwig40</th>
      <th>high_mwig40</th>
      <th>low_mwig40</th>
      <th>close_mwig40</th>
    </tr>
    <tr>
      <th>Data</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-12-21</th>
      <td>5876.86</td>
      <td>5880.79</td>
      <td>5833.93</td>
      <td>5880.79</td>
    </tr>
    <tr>
      <th>2023-12-22</th>
      <td>5874.07</td>
      <td>5876.90</td>
      <td>5844.89</td>
      <td>5864.68</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>5872.31</td>
      <td>5883.49</td>
      <td>5842.38</td>
      <td>5847.11</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>5849.55</td>
      <td>5875.54</td>
      <td>5836.06</td>
      <td>5845.89</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>5848.65</td>
      <td>5853.96</td>
      <td>5760.91</td>
      <td>5785.21</td>
    </tr>
  </tbody>
</table>
</div>




```python
swig80.info()
swig80.tail()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 7257 entries, 1994-12-29 to 2023-12-29
    Data columns (total 4 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   open_swig80   7257 non-null   float64
     1   high_swig80   7257 non-null   float64
     2   low_swig80    7257 non-null   float64
     3   close_swig80  7257 non-null   float64
    dtypes: float64(4)
    memory usage: 283.5 KB





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
      <th>open_swig80</th>
      <th>high_swig80</th>
      <th>low_swig80</th>
      <th>close_swig80</th>
    </tr>
    <tr>
      <th>Data</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-12-21</th>
      <td>22627.12</td>
      <td>22644.29</td>
      <td>22520.92</td>
      <td>22596.17</td>
    </tr>
    <tr>
      <th>2023-12-22</th>
      <td>22592.39</td>
      <td>22705.37</td>
      <td>22586.68</td>
      <td>22674.27</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>22758.07</td>
      <td>22858.09</td>
      <td>22758.07</td>
      <td>22802.65</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>22867.93</td>
      <td>22950.33</td>
      <td>22867.67</td>
      <td>22950.33</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>22975.52</td>
      <td>23059.90</td>
      <td>22869.68</td>
      <td>22904.49</td>
    </tr>
  </tbody>
</table>
</div>




```python
wig.info()
wig.tail()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 7732 entries, 1991-04-16 to 2023-12-29
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   open_wig   7732 non-null   float64
     1   high_wig   7732 non-null   float64
     2   low_wig    7732 non-null   float64
     3   close_wig  7732 non-null   float64
    dtypes: float64(4)
    memory usage: 302.0 KB





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
      <th>open_wig</th>
      <th>high_wig</th>
      <th>low_wig</th>
      <th>close_wig</th>
    </tr>
    <tr>
      <th>Data</th>
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
# The data looks good 
# 1. No null values (datasets are complete)
# 2. Date columns are of the DatetimeIndex type (reliable future timeseries data manipulation)
# 3. The numerical data is of float64 type (good because....)
```


```python
# Let's merge the 4 obtained datasets into one dataset
# First, inspect the datasets shapes
wig.shape, wig20.shape, mwig40.shape, swig80.shape
```




    ((7732, 4), (7399, 4), (6509, 4), (7257, 4))




```python
# Second, combine 4 datasets with outer joins (to have all 4 datasets in a single dataset, no data lost in the process)
data_merged = pd.merge(wig, wig20, how="outer", left_index=True, right_index=True)
data_merged = pd.merge(data_merged, mwig40, how="outer", left_index=True, right_index=True)
data_merged = pd.merge(data_merged, swig80, how="outer", left_index=True, right_index=True)
data_merged
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
      <th>open_wig</th>
      <th>high_wig</th>
      <th>low_wig</th>
      <th>close_wig</th>
      <th>open_wig20</th>
      <th>high_wig20</th>
      <th>low_wig20</th>
      <th>close_wig20</th>
      <th>open_mwig40</th>
      <th>high_mwig40</th>
      <th>low_mwig40</th>
      <th>close_mwig40</th>
      <th>open_swig80</th>
      <th>high_swig80</th>
      <th>low_swig80</th>
      <th>close_swig80</th>
    </tr>
    <tr>
      <th>Data</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1991-04-16</th>
      <td>1000.00</td>
      <td>1000.00</td>
      <td>1000.00</td>
      <td>1000.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1991-04-23</th>
      <td>967.70</td>
      <td>967.70</td>
      <td>967.70</td>
      <td>967.70</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1991-04-30</th>
      <td>945.60</td>
      <td>945.60</td>
      <td>945.60</td>
      <td>945.60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1991-05-14</th>
      <td>939.60</td>
      <td>939.60</td>
      <td>939.60</td>
      <td>939.60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1991-05-21</th>
      <td>966.10</td>
      <td>966.10</td>
      <td>966.10</td>
      <td>966.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-12-21</th>
      <td>78341.36</td>
      <td>78840.93</td>
      <td>78237.66</td>
      <td>78723.40</td>
      <td>2333.32</td>
      <td>2355.14</td>
      <td>2329.78</td>
      <td>2348.67</td>
      <td>5876.86</td>
      <td>5880.79</td>
      <td>5833.93</td>
      <td>5880.79</td>
      <td>22627.12</td>
      <td>22644.29</td>
      <td>22520.92</td>
      <td>22596.17</td>
    </tr>
    <tr>
      <th>2023-12-22</th>
      <td>78560.60</td>
      <td>78814.86</td>
      <td>78409.07</td>
      <td>78667.79</td>
      <td>2345.16</td>
      <td>2353.47</td>
      <td>2338.03</td>
      <td>2346.70</td>
      <td>5874.07</td>
      <td>5876.90</td>
      <td>5844.89</td>
      <td>5864.68</td>
      <td>22592.39</td>
      <td>22705.37</td>
      <td>22586.68</td>
      <td>22674.27</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>78916.60</td>
      <td>79460.94</td>
      <td>78784.47</td>
      <td>79246.98</td>
      <td>2348.85</td>
      <td>2378.32</td>
      <td>2348.85</td>
      <td>2371.28</td>
      <td>5872.31</td>
      <td>5883.49</td>
      <td>5842.38</td>
      <td>5847.11</td>
      <td>22758.07</td>
      <td>22858.09</td>
      <td>22758.07</td>
      <td>22802.65</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>79451.43</td>
      <td>79548.69</td>
      <td>78935.53</td>
      <td>78937.48</td>
      <td>2374.38</td>
      <td>2380.55</td>
      <td>2356.32</td>
      <td>2356.32</td>
      <td>5849.55</td>
      <td>5875.54</td>
      <td>5836.06</td>
      <td>5845.89</td>
      <td>22867.93</td>
      <td>22950.33</td>
      <td>22867.67</td>
      <td>22950.33</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>78961.91</td>
      <td>79159.47</td>
      <td>78202.60</td>
      <td>78459.91</td>
      <td>2356.84</td>
      <td>2367.01</td>
      <td>2334.30</td>
      <td>2342.99</td>
      <td>5848.65</td>
      <td>5853.96</td>
      <td>5760.91</td>
      <td>5785.21</td>
      <td>22975.52</td>
      <td>23059.90</td>
      <td>22869.68</td>
      <td>22904.49</td>
    </tr>
  </tbody>
</table>
<p>7732 rows × 16 columns</p>
</div>




```python
# Thrid, check the shape of the combined dataset
data_merged.shape
```




    (7732, 16)




```python
# Let's create a DataFrame object out of the merged data and inspect it
df = pd.DataFrame(data_merged)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 7732 entries, 1991-04-16 to 2023-12-29
    Data columns (total 16 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   open_wig      7732 non-null   float64
     1   high_wig      7732 non-null   float64
     2   low_wig       7732 non-null   float64
     3   close_wig     7732 non-null   float64
     4   open_wig20    7399 non-null   float64
     5   high_wig20    7399 non-null   float64
     6   low_wig20     7399 non-null   float64
     7   close_wig20   7399 non-null   float64
     8   open_mwig40   6509 non-null   float64
     9   high_mwig40   6509 non-null   float64
     10  low_mwig40    6509 non-null   float64
     11  close_mwig40  6509 non-null   float64
     12  open_swig80   7257 non-null   float64
     13  high_swig80   7257 non-null   float64
     14  low_swig80    7257 non-null   float64
     15  close_swig80  7257 non-null   float64
    dtypes: float64(16)
    memory usage: 1.0 MB



```python
# Let's visualise some index level plots
# Prepare a canvas
# Add some titles
# Do some housekeeping
# Plot index closing levels
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
ax11 = axes[0][0]
ax12 = axes[0][1]
ax21 = axes[1][0]
ax22 = axes[1][1]
ax11.title.set_text('WIG Index')
ax12.title.set_text('WIG20 Index')
ax21.title.set_text('mWIG40 Index')
ax22.title.set_text('sWIG80 Index')
df['close_wig'].plot(ax=ax11)
df['close_wig20'].plot(ax=ax12)
df['close_mwig40'].plot(ax=ax21)
df['close_swig80'].plot(ax=ax22)
ax11.set(xlabel=None)
ax12.set(xlabel=None)
ax21.set(xlabel=None)
ax22.set(xlabel=None);
```


![png](output_21_0.png)



```python
# Let's calculate the relative strength of Polish equity indices since 2000 (based on daily log returns)
df1 = df.copy()
df1['wig_r'] = np.log(df1['close_wig'] / df1['close_wig'].shift(1))
df1['wig20_r'] = np.log(df1['close_wig20'] / df1['close_wig20'].shift(1))
df1['mwig40_r'] = np.log(df1['close_mwig40'] / df1['close_mwig40'].shift(1))
df1['swig80_r'] = np.log(df1['close_swig80'] / df1['close_swig80'].shift(1))
df1.dropna(inplace=True)
df1[['wig20_r', 'wig_r', 'mwig40_r', 'swig80_r']].loc['2000-01-01':].sum().apply(np.exp)
```




    wig20_r      1.309958
    wig_r        4.338733
    mwig40_r     5.189460
    swig80_r    10.980627
    dtype: float64




```python
# Plot the relative strength
df1[['wig_r', 'wig20_r', 'mwig40_r', 'swig80_r']].loc['2000-01-01':].cumsum().apply(np.exp).plot();
```


![png](output_23_0.png)


## Further EDA analysis

## WIG Index


```python
qs.plots.snapshot(df['close_wig'], title='WIG index performance', fontname='sans-serif')
qs.plots.drawdowns_periods(df['close_wig'], title='WIG', fontname='sans-serif')
qs.plots.monthly_heatmap(df['close_wig'], fontname='sans-serif', returns_label='WIG')
```


![png](output_26_0.png)



![png](output_26_1.png)



![png](output_26_2.png)


## WIG20 Index


```python
qs.plots.snapshot(df['close_wig20'], title='WIG20 index performance', fontname='sans-serif')
qs.plots.drawdowns_periods(df['close_wig20'], title='WIG20', fontname='sans-serif')
qs.plots.monthly_heatmap(df['close_wig20'], fontname='sans-serif', returns_label='WIG20')
```


![png](output_28_0.png)



![png](output_28_1.png)



![png](output_28_2.png)


## mWIG40 Index


```python
qs.plots.snapshot(df['close_mwig40'], title='mWIG40 index performance', fontname='sans-serif')
qs.plots.drawdowns_periods(df['close_mwig40'], title='mWIG40', fontname='sans-serif')
qs.plots.monthly_heatmap(df['close_mwig40'], fontname='sans-serif', returns_label='mWIG40')
```


![png](output_30_0.png)



![png](output_30_1.png)



![png](output_30_2.png)


## sWIG80 Index


```python
qs.plots.snapshot(df['close_swig80'], title='sWIG80 index performance', fontname='sans-serif')
qs.plots.drawdowns_periods(df['close_swig80'], title='sWIG80', fontname='sans-serif')
qs.plots.monthly_heatmap(df['close_swig80'], fontname='sans-serif', returns_label='sWIG80')
```


![png](output_32_0.png)



![png](output_32_1.png)



![png](output_32_2.png)


## Index level prediction (linear regression)


```python
# Inspect columns of the combined DataFrame
df.columns
```




    Index(['open_wig', 'high_wig', 'low_wig', 'close_wig', 'open_wig20',
           'high_wig20', 'low_wig20', 'close_wig20', 'open_mwig40', 'high_mwig40',
           'low_mwig40', 'close_mwig40', 'open_swig80', 'high_swig80',
           'low_swig80', 'close_swig80'],
          dtype='object')




```python
# Drop unnecessary columns
df1 = df.drop(['open_wig', 
               'high_wig', 
               'low_wig', 
               'open_wig20', 
               'high_wig20', 
               'low_wig20', 
               'open_mwig40', 
               'high_mwig40', 
               'low_mwig40', 
               'open_swig80', 
               'high_swig80', 
               'low_swig80'], axis=1).copy()
df1.shape
```




    (7732, 4)




```python
# Create 5 lagged columns
df_lag = df1.copy()
lag = 5
for lag in range(1, lag + 1):
    shifted = df1.shift(lag)
    shifted.columns = [i + "_lag" + str(lag) for i in df1.columns]
    df_lag = pd.concat((df_lag, shifted), axis=1)
df_lag = df_lag.dropna()
```


```python
# Check for reasonableness
df_lag.columns
```




    Index(['close_wig', 'close_wig20', 'close_mwig40', 'close_swig80',
           'close_wig_lag1', 'close_wig20_lag1', 'close_mwig40_lag1',
           'close_swig80_lag1', 'close_wig_lag2', 'close_wig20_lag2',
           'close_mwig40_lag2', 'close_swig80_lag2', 'close_wig_lag3',
           'close_wig20_lag3', 'close_mwig40_lag3', 'close_swig80_lag3',
           'close_wig_lag4', 'close_wig20_lag4', 'close_mwig40_lag4',
           'close_swig80_lag4', 'close_wig_lag5', 'close_wig20_lag5',
           'close_mwig40_lag5', 'close_swig80_lag5'],
          dtype='object')




```python
# Check: 4 indices + 5 lags each: 4+5*4 = 24 columns 
df_lag.shape
```




    (6504, 24)




```python
# Define target and predictors
target = df_lag[['close_wig', 'close_wig20', 'close_mwig40', 'close_swig80']]
predictors = df_lag[['close_wig_lag1', 'close_wig20_lag1', 'close_mwig40_lag1', 'close_swig80_lag1',
                     'close_wig_lag2', 'close_wig20_lag2', 'close_mwig40_lag2', 'close_swig80_lag2',
                     'close_wig_lag3', 'close_wig20_lag3', 'close_mwig40_lag3', 'close_swig80_lag3',
                     'close_wig_lag4', 'close_wig20_lag4', 'close_mwig40_lag4', 'close_swig80_lag4',
                     'close_wig_lag5', 'close_wig20_lag5', 'close_mwig40_lag5', 'close_swig80_lag5']]
```


```python
# Sense check
df_lag.shape, target.shape, predictors.shape
```




    ((6504, 24), (6504, 4), (6504, 20))




```python
# Perform linear regression (target = close, predictors = lagged close)
# Print the fitted regression parameters
lin_reg = np.linalg.lstsq(predictors[['close_wig_lag1', 'close_wig_lag2', 'close_wig_lag3', 'close_wig_lag4', 'close_wig_lag5']], target['close_wig'], rcond=None)[0]
lin_reg.round(2)
```




    array([ 1.04, -0.07,  0.05, -0.03, -0.  ])




```python
# Add the model's preds to the DataFrame
df_lag['pred'] = np.dot(predictors[['close_wig_lag1', 'close_wig_lag2', 'close_wig_lag3', 'close_wig_lag4', 'close_wig_lag5']], lin_reg)
```


```python
# Plot the index level vs the model predicted level
df_lag[['close_wig', 'pred']].loc['2023-06-01':].plot();
```


![png](output_43_0.png)



```python
## Comment about liner regression assumptions NOT met...
```

## Index return prediction (linear regression)


```python
# Create 5 lagged columns
df_lag = df1.copy()
lag = 5
for lag in range(1, lag + 1):
    shifted = df1.shift(lag)
    shifted.columns = [i + "_lag" + str(lag) for i in df1.columns]
    df_lag = pd.concat((df_lag, shifted), axis=1)
df_lag = df_lag.dropna()
```


```python
# Calculate the daily log returns
# Drop the na value created in the process
df_lag['wig_r'] = np.log(df_lag['close_wig'] / df_lag['close_wig'].shift(1))
df_lag.dropna(inplace=True)
df_lag.columns
```




    Index(['close_wig', 'close_wig20', 'close_mwig40', 'close_swig80',
           'close_wig_lag1', 'close_wig20_lag1', 'close_mwig40_lag1',
           'close_swig80_lag1', 'close_wig_lag2', 'close_wig20_lag2',
           'close_mwig40_lag2', 'close_swig80_lag2', 'close_wig_lag3',
           'close_wig20_lag3', 'close_mwig40_lag3', 'close_swig80_lag3',
           'close_wig_lag4', 'close_wig20_lag4', 'close_mwig40_lag4',
           'close_swig80_lag4', 'close_wig_lag5', 'close_wig20_lag5',
           'close_mwig40_lag5', 'close_swig80_lag5', 'wig_r'],
          dtype='object')




```python
# Plot a histogram of daily log returns against a normal pdf curve
# The realized returns do not follow a normal distribution
plt.hist(df_lag['wig_r'], bins=300, density=True, label='log returns')
plt.xlabel('Log returns')
plt.ylabel('Probability density')
plt.title('WIG log returns distribution')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, df_lag['wig_r'].mean(), df_lag['wig_r'].std()), 'r', lw=2.0, label='pdf')
plt.legend();
```


![png](output_48_0.png)



```python
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'wig_r_lag_{lag}'
    df_lag[col] = df_lag['wig_r'].shift(lag)
    cols.append(col)
df_lag.dropna(inplace=True)
```


```python
df_lag.columns
```




    Index(['close_wig', 'close_wig20', 'close_mwig40', 'close_swig80',
           'close_wig_lag1', 'close_wig20_lag1', 'close_mwig40_lag1',
           'close_swig80_lag1', 'close_wig_lag2', 'close_wig20_lag2',
           'close_mwig40_lag2', 'close_swig80_lag2', 'close_wig_lag3',
           'close_wig20_lag3', 'close_mwig40_lag3', 'close_swig80_lag3',
           'close_wig_lag4', 'close_wig20_lag4', 'close_mwig40_lag4',
           'close_swig80_lag4', 'close_wig_lag5', 'close_wig20_lag5',
           'close_mwig40_lag5', 'close_swig80_lag5', 'wig_r', 'wig_r_lag_1',
           'wig_r_lag_2', 'wig_r_lag_3', 'wig_r_lag_4', 'wig_r_lag_5'],
          dtype='object')




```python
# Perform a linear regression (target = log daily return, predictors = lagged log daily returns)
# Print the fitted regression parameters
lin_reg = np.linalg.lstsq(df_lag[cols], df_lag['wig_r'], rcond=None)[0]
lin_reg.round(2)
```




    array([ 0.08, -0.02,  0.01, -0.01,  0.01])




```python
# Add the model's preds to the DataFrame
df_lag['pred'] = np.dot(df_lag[cols], lin_reg)
```


```python
# Plot the index realized return vs the model predicted return
df_lag[['wig_r', 'pred']].plot();
```


![png](output_53_0.png)



```python
# Calculate a 'hits' ratio (% of correct index direction preds)
hits = np.sign(df_lag['wig_r'] * df_lag['pred']).value_counts()
hits.values[0] / sum(hits)
```




    0.5093875038473377



## Index direction prediction (linear regression)


```python
df_lag = df1.copy()
lag = 5
for lag in range(1, lag + 1):
    shifted = df1.shift(lag)
    shifted.columns = [i + "_lag" + str(lag) for i in df1.columns]
    df_lag = pd.concat((df_lag, shifted), axis=1)
df_lag = df_lag.dropna()
```


```python
df_lag['wig_r'] = np.log(df_lag['close_wig'] / df_lag['close_wig'].shift(1))
df_lag.dropna(inplace=True)
```


```python
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'wig_r_lag_{lag}'
    df_lag[col] = df_lag['wig_r'].shift(lag)
    cols.append(col)
df_lag.dropna(inplace=True)
```


```python
# Build a linear regression model (target = sign of log daily return, predictors = lagged log daily returns)
# Print the fitted regression parameters
lin_reg = np.linalg.lstsq(df_lag[cols], np.sign(df_lag['wig_r']), rcond=None)[0]
lin_reg.round(2)
```




    array([ 2.13, -2.63, -0.64, -0.35,  0.78])




```python
# Add the model's preds to the DataFrame
df_lag['pred'] = np.sign(np.dot(df_lag[cols], lin_reg))
df_lag['pred'].value_counts()
```




    pred
    -1.0    3370
     1.0    3128
    Name: count, dtype: int64




```python
# Calculate a 'hits' ratio (% of correct index direction preds)
hits = np.sign(df_lag['wig_r'] * df_lag['pred']).value_counts()
hits.values[0] / sum(hits)
```




    0.5221606648199446



## Vectorized backtest


```python
# Calculate the 'strategy return' - return realized if investing in the index based on the model's preds (no transaction costs) 
df_lag['strategy_r'] = df_lag['pred'] * df_lag['wig_r']
df_lag[['wig_r', 'strategy_r']].sum().apply(np.exp)
```




    wig_r           5.753205
    strategy_r    116.029776
    dtype: float64




```python
# Plot cumulative returns of the strategy vs the index return
df_lag[['strategy_r', 'wig_r',]].dropna().cumsum().apply(np.exp).plot();
```


![png](output_64_0.png)


## Index direction prediction (logistic regression)


```python
df_lag = df1.copy()
lag = 5
for lag in range(1, lag + 1):
    shifted = df1.shift(lag)
    shifted.columns = [i + "_lag" + str(lag) for i in df1.columns]
    df_lag = pd.concat((df_lag, shifted), axis=1)
df_lag = df_lag.dropna()
```


```python
df_lag['wig_r'] = np.log(df_lag['close_wig'] / df_lag['close_wig'].shift(1))
df_lag.dropna(inplace=True)
```


```python
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'wig_r_lag_{lag}'
    df_lag[col] = df_lag['wig_r'].shift(lag)
    cols.append(col)
df_lag.dropna(inplace=True)
```


```python
# Instantiate a logistic regression model (large C value for no regularization,....)
log_reg = linear_model.LogisticRegression(C=1e7, solver='lbfgs', multi_class='auto', max_iter=1000)
```


```python
# Perform a logistic regression (target = log daily return, predictors = lagged log daily returns)
log_reg.fit(df_lag[cols], np.sign(df_lag['wig_r']))
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=10000000.0, max_iter=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(C=10000000.0, max_iter=1000)</pre></div></div></div></div></div>




```python
# Add the model's preds to the DataFrame
df_lag['pred'] = log_reg.predict(df_lag[cols])
df_lag['pred'].value_counts()
```




    pred
     1.0    5545
    -1.0     953
    Name: count, dtype: int64




```python
# Calculate a 'hits' ratio (% of correct index direction preds)
hits = np.sign(df_lag['wig_r'] * df_lag['pred']).value_counts()
hits
```




     1.0    3392
    -1.0    3105
     0.0       1
    Name: count, dtype: int64




```python
# Calculate accuracy_score metric
accuracy_score(df_lag['pred'], np.sign(df_lag['wig_r']))
```




    0.5220067713142506




```python
# Calculate the 'strategy return' - return realized if investing in the index based on the model's preds (no transaction costs) 
df_lag['strategy_r'] = df_lag['pred'] * df_lag['wig_r']
df_lag[['strategy_r', 'wig_r']].sum().apply(np.exp)
```




    strategy_r    26.902410
    wig_r          5.753205
    dtype: float64




```python
# Plot a cumulative returns of the strategy vs the index
df_lag[['strategy_r', 'wig_r']].dropna().cumsum().apply(np.exp).plot();
```


![png](output_75_0.png)


## Index direction prediction (DNNs)


```python
df_lag = df1.copy()
lag = 5
for lag in range(1, lag + 1):
    shifted = df1.shift(lag)
    shifted.columns = [i + "_lag" + str(lag) for i in df1.columns]
    df_lag = pd.concat((df_lag, shifted), axis=1)
df_lag = df_lag.dropna()
```


```python
df_lag['wig_r'] = np.log(df_lag['close_wig'] / df_lag['close_wig'].shift(1))
df_lag.dropna(inplace=True)
```


```python
# Convert daily log returns into 0 (negative return) or 1 (positive return) 
df_lag['dir'] = np.where(df_lag['wig_r'] > 0, 1, 0)
```


```python
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'wig_r_lag_{lag}'
    df_lag[col] = df_lag['wig_r'].shift(lag)
    cols.append(col)
df_lag.dropna(inplace=True)
df_lag.shape
```




    (6498, 31)




```python
# Perform a train/val/test split, set shuffle to false
# 80% training
# 20% testing 
# 10% validation
x_train, x_test, y_train, y_test = train_test_split(df_lag, df_lag, test_size=0.2, random_state=100, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=100, shuffle=False)

print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)
print("Shape of x_val: ", x_val.shape)
print("Shape of y_val: ", y_val.shape)
```

    Shape of x_train:  (4678, 31)
    Shape of y_train:  (4678, 31)
    Shape of x_test:  (1300, 31)
    Shape of y_test:  (1300, 31)
    Shape of x_val:  (520, 31)
    Shape of y_val:  (520, 31)



```python
print(len(x_train) + len(x_val) + len(x_test))
print(len(y_train) + len(y_val) + len(y_test))
```

    6498
    6498


## A DNN model


```python
set_seeds()
model = Sequential()
model.add(Dense(8, activation="relu", input_shape=(len(cols),)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
```


```python
model.fit(x_train[cols], x_train['dir'], epochs = 25, verbose=False, shuffle=False)
```




    <keras.callbacks.History at 0x7f194508d6d0>




```python
# Plot learning curves
res = pd.DataFrame(model.history.history)
res[['loss', 'accuracy']].plot();
```


![png](output_86_0.png)



```python
model.evaluate(x_train[cols], x_train['dir'])
```

    147/147 [==============================] - 1s 2ms/step - loss: 0.6915 - accuracy: 0.5209





    [0.691490650177002, 0.520949125289917]




```python
pred = np.where(model.predict(x_train[cols]) > 0.5, 1, 0)
x_train['pred'] = np.where(pred > 0, 1, -1)
x_train['strategy_r'] = (x_train['pred'] * x_train['wig_r'])
x_train[['strategy_r', 'wig_r']].cumsum().apply(np.exp).plot();
```

    147/147 [==============================] - 0s 2ms/step



![png](output_88_1.png)



```python
test_scores = model.evaluate(x_test[cols], x_test['dir'], verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

    41/41 - 0s - loss: 0.6928 - accuracy: 0.5162 - 78ms/epoch - 2ms/step
    Test loss: 0.6927697658538818
    Test accuracy: 0.516153872013092



```python
pred = np.where(model.predict(x_test[cols]) > 0.5, 1, 0)
x_test['pred'] = np.where(pred > 0, 1, -1)
x_test['pred'].value_counts()
```

    41/41 [==============================] - 0s 1ms/step





    pred
     1    1298
    -1       2
    Name: count, dtype: int64




```python
x_test['strategy_r'] = (x_test['pred'] * x_test['wig_r'])
x_test[['strategy_r', 'wig_r']].sum().apply(np.exp)
```




    strategy_r    1.500351
    wig_r         1.372728
    dtype: float64




```python
x_test[['strategy_r', 'wig_r', ]].cumsum().apply(np.exp).plot();
```


![png](output_92_0.png)


## A DNN model (more features)


```python
# Let's add some additional features
df_lag['mom'] = df_lag['wig_r'].rolling(5).mean().shift(1)
df_lag['vol'] = df_lag['wig_r'].rolling(20).std().shift(1)
df_lag['dis'] = (df_lag['close_wig'] - df_lag['close_wig'].rolling(50).mean()).shift(1)
df_lag.dropna(inplace=True)
cols.extend(['mom', 'vol', 'dis'])
```


```python
x_train, x_test, y_train, y_test = train_test_split(df_lag, df_lag, test_size=0.2, random_state=100, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=100, shuffle=False)
```


```python
set_seeds()
model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(len(cols),)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
```


```python
model.fit(x_train[cols], x_train['dir'], epochs = 50, verbose=False, shuffle=False)
```




    <keras.callbacks.History at 0x7f1946cdc700>




```python
res = pd.DataFrame(model.history.history)
res[['loss', 'accuracy']].plot(style='--');
```


![png](output_98_0.png)



```python
model.evaluate(x_train[cols], x_train['dir'])
```

    146/146 [==============================] - 0s 2ms/step - loss: 2.6087 - accuracy: 0.5002





    [2.608665943145752, 0.5002154111862183]




```python
pred = np.where(model.predict(x_train[cols]) > 0.5, 1, 0)
x_train['pred'] = np.where(pred > 0, 1, -1)
x_train['strategy_r'] = (x_train['pred'] * x_train['wig_r'])
x_train[['strategy_r', 'wig_r', ]].cumsum().apply(np.exp).plot();
```

    146/146 [==============================] - 0s 2ms/step



![png](output_100_1.png)



```python
test_scores = model.evaluate(x_test[cols], x_test['dir'], verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

    41/41 - 0s - loss: 4.8396 - accuracy: 0.4837 - 102ms/epoch - 2ms/step
    Test loss: 4.839567184448242
    Test accuracy: 0.48372092843055725



```python
pred = np.where(model.predict(x_test[cols]) > 0.5, 1, 0)
x_test['pred'] = np.where(pred > 0, 1, -1)
x_test['pred'].value_counts()
```

    41/41 [==============================] - 0s 2ms/step





    pred
    -1    782
     1    508
    Name: count, dtype: int64




```python
x_test['strategy_r'] = (x_test['pred'] * x_test['wig_r'])
x_test[['strategy_r', 'wig_r', ]].sum().apply(np.exp)
```




    strategy_r    0.422195
    wig_r         1.436764
    dtype: float64




```python
x_test[['strategy_r', 'wig_r']].cumsum().apply(np.exp).plot();
```


![png](output_104_0.png)

