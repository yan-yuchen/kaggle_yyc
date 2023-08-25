#  1 | Introduction 

在本次竞赛中，我们的任务是预测非洲 497 个不同地点 2022 年的二氧化碳排放量。 在训练数据中，我们有 2019-2021 年的二氧化碳排放量
    
**本笔记本的内容：**
    
1.通过平滑消除2020年一次性的新冠疫情趋势。 或者，用 2019 年和 2021 年的平均值来估算 2020 年也是一种有效的方法，但此处未实施
    
2. 观察靠近最大排放位置的位置也具有较高的排放水平。 执行 K-Means 聚类以根据数据点的位置对数据点进行聚类。 这允许具有相似排放的数据点被分组在一起
    
3. 以 2019 年和 2020 年为训练数据，用一些集成模型进行实验，以测试其在 2021 年数据上的 CV


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm
from sklearn.preprocessing import SplineTransformer
from holidays import CountryHoliday
from tqdm.notebook import tqdm
from typing import List



from category_encoders import OneHotEncoder, MEstimateEncoder, GLMMEncoder, OrdinalEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold, RepeatedKFold, TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, LabelEncoder, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgbm
from lightgbm import LGBMRegressor, LGBMClassifier
from lightgbm import log_evaluation, early_stopping, record_evaluation
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn import set_config
from sklearn.multioutput import MultiOutputClassifier
from datetime import datetime, timedelta
import gc

import warnings
warnings.filterwarnings('ignore')

set_config(transform_output = 'pandas')

pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)
```


```python
M = 1.07
```

# 2 | Examine Data 

**2.1**
    
在这里，我们试图平滑 2020 年的数据以消除新冠趋势
    
1.使用平滑导入的数据集
2. 使用 2019 年和 2021 年值的平均值 [https://www.kaggle.com/code/kacperrabczewski/rwanda-co2-step-by-step-guide]


```python
extrp = pd.read_csv("./data/PS3E20_train_covid_updated")
extrp = extrp[(extrp["year"] == 2020)]
```


```python
extrp
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
      <th>ID_LAT_LON_YEAR_WEEK</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>year</th>
      <th>week_no</th>
      <th>SulphurDioxide_SO2_column_number_density</th>
      <th>SulphurDioxide_SO2_column_number_density_amf</th>
      <th>SulphurDioxide_SO2_slant_column_number_density</th>
      <th>SulphurDioxide_cloud_fraction</th>
      <th>SulphurDioxide_sensor_azimuth_angle</th>
      <th>...</th>
      <th>Cloud_cloud_top_height</th>
      <th>Cloud_cloud_base_pressure</th>
      <th>Cloud_cloud_base_height</th>
      <th>Cloud_cloud_optical_depth</th>
      <th>Cloud_surface_albedo</th>
      <th>Cloud_sensor_azimuth_angle</th>
      <th>Cloud_sensor_zenith_angle</th>
      <th>Cloud_solar_azimuth_angle</th>
      <th>Cloud_solar_zenith_angle</th>
      <th>emission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53</th>
      <td>ID_-0.510_29.290_2020_00</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2020</td>
      <td>0</td>
      <td>0.000064</td>
      <td>0.970290</td>
      <td>0.000073</td>
      <td>0.163462</td>
      <td>-100.602665</td>
      <td>...</td>
      <td>5388.602054</td>
      <td>60747.063530</td>
      <td>4638.602176</td>
      <td>6.287709</td>
      <td>0.283116</td>
      <td>-13.291375</td>
      <td>33.679610</td>
      <td>-140.309173</td>
      <td>30.053447</td>
      <td>3.753601</td>
    </tr>
    <tr>
      <th>54</th>
      <td>ID_-0.510_29.290_2020_01</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2020</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>6361.488754</td>
      <td>53750.174162</td>
      <td>5361.488754</td>
      <td>19.167269</td>
      <td>0.317732</td>
      <td>-30.474972</td>
      <td>48.119754</td>
      <td>-139.437777</td>
      <td>30.391957</td>
      <td>4.051966</td>
    </tr>
    <tr>
      <th>55</th>
      <td>ID_-0.510_29.290_2020_02</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2020</td>
      <td>2</td>
      <td>-0.000361</td>
      <td>0.668526</td>
      <td>-0.000231</td>
      <td>0.086199</td>
      <td>73.269733</td>
      <td>...</td>
      <td>5320.715902</td>
      <td>61012.625000</td>
      <td>4320.715861</td>
      <td>48.203733</td>
      <td>0.265554</td>
      <td>-12.461150</td>
      <td>35.809728</td>
      <td>-137.854449</td>
      <td>29.100415</td>
      <td>4.154116</td>
    </tr>
    <tr>
      <th>56</th>
      <td>ID_-0.510_29.290_2020_03</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2020</td>
      <td>3</td>
      <td>0.000597</td>
      <td>0.553729</td>
      <td>0.000331</td>
      <td>0.149257</td>
      <td>73.522247</td>
      <td>...</td>
      <td>6219.319294</td>
      <td>55704.782998</td>
      <td>5219.319269</td>
      <td>12.809350</td>
      <td>0.267030</td>
      <td>16.381079</td>
      <td>35.836898</td>
      <td>-139.017754</td>
      <td>26.265561</td>
      <td>4.165751</td>
    </tr>
    <tr>
      <th>57</th>
      <td>ID_-0.510_29.290_2020_04</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2020</td>
      <td>4</td>
      <td>0.000107</td>
      <td>1.045238</td>
      <td>0.000112</td>
      <td>0.224283</td>
      <td>77.588455</td>
      <td>...</td>
      <td>6348.560006</td>
      <td>54829.331776</td>
      <td>5348.560014</td>
      <td>35.283981</td>
      <td>0.268983</td>
      <td>-12.193650</td>
      <td>47.092968</td>
      <td>-134.474279</td>
      <td>27.061321</td>
      <td>4.233635</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78965</th>
      <td>ID_-3.299_30.301_2020_48</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2020</td>
      <td>48</td>
      <td>0.000114</td>
      <td>1.123935</td>
      <td>0.000125</td>
      <td>0.149885</td>
      <td>74.376836</td>
      <td>...</td>
      <td>6092.323722</td>
      <td>57479.397776</td>
      <td>5169.185142</td>
      <td>15.331296</td>
      <td>0.261608</td>
      <td>16.309625</td>
      <td>39.924967</td>
      <td>-132.258700</td>
      <td>30.393604</td>
      <td>26.929207</td>
    </tr>
    <tr>
      <th>78966</th>
      <td>ID_-3.299_30.301_2020_49</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2020</td>
      <td>49</td>
      <td>0.000051</td>
      <td>0.617927</td>
      <td>0.000031</td>
      <td>0.213135</td>
      <td>72.364738</td>
      <td>...</td>
      <td>5992.053006</td>
      <td>57739.300155</td>
      <td>4992.053006</td>
      <td>27.214085</td>
      <td>0.276616</td>
      <td>-0.287656</td>
      <td>45.624810</td>
      <td>-134.460418</td>
      <td>30.911741</td>
      <td>26.606790</td>
    </tr>
    <tr>
      <th>78967</th>
      <td>ID_-3.299_30.301_2020_50</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2020</td>
      <td>50</td>
      <td>-0.000235</td>
      <td>0.633192</td>
      <td>-0.000149</td>
      <td>0.257000</td>
      <td>-99.141518</td>
      <td>...</td>
      <td>6104.231241</td>
      <td>56954.517231</td>
      <td>5181.570213</td>
      <td>26.270365</td>
      <td>0.260574</td>
      <td>-50.411241</td>
      <td>37.645974</td>
      <td>-132.193161</td>
      <td>32.516685</td>
      <td>27.256273</td>
    </tr>
    <tr>
      <th>78968</th>
      <td>ID_-3.299_30.301_2020_51</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2020</td>
      <td>51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4855.537585</td>
      <td>64839.955718</td>
      <td>3858.187453</td>
      <td>14.519789</td>
      <td>0.248484</td>
      <td>30.840922</td>
      <td>39.529722</td>
      <td>-138.964016</td>
      <td>28.574091</td>
      <td>25.591976</td>
    </tr>
    <tr>
      <th>78969</th>
      <td>ID_-3.299_30.301_2020_52</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2020</td>
      <td>52</td>
      <td>0.000025</td>
      <td>1.103025</td>
      <td>0.000028</td>
      <td>0.265622</td>
      <td>-99.811790</td>
      <td>...</td>
      <td>5345.679464</td>
      <td>62098.716546</td>
      <td>4345.679397</td>
      <td>13.082162</td>
      <td>0.283677</td>
      <td>-13.002957</td>
      <td>38.243055</td>
      <td>-136.660958</td>
      <td>29.584058</td>
      <td>25.559870</td>
    </tr>
  </tbody>
</table>
<p>26341 rows × 76 columns</p>
</div>




```python
DATA_DIR = "./data/"
train = pd.read_csv(DATA_DIR + "train.csv")
test = pd.read_csv(DATA_DIR + "test.csv")

def add_features(df):
    #df["week"] = df["year"].astype(str) + "-" + df["week_no"].astype(str)
    #df["date"] = df["week"].apply(lambda x: get_date_from_week_string(x))
    #df = df.drop(columns = ["week"])
    df["week"] = (df["year"] - 2019) * 53 + df["week_no"]
    #df["lat_long"] = df["latitude"].astype(str) + "#" + df["longitude"].astype(str)
    return df

train = add_features(train)
test = add_features(test)
```

**2.2**
       
对预测进行一些有风险的后处理。
    
假设数据点的 MAX = max(2019 年排放量、2020 年排放量、2021 年排放量)。
    
如果 2021 年排放量 > 2019 年排放量，我们将 MAX * 1.07 分配给预测，否则我们只分配 MAX。 参考：https://www.kaggle.com/competitions/playground-series-s3e20/discussion/430152


```python
vals = set()
for x in train[["latitude", "longitude"]].values:
    vals.add(tuple(x))
    
vals = list(vals)
```


```python
zeros = []

for lat, long in vals:
    subset = train[(train["latitude"] == lat) & (train["longitude"] == long)]
    em_vals = subset["emission"].values
    if all(x == 0 for x in em_vals):
        zeros.append([lat, long])
```


```python
test["2021_emission"] = test["week_no"]
test["2020_emission"] = test["week_no"]
test["2019_emission"] = test["week_no"]

for lat, long in vals:
    test.loc[(test.latitude == lat) & (test.longitude == long), "2021_emission"] = train.loc[(train.latitude == lat) & (train.longitude == long) & (train.year == 2021) & (train.week_no <= 48), "emission"].values
    test.loc[(test.latitude == lat) & (test.longitude == long), "2020_emission"] = train.loc[(train.latitude == lat) & (train.longitude == long) & (train.year == 2020) & (train.week_no <= 48), "emission"].values
    test.loc[(test.latitude == lat) & (test.longitude == long), "2019_emission"] = train.loc[(train.latitude == lat) & (train.longitude == long) & (train.year == 2019) & (train.week_no <= 48), "emission"].values
    #print(train.loc[(train.latitude == lat) & (train.longitude == long) & (train.year == 2021), "emission"])
    
test["ratio"] = (test["2021_emission"] / test["2019_emission"]).replace(np.nan, 0)
test["pos_ratio"] = test["ratio"].apply(lambda x: max(x, 1))
test["pos_ratio"] = test["pos_ratio"].apply(lambda x: 1.07 if x > 1 else x)
test["max"] = test[["2019_emission", "2020_emission", "2021_emission"]].max(axis=1)
test["lazy_pred"] = test["max"] * test["pos_ratio"]
test = test.drop(columns = ["ratio", "pos_ratio", "max", "2019_emission", "2020_emission", "2021_emission"])
```


```python
train.loc[train.year == 2020, "emission"] = extrp
```


```python
train
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
      <th>ID_LAT_LON_YEAR_WEEK</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>year</th>
      <th>week_no</th>
      <th>SulphurDioxide_SO2_column_number_density</th>
      <th>SulphurDioxide_SO2_column_number_density_amf</th>
      <th>SulphurDioxide_SO2_slant_column_number_density</th>
      <th>SulphurDioxide_cloud_fraction</th>
      <th>SulphurDioxide_sensor_azimuth_angle</th>
      <th>...</th>
      <th>Cloud_cloud_base_pressure</th>
      <th>Cloud_cloud_base_height</th>
      <th>Cloud_cloud_optical_depth</th>
      <th>Cloud_surface_albedo</th>
      <th>Cloud_sensor_azimuth_angle</th>
      <th>Cloud_sensor_zenith_angle</th>
      <th>Cloud_solar_azimuth_angle</th>
      <th>Cloud_solar_zenith_angle</th>
      <th>emission</th>
      <th>week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID_-0.510_29.290_2019_00</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>0</td>
      <td>-0.000108</td>
      <td>0.603019</td>
      <td>-0.000065</td>
      <td>0.255668</td>
      <td>-98.593887</td>
      <td>...</td>
      <td>61085.809570</td>
      <td>2615.120483</td>
      <td>15.568533</td>
      <td>0.272292</td>
      <td>-12.628986</td>
      <td>35.632416</td>
      <td>-138.786423</td>
      <td>30.752140</td>
      <td>3.750994</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID_-0.510_29.290_2019_01</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>1</td>
      <td>0.000021</td>
      <td>0.728214</td>
      <td>0.000014</td>
      <td>0.130988</td>
      <td>16.592861</td>
      <td>...</td>
      <td>66969.478735</td>
      <td>3174.572424</td>
      <td>8.690601</td>
      <td>0.256830</td>
      <td>30.359375</td>
      <td>39.557633</td>
      <td>-145.183930</td>
      <td>27.251779</td>
      <td>4.025176</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID_-0.510_29.290_2019_02</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>2</td>
      <td>0.000514</td>
      <td>0.748199</td>
      <td>0.000385</td>
      <td>0.110018</td>
      <td>72.795837</td>
      <td>...</td>
      <td>60068.894448</td>
      <td>3516.282669</td>
      <td>21.103410</td>
      <td>0.251101</td>
      <td>15.377883</td>
      <td>30.401823</td>
      <td>-142.519545</td>
      <td>26.193296</td>
      <td>4.231381</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID_-0.510_29.290_2019_03</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>51064.547339</td>
      <td>4180.973322</td>
      <td>15.386899</td>
      <td>0.262043</td>
      <td>-11.293399</td>
      <td>24.380357</td>
      <td>-132.665828</td>
      <td>28.829155</td>
      <td>4.305286</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID_-0.510_29.290_2019_04</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>4</td>
      <td>-0.000079</td>
      <td>0.676296</td>
      <td>-0.000048</td>
      <td>0.121164</td>
      <td>4.121269</td>
      <td>...</td>
      <td>63751.125781</td>
      <td>3355.710107</td>
      <td>8.114694</td>
      <td>0.235847</td>
      <td>38.532263</td>
      <td>37.392979</td>
      <td>-141.509805</td>
      <td>22.204612</td>
      <td>4.347317</td>
      <td>4</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79018</th>
      <td>ID_-3.299_30.301_2021_48</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>48</td>
      <td>0.000284</td>
      <td>1.195643</td>
      <td>0.000340</td>
      <td>0.191313</td>
      <td>72.820518</td>
      <td>...</td>
      <td>60657.101913</td>
      <td>4590.879504</td>
      <td>20.245954</td>
      <td>0.304797</td>
      <td>-35.140368</td>
      <td>40.113533</td>
      <td>-129.935508</td>
      <td>32.095214</td>
      <td>29.404171</td>
      <td>154</td>
    </tr>
    <tr>
      <th>79019</th>
      <td>ID_-3.299_30.301_2021_49</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>49</td>
      <td>0.000083</td>
      <td>1.130868</td>
      <td>0.000063</td>
      <td>0.177222</td>
      <td>-12.856753</td>
      <td>...</td>
      <td>60168.191528</td>
      <td>4659.130378</td>
      <td>6.104610</td>
      <td>0.314015</td>
      <td>4.667058</td>
      <td>47.528435</td>
      <td>-134.252871</td>
      <td>30.771469</td>
      <td>29.186497</td>
      <td>155</td>
    </tr>
    <tr>
      <th>79020</th>
      <td>ID_-3.299_30.301_2021_50</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>56596.027209</td>
      <td>5222.646823</td>
      <td>14.817885</td>
      <td>0.288058</td>
      <td>-0.340922</td>
      <td>35.328098</td>
      <td>-134.731723</td>
      <td>30.716166</td>
      <td>29.131205</td>
      <td>156</td>
    </tr>
    <tr>
      <th>79021</th>
      <td>ID_-3.299_30.301_2021_51</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>51</td>
      <td>-0.000034</td>
      <td>0.879397</td>
      <td>-0.000028</td>
      <td>0.184209</td>
      <td>-100.344827</td>
      <td>...</td>
      <td>46533.348194</td>
      <td>6946.858022</td>
      <td>32.594768</td>
      <td>0.274047</td>
      <td>8.427699</td>
      <td>48.295652</td>
      <td>-139.447849</td>
      <td>29.112868</td>
      <td>28.125792</td>
      <td>157</td>
    </tr>
    <tr>
      <th>79022</th>
      <td>ID_-3.299_30.301_2021_52</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>52</td>
      <td>-0.000091</td>
      <td>0.871951</td>
      <td>-0.000079</td>
      <td>0.000000</td>
      <td>76.825638</td>
      <td>...</td>
      <td>47771.681887</td>
      <td>6553.295018</td>
      <td>19.464032</td>
      <td>0.226276</td>
      <td>-12.808528</td>
      <td>47.923441</td>
      <td>-136.299984</td>
      <td>30.246387</td>
      <td>27.239302</td>
      <td>158</td>
    </tr>
  </tbody>
</table>
<p>79023 rows × 77 columns</p>
</div>




```python
test
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
      <th>ID_LAT_LON_YEAR_WEEK</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>year</th>
      <th>week_no</th>
      <th>SulphurDioxide_SO2_column_number_density</th>
      <th>SulphurDioxide_SO2_column_number_density_amf</th>
      <th>SulphurDioxide_SO2_slant_column_number_density</th>
      <th>SulphurDioxide_cloud_fraction</th>
      <th>SulphurDioxide_sensor_azimuth_angle</th>
      <th>...</th>
      <th>Cloud_cloud_base_pressure</th>
      <th>Cloud_cloud_base_height</th>
      <th>Cloud_cloud_optical_depth</th>
      <th>Cloud_surface_albedo</th>
      <th>Cloud_sensor_azimuth_angle</th>
      <th>Cloud_sensor_zenith_angle</th>
      <th>Cloud_solar_azimuth_angle</th>
      <th>Cloud_solar_zenith_angle</th>
      <th>week</th>
      <th>lazy_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID_-0.510_29.290_2022_00</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>41047.937500</td>
      <td>7472.313477</td>
      <td>7.935617</td>
      <td>0.240773</td>
      <td>-100.113792</td>
      <td>33.697044</td>
      <td>-133.047546</td>
      <td>33.779583</td>
      <td>159</td>
      <td>3.753601</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID_-0.510_29.290_2022_01</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>1</td>
      <td>0.000456</td>
      <td>0.691164</td>
      <td>0.000316</td>
      <td>0.000000</td>
      <td>76.239196</td>
      <td>...</td>
      <td>54915.708579</td>
      <td>5476.147161</td>
      <td>11.448437</td>
      <td>0.293119</td>
      <td>-30.510319</td>
      <td>42.402593</td>
      <td>-138.632822</td>
      <td>31.012380</td>
      <td>160</td>
      <td>4.051966</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID_-0.510_29.290_2022_02</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>2</td>
      <td>0.000161</td>
      <td>0.605107</td>
      <td>0.000106</td>
      <td>0.079870</td>
      <td>-42.055341</td>
      <td>...</td>
      <td>39006.093750</td>
      <td>7984.795703</td>
      <td>10.753179</td>
      <td>0.267130</td>
      <td>39.087361</td>
      <td>45.936480</td>
      <td>-144.784988</td>
      <td>26.743361</td>
      <td>161</td>
      <td>4.231381</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID_-0.510_29.290_2022_03</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>3</td>
      <td>0.000350</td>
      <td>0.696917</td>
      <td>0.000243</td>
      <td>0.201028</td>
      <td>72.169566</td>
      <td>...</td>
      <td>57646.368368</td>
      <td>5014.724115</td>
      <td>11.764556</td>
      <td>0.304679</td>
      <td>-24.465127</td>
      <td>42.140419</td>
      <td>-135.027891</td>
      <td>29.604774</td>
      <td>162</td>
      <td>4.305286</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID_-0.510_29.290_2022_04</td>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>4</td>
      <td>-0.000317</td>
      <td>0.580527</td>
      <td>-0.000184</td>
      <td>0.204352</td>
      <td>76.190865</td>
      <td>...</td>
      <td>52896.541873</td>
      <td>5849.280394</td>
      <td>13.065317</td>
      <td>0.284221</td>
      <td>-12.907850</td>
      <td>30.122641</td>
      <td>-135.500119</td>
      <td>26.276807</td>
      <td>163</td>
      <td>4.347317</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24348</th>
      <td>ID_-3.299_30.301_2022_44</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>44</td>
      <td>-0.000618</td>
      <td>0.745549</td>
      <td>-0.000461</td>
      <td>0.234492</td>
      <td>72.306198</td>
      <td>...</td>
      <td>55483.459980</td>
      <td>5260.120056</td>
      <td>30.398508</td>
      <td>0.180046</td>
      <td>-25.528588</td>
      <td>45.284576</td>
      <td>-116.521412</td>
      <td>29.992562</td>
      <td>203</td>
      <td>30.327420</td>
    </tr>
    <tr>
      <th>24349</th>
      <td>ID_-3.299_30.301_2022_45</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>45</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>53589.917383</td>
      <td>5678.951521</td>
      <td>19.223844</td>
      <td>0.177833</td>
      <td>-13.380005</td>
      <td>43.770351</td>
      <td>-122.405759</td>
      <td>29.017975</td>
      <td>204</td>
      <td>30.811167</td>
    </tr>
    <tr>
      <th>24350</th>
      <td>ID_-3.299_30.301_2022_46</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>46</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>62646.761340</td>
      <td>4336.282491</td>
      <td>13.801194</td>
      <td>0.219471</td>
      <td>-5.072065</td>
      <td>33.226455</td>
      <td>-124.530639</td>
      <td>30.187472</td>
      <td>205</td>
      <td>31.162886</td>
    </tr>
    <tr>
      <th>24351</th>
      <td>ID_-3.299_30.301_2022_47</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>47</td>
      <td>0.000071</td>
      <td>1.003805</td>
      <td>0.000077</td>
      <td>0.205077</td>
      <td>74.327427</td>
      <td>...</td>
      <td>50728.313991</td>
      <td>6188.578464</td>
      <td>27.887489</td>
      <td>0.247275</td>
      <td>-0.668714</td>
      <td>45.885617</td>
      <td>-129.006797</td>
      <td>30.427455</td>
      <td>206</td>
      <td>31.439606</td>
    </tr>
    <tr>
      <th>24352</th>
      <td>ID_-3.299_30.301_2022_48</td>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>48</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>46260.039092</td>
      <td>6777.863819</td>
      <td>23.771269</td>
      <td>0.239684</td>
      <td>-40.826139</td>
      <td>30.680056</td>
      <td>-124.895473</td>
      <td>34.457720</td>
      <td>207</td>
      <td>29.944366</td>
    </tr>
  </tbody>
</table>
<p>24353 rows × 77 columns</p>
</div>



<div style="border-radius:10px; border:#FF0000 solid; padding: 15px; background-color: #F3f9ed; font-size:100%; text-align:left">
    
**Insights**
    
The train dataset has 79023 observations and the test dataset has 24353 observations. As we observe, some columns have null values

#  3 | EDA and Data Distribution


```python
def plot_emission(train):
    
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=train, x="week", y="emission", label="Emission", alpha=0.7, color='blue')

    plt.xlabel('Week')
    plt.ylabel('Emission')
    plt.title('Emission over time')

    plt.legend()
    plt.tight_layout()
    plt.show()
    
plot_emission(train)
```


    
![png](private-lb-9-61-s3e20-kmeans-lightgbm_files/private-lb-9-61-s3e20-kmeans-lightgbm_18_0.png)
    



```python
sns.histplot(train["emission"])
```




    <Axes: xlabel='emission', ylabel='Count'>




    
![png](private-lb-9-61-s3e20-kmeans-lightgbm_files/private-lb-9-61-s3e20-kmeans-lightgbm_19_1.png)
    


#  4 | Data Transformation 



```python
print(len(vals))
```

    497
    

 **Insights**
    
有 497 个独特的经纬度组合

<div style="border-radius:10px; border:#2e3ca5 solid; padding: 15px; background-color: #F3f9ed; font-size:100%; text-align:left">
    
**4.1**
    
Most of the features are just noise, we can remove them. (Reference: multiple discussion posts)


```python
#train = train.drop(columns = ["ID_LAT_LON_YEAR_WEEK", "lat_long"])
#test = test.drop(columns = ["ID_LAT_LON_YEAR_WEEK", "lat_long"])

train = train[["latitude", "longitude", "year", "week_no", "emission"]]
test = test[["latitude", "longitude", "year", "week_no", "lazy_pred"]]
```

**4.2**
    
K Means Clustering + Distance to highest emission


```python
#https://www.kaggle.com/code/lucasboesen/simple-catboost-6-features-cv-21-7
from sklearn.cluster import KMeans
import haversine as hs

km_train = train.groupby(by=['latitude', 'longitude'], as_index=False)['emission'].mean()
model = KMeans(n_clusters = 7, random_state = 42)
model.fit(km_train)
yhat_train = model.predict(km_train)
km_train['kmeans_group'] = yhat_train

""" Own Groups """
# Some locations have emission == 0
km_train['is_zero'] = km_train['emission'].apply(lambda x: 'no_emission_recorded' if x==0 else 'emission_recorded')

# Distance to the highest emission location
max_lat_lon_emission = km_train.loc[km_train['emission']==km_train['emission'].max(), ['latitude', 'longitude']]
km_train['distance_to_max_emission'] = km_train.apply(lambda x: hs.haversine((x['latitude'], x['longitude']), (max_lat_lon_emission['latitude'].values[0], max_lat_lon_emission['longitude'].values[0])), axis=1)

train = train.merge(km_train[['latitude', 'longitude', 'kmeans_group', 'distance_to_max_emission']], on=['latitude', 'longitude'])
test = test.merge(km_train[['latitude', 'longitude', 'kmeans_group', 'distance_to_max_emission']], on=['latitude', 'longitude'])
#train = train.drop(columns = ["latitude", "longitude"])
#test = test.drop(columns = ["latitude", "longitude"])
```


```python
train
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
      <th>latitude</th>
      <th>longitude</th>
      <th>year</th>
      <th>week_no</th>
      <th>emission</th>
      <th>kmeans_group</th>
      <th>distance_to_max_emission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>0</td>
      <td>3.750994</td>
      <td>6</td>
      <td>207.849890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>1</td>
      <td>4.025176</td>
      <td>6</td>
      <td>207.849890</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>2</td>
      <td>4.231381</td>
      <td>6</td>
      <td>207.849890</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>3</td>
      <td>4.305286</td>
      <td>6</td>
      <td>207.849890</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2019</td>
      <td>4</td>
      <td>4.347317</td>
      <td>6</td>
      <td>207.849890</td>
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
    </tr>
    <tr>
      <th>79018</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>48</td>
      <td>29.404171</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
    <tr>
      <th>79019</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>49</td>
      <td>29.186497</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
    <tr>
      <th>79020</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>50</td>
      <td>29.131205</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
    <tr>
      <th>79021</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>51</td>
      <td>28.125792</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
    <tr>
      <th>79022</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2021</td>
      <td>52</td>
      <td>27.239302</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
  </tbody>
</table>
<p>79023 rows × 7 columns</p>
</div>




```python
test
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
      <th>latitude</th>
      <th>longitude</th>
      <th>year</th>
      <th>week_no</th>
      <th>lazy_pred</th>
      <th>kmeans_group</th>
      <th>distance_to_max_emission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>0</td>
      <td>3.753601</td>
      <td>6</td>
      <td>207.849890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>1</td>
      <td>4.051966</td>
      <td>6</td>
      <td>207.849890</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>2</td>
      <td>4.231381</td>
      <td>6</td>
      <td>207.849890</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>3</td>
      <td>4.305286</td>
      <td>6</td>
      <td>207.849890</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.510</td>
      <td>29.290</td>
      <td>2022</td>
      <td>4</td>
      <td>4.347317</td>
      <td>6</td>
      <td>207.849890</td>
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
    </tr>
    <tr>
      <th>24348</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>44</td>
      <td>30.327420</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
    <tr>
      <th>24349</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>45</td>
      <td>30.811167</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
    <tr>
      <th>24350</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>46</td>
      <td>31.162886</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
    <tr>
      <th>24351</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>47</td>
      <td>31.439606</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
    <tr>
      <th>24352</th>
      <td>-3.299</td>
      <td>30.301</td>
      <td>2022</td>
      <td>48</td>
      <td>29.944366</td>
      <td>6</td>
      <td>157.630611</td>
    </tr>
  </tbody>
</table>
<p>24353 rows × 7 columns</p>
</div>




```python
cat_params = {
    
    'n_estimators': 799, 
    'learning_rate': 0.09180872710592884,
    'depth': 8, 
    'l2_leaf_reg': 1.0242996861886846, 
    'subsample': 0.38227256755249117, 
    'colsample_bylevel': 0.7183481537623551,
    'random_state': 42,
    "silent": True,
}

lgb_params = {
    
    'n_estimators': 835, 
    'max_depth': 12, 
    'reg_alpha': 3.849279869880706, 
    'reg_lambda': 0.6840221712299135, 
    'min_child_samples': 10, 
    'subsample': 0.6810493885301987, 
    'learning_rate': 0.0916362259866008, 
    'colsample_bytree': 0.3133780298325982, 
    'colsample_bynode': 0.7966712089198238,
    "random_state": 42,
}

xgb_params = {
    
    "random_state": 42,
}

rf_params = {
    
    'n_estimators': 263, 
    'max_depth': 41, 
    'min_samples_split': 10, 
    'min_samples_leaf': 3,
    "random_state": 42,
    "verbose": 0
}

et_params = {
    
    "random_state": 42,
    "verbose": 0
}
```

#  5 | Validate Performance on 2021 data


```python
def rmse(a, b):
    return mean_squared_error(a, b, squared=False)
```


```python
validation = train[train.year == 2021]
clusters = train["kmeans_group"].unique()

for i in range(len(clusters)):
               
    cluster = clusters[i]
    
    print("==============================================")
    print(f" Cluster {cluster} ")
    
    
    train_c = train[train["kmeans_group"] == cluster]
    
    X_train = train_c[train_c.year < 2021].drop(columns = ["emission", "kmeans_group"])
    y_train = train_c[train_c.year < 2021]["emission"].copy()
    X_val = train_c[train_c.year >= 2021].drop(columns = ["emission", "kmeans_group"])
    y_val = train_c[train_c.year >= 2021]["emission"].copy()
    
    
    
    #=======================================================================================
    catboost_reg = CatBoostRegressor(**cat_params)
    catboost_reg.fit(X_train, y_train, eval_set=(X_val, y_val))

    catboost_pred = catboost_reg.predict(X_val) * M
    print(f"RMSE of CatBoost: {rmse(catboost_pred, y_val)}")

    #=======================================================================================
    lightgbm_reg = LGBMRegressor(**lgb_params,verbose=-1)
    lightgbm_reg.fit(X_train, y_train, eval_set=(X_val, y_val))

    lightgbm_pred = lightgbm_reg.predict(X_val) * M
    print(f"RMSE of LightGBM: {rmse(lightgbm_pred, y_val)}")

    #=======================================================================================
    xgb_reg = XGBRegressor(**xgb_params)
    xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose = False)

    xgb_pred = xgb_reg.predict(X_val) * M
    print(f"RMSE of XGBoost: {rmse(xgb_pred, y_val)}")

    #=======================================================================================
    rf_reg = RandomForestRegressor(**rf_params)
    rf_reg.fit(X_train, y_train)

    rf_pred = rf_reg.predict(X_val) * M
    print(f"RMSE of Random Forest: {rmse(rf_pred, y_val)}")

    #=======================================================================================
    et_reg = ExtraTreesRegressor(**et_params)
    et_reg.fit(X_train, y_train)

    et_pred = et_reg.predict(X_val) * M
    print(f"RMSE of Extra Trees: {rmse(et_pred, y_val)}")
    
    
    overall_pred = lightgbm_pred #(catboost_pred + lightgbm_pred) / 2
    validation.loc[validation["kmeans_group"] == cluster, "emission"] = overall_pred
    
    print(f"RMSE Overall: {rmse(overall_pred, y_val)}")

print("==============================================")
print(f"[DONE] RMSE of all clusters: {rmse(validation['emission'], train[train.year == 2021]['emission'])}")
print(f"[DONE] RMSE of all clusters Week 1-20: {rmse(validation[validation.week_no < 21]['emission'], train[(train.year == 2021) & (train.week_no < 21)]['emission'])}")
print(f"[DONE] RMSE of all clusters Week 21+: {rmse(validation[validation.week_no >= 21]['emission'], train[(train.year == 2021) & (train.week_no  >= 21)]['emission'])}")
```

    ==============================================
     Cluster 6 
    RMSE of CatBoost: 2.3575606902299895
    RMSE of LightGBM: 2.2103640167714094
    RMSE of XGBoost: 2.5018849673349863
    RMSE of Random Forest: 2.6335510523545556
    RMSE of Extra Trees: 3.0029623116826776
    RMSE Overall: 2.2103640167714094
    ==============================================
     Cluster 5 
    RMSE of CatBoost: 19.175306730779514
    RMSE of LightGBM: 17.910821889134688
    RMSE of XGBoost: 19.6677120674706
    RMSE of Random Forest: 18.856743714624777
    RMSE of Extra Trees: 20.70417439300032
    RMSE Overall: 17.910821889134688
    ==============================================
     Cluster 1 
    RMSE of CatBoost: 9.26195004601851
    RMSE of LightGBM: 8.513309514506675
    RMSE of XGBoost: 10.137965612920658
    RMSE of Random Forest: 9.838001199034126
    RMSE of Extra Trees: 11.043246766709913
    RMSE Overall: 8.513309514506675
    ==============================================
     Cluster 4 
    RMSE of CatBoost: 44.564695183442716
    RMSE of LightGBM: 43.946690922308754
    RMSE of XGBoost: 50.18811358270916
    RMSE of Random Forest: 46.39201148051631
    RMSE of Extra Trees: 50.58999576441371
    RMSE Overall: 43.946690922308754
    ==============================================
     Cluster 0 
    RMSE of CatBoost: 28.408461784012662
    RMSE of LightGBM: 26.872533954605416
    RMSE of XGBoost: 30.622689084145943
    RMSE of Random Forest: 28.46657485784377
    RMSE of Extra Trees: 31.733046766544884
    RMSE Overall: 26.872533954605416
    ==============================================
     Cluster 3 
    RMSE of CatBoost: 263.29528869714665
    RMSE of LightGBM: 326.12883397111284
    RMSE of XGBoost: 336.5771065570381
    RMSE of Random Forest: 303.9321016178147
    RMSE of Extra Trees: 336.67756932119914
    RMSE Overall: 326.12883397111284
    ==============================================
     Cluster 2 
    RMSE of CatBoost: 206.96165808156715
    RMSE of LightGBM: 222.40891682146665
    RMSE of XGBoost: 281.12604107718465
    RMSE of Random Forest: 232.11332438348992
    RMSE of Extra Trees: 281.29392713471816
    RMSE Overall: 222.40891682146665
    ==============================================
    [DONE] RMSE of all clusters: 23.275548123498453
    [DONE] RMSE of all clusters Week 1-20: 31.92891146501802
    [DONE] RMSE of all clusters Week 21+: 15.108200701163458
    

#  6 | Predicting 2022 result


```python
clusters = train["kmeans_group"].unique()

for i in tqdm(range(len(clusters))):
    
    cluster = clusters[i]
    
    train_c = train[train["kmeans_group"] == cluster]
    if "emission" in test.columns:
        test_c = test[test["kmeans_group"] == cluster].drop(columns = ["emission", "kmeans_group", "lazy_pred"])
    else:
        test_c = test[test["kmeans_group"] == cluster].drop(columns = ["kmeans_group", "lazy_pred"])
    
    X = train_c.drop(columns = ["emission", "kmeans_group"])
    y = train_c["emission"].copy()
    #=======================================================================================
    catboost_reg = CatBoostRegressor(**cat_params)
    catboost_reg.fit(X, y)
    #print(test_c)

    catboost_pred = catboost_reg.predict(test_c)

    #=======================================================================================
    lightgbm_reg = LGBMRegressor(**lgb_params,verbose=-1)
    lightgbm_reg.fit(X, y)
    #print(test_c)

    lightgbm_pred = lightgbm_reg.predict(test_c)

    #=======================================================================================
    #xgb_reg = XGBRegressor(**xgb_params)
    #xgb_reg.fit(X, y, verbose = False)

    #xgb_pred = xgb_reg.predict(test)

    #=======================================================================================
    rf_reg = RandomForestRegressor(**rf_params)
    rf_reg.fit(X, y)

    rf_pred = rf_reg.predict(test_c)

    #=======================================================================================
    #et_reg = ExtraTreesRegressor(**et_params)
    #et_reg.fit(X, y)

    #et_pred = et_reg.predict(test)

    overall_pred = lightgbm_pred #(catboost_pred + lightgbm_pred) / 2
    test.loc[test["kmeans_group"] == cluster, "emission"] = overall_pred
```


      0%|          | 0/7 [00:00<?, ?it/s]



```python
test["emission"] = test["emission"] * 1.07
```


```python
test.to_csv('submission.csv', index=False)
```
