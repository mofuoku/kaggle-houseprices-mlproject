#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Let us load all the necessary libraries
import numpy as np
import os
from time import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
import sklearn
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold,ShuffleSplit,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
from sklearn.metrics import r2_score, make_scorer, mean_squared_error # import metrics from sklearn
from mlxtend.regressor import StackingRegressor

from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score
from sklearn_pandas import DataFrameMapper
from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.externals import joblib

from scipy.stats import skew,randint
from scipy.special import boxcox1p

get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:



#and write a function get_data() to provide us with a dataset to work through this recipe:

#Let us write the following three functions.The function build _model, which implements the Gradient Boosting routine.
#The functions view_model and model_worth, which are used to inspect the model that we have built:

# function handled missing values

# find out the percentage of missing values

def print_feature_importances(model,X):
    important_features = pd.Series(data=rf_model.feature_importances_,index=X.columns)
    important_features.sort_values(ascending=False,inplace=True)
    print(important_features.head(50))
    
def get_cat_columns_by_type(df):
    out = []
    for colname,col_values in df.items():
        if is_string_dtype(col_values):
            out.append((colname,'string') )
        elif not is_numeric_dtype(col_values):
            out.append((colname,'categorical') )
    return out       

def get_numeric_columns(df):
    out = []
    for colname,col_values in df.items():
        if is_numeric_dtype(col_values):
            out.append(colname)
    return out       
    
def get_missing_values_percentage(df):
    missing_values_counts_list = df.isnull().sum()
    total_values = np.product(df.shape)
    total_missing = missing_values_counts_list.sum()
    return (total_missing/total_values) * 100


def convert_to_str_type(df_in,columns,inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
        
    for col in columns:
        df[col] = df[col].astype(str)
    return df

    
def handle_missing_values(df_in,cat_cols=[], num_cols=[],na_dict=None,add_nan_col=True,inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
 
    if na_dict is None:
        na_dict = {}

    for colname, col_values in df.items():   
        if colname not in num_cols:
            continue
        if pd.isnull(col_values).sum():
            df[colname+'_na'] = pd.isnull(col_values)
            filler = na_dict[colname] if colname in na_dict else col_values.median()
            df[colname] = col_values.fillna(filler)
            na_dict[colname] = filler
    for colname in cat_cols:
        if colname not in df.columns:
            continue
        df[colname].fillna(df[colname].mode()[0], inplace=True)
        lbl = LabelEncoder() 
        lbl.fit(list(df[colname].values)) 
        df[colname] = lbl.transform(list(df[colname].values))
    
    return (df,na_dict)



def scale_num_cols(df_in, mapper, inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
        
    if mapper is None:
        map_f = [([c],StandardScaler()) for c in df.columns if is_numeric_dtype(df[c])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return (df,mapper)



def extract_and_drop_target_column(df_in, y_name, inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
    if not is_numeric_dtype(df[y_name]):
        df[y_name] = df[y_name].cat.codes
        y = df[y_name].values
    else:
        y = df[y_name]
    df.drop([y_name], axis=1, inplace=True)
    return (df,y)

def print_mse(m,X_train, X_valid, y_train, y_valid):
    res = [mean_squared_error(y_train,m.predict(X_train)),
                mean_squared_error(y_valid,m.predict(X_valid)),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    print('MSE Training set = {}, MSE Validation set = {}, score Training Set = {}, score on Validation Set = {}'.format(res[0],res[1],res[2], res[3]))
    if hasattr(m, 'oob_score_'):
          print('OOB Score = {}'.format(m.oob_score_))      

def get_iqr_min_max(df,cols):
    out = {}
    for colname, col_values in df.items():
        if colname not in cols:
            continue
        quartile75, quartile25 = np.percentile(col_values, [75 ,25])
        ## Inter Quartile Range ##
        IQR = quartile75 - quartile25
        min_value = quartile25 - (IQR*1.5)
        max_value = quartile75 + (IQR*1.5)
        out[colname] = (min_value,max_value)
    return out


def bin_numerical_columns(df_in,cols,inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
        
    for col in cols.keys():
        bins = cols[col]
        buckets_ = np.linspace(bins[0],bins[1],bins[2])
        df[col] = pd.cut(df[col],buckets_,include_lowest=True)
    return df


# In[13]:


def preprocess_df(df_train,df_test=None,
                  log_y=True,
                  id_col= None,
                  drop_target=True,
                  convert_to_cat_cols=None,
                  remove_skewness=False,scale_mapper=None,
                  bin_columns_dict=None,
                  new_features_func=None):
    
    if drop_target:
        df,y = extract_and_drop_target_column(df_train,'SalePrice',inplace=True)
    if log_y:
        y = np.log1p(y)
    else:
        y = None
        
    combined = pd.concat((df, df_test)).reset_index(drop=True)
    
    
    if id_col is not None:
        combined.drop(id_col, axis=1,inplace=True)
        if df_test is not None:
            test_id = df_test[id_col].copy()
        else: test_id = None
   
    if new_features_func is not None:
        combined = new_features_func(combined)
    
    
    if convert_to_cat_cols is not None:
        combined = convert_to_str_type(combined,convert_to_cat_cols,inplace=True)
    
        
    if bin_columns_dict is not None:
        combined = bin_numerical_columns(combined,bin_columns_dict,inplace=True)
    
    
    cat_cols = get_cat_columns_by_type(combined)
    cat_cols = [cat_cols[i][0] for i in range(len(cat_cols))]
    num_cols = [col for col in combined.columns if col not in cat_cols]
    
    combined = pd.get_dummies(combined,columns=cat_cols, dummy_na=True)
    
    n_train = df.shape[0]
    n_test = df_test.shape[0]
      
    
    combined,d = handle_missing_values(combined,cat_cols=cat_cols,
                                       num_cols=num_cols,inplace=True)
    
    
    if remove_skewness:
        skewed_cols = combined[num_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_cols})
        skewness_log = skewness[skewness > 4.0]
        skewness_other = skewness[skewness <= 4.0]
        skewed_features_log = skewness_log.index
        skewed_features_other = skewness_other.index
        lambda_ = 0.0
        for feature in skewed_features_log:
            combined[feature] = boxcox1p(combined[feature],lambda_)
        lambda_ = 0.15
        for feature in skewed_features_other:
            combined[feature] = boxcox1p(combined[feature],lambda_)
    
    if scale_mapper is not None:
        map_f = [([c],scale_mapper) for c in num_cols]
        mapper = DataFrameMapper(map_f).fit(combined)
    else:
        mapper = None
        
    combined,_ = scale_num_cols(combined,mapper,inplace=True) 
    
    print(get_missing_values_percentage(combined))
    
    return combined,df,y,cat_cols,num_cols,test_id,n_train,n_test


# In[14]:


def add_new_features1(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    return df
def add_new_features2(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    return df
def add_new_features3(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df['TotalLivArea'] = df['GrLivArea'] + df['GarageArea'] + df['LotArea']
    return df

def add_new_features4(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df['TotalLivArea'] = df['GrLivArea'] + df['GarageArea'] + df['LotArea']
    
    df["GrLivArea-2"] = df["GrLivArea"] ** 2
    df["GrLivArea-3"] = df["GrLivArea"] ** 3
    df["GrLivArea-Sq"] = np.sqrt(df["GrLivArea"])
    df["GarageArea-2"] = df["GarageArea"] ** 2
    df["GarageArea-3"] = df["GarageArea"] ** 3
    df["GarageArea-Sq"] = np.sqrt(df["GarageArea"])
    return df


# In[15]:


df_raw = pd.read_csv('/Users/mofuoku/Documents/kaggle-houseprices-mlproject/train.csv')
df_test = pd.read_csv('/Users/mofuoku/Documents/kaggle-houseprices-mlproject/test.csv')
df = df_raw.copy()
stratify_col = df['OverallQual'].copy()
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(
                                       df_train=df,df_test=df_test,
                                       drop_target=True,
                                       new_features_func=add_new_features4,
                                       id_col='Id',
                                       log_y=True,
                                       convert_to_cat_cols=['GarageCars','CentralAir','MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                       remove_skewness=True,
                                       scale_mapper=RobustScaler(),
                                       bin_columns_dict={'OverallQual':(1,11,10),'OverallCond':(1,11,10)} 
                                       )


# In[16]:


df = combined[:n_train]
df_test = combined[n_train:]

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.10,
                                  stratify=stratify_col,shuffle = True,random_state=20)

stratify_X_train = stratify_col[:X_train.shape[0]].copy()
X_train.shape,X_test.shape,y_train.shape,y_test.shape, stratify_X_train.shape


# In[ ]:


X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.10,
                                  stratify=stratify_X_train,shuffle = True,random_state=20)
X_train.shape,X_valid.shape,y_train.shape,y_valid.shape

