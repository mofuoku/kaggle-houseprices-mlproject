#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import os
from time import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sklearn
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold,ShuffleSplit,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import Lasso


from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score
from sklearn_pandas import DataFrameMapper
from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.externals import joblib

from scipy.stats import skew,randint
from scipy.special import boxcox1p
import seaborn as sns
import warnings
warnings.warn("once")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:



#Create Function That Constructs A Neural Network

# #standardizing the input feature

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

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
    # percent of data that is missing
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


# Drop Id column
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


# Log transform of the skewed numerical features to lessen impact of outliers
# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed


# In[4]:


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


# In[5]:


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


# In[6]:


rs_const = 80 # to apply in all model tuning process
test_ratio_const = 0.2 

#Reading the data
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
 
# creating training and test data


# In[7]:


# Preprocessing
df = combined[:n_train]
df_test = combined[n_train:]


#creating training and test data split
X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.10,
                                  stratify=stratify_col,shuffle = True,random_state=20)

stratify_X_train = stratify_col[:X_train.shape[0]].copy()
X_train.shape,X_test.shape,y_train.shape,y_test.shape, stratify_X_train.shape


# In[8]:


X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.10,
                                  stratify=stratify_X_train,shuffle = True,random_state=20)
X_train.shape,X_valid.shape,y_train.shape,y_valid.shape


# In[9]:


### Model #1: Linear regression


# In[10]:


# 1. linear regression:
model_lr = linear_model.LinearRegression()
model_lr.fit(X_train, y_train)

print("RMSE train: {}".format(rmse(y_train, model_lr.predict(X_train))))
print("RMSE test : {}".format(rmse(y_test,  model_lr.predict(X_test))))


# In[11]:


### Model #2: LASSO


# In[12]:


#Setting the hyperparameters
grid_param = [{'alpha': np.logspace(-4, 4, 20)}]

# Create grid search
gs = GridSearchCV(estimator=Lasso(random_state=rs_const), param_grid=grid_param, cv=5)

#Fit grid search
gs.fit(X_train, y_train)

#View hyperparameters of best elastic net: best parameters and best score 
print('Best params: {}'.format(gs.best_params_))
print('Best score : {}'.format(gs.best_score_))

#print('')

model_lasso = gs.best_estimator_
print("RMSE train: {}".format(rmse(y_train, model_lasso.predict(X_train))))
print("RMSE test : {}".format(rmse(y_test,  model_lasso.predict(X_test))))


# In[13]:


### Model #3: Ridge


# In[14]:


#Setting the hyperparameters
grid_param = [{'alpha': np.logspace(-4, 4, 20)}]

# Create grid search
gs = GridSearchCV(estimator=linear_model.Ridge(random_state=rs_const), param_grid=grid_param, cv=5)

#Fit grid search
gs.fit(X_train, y_train)

#View hyperparameters of best elastic net: best parameters and best score 
print('Best params: {}'.format(gs.best_params_))
print('Best score : {}'.format(gs.best_score_))

#print('')

#Fit grid search
model_ridge = gs.best_estimator_
print("RMSE train: {}".format(rmse(y_train, model_ridge.predict(X_train))))
print("RMSE test : {}".format(rmse(y_test,  model_ridge.predict(X_test))))


#Best params: {'alpha': 29.763514416313132}
#Best score : 0.8855270998384609
#RMSE train: 0.10798557947419832
#RMSE test : 0.11388274212447386


# In[15]:


from sklearn.linear_model import Ridge

from yellowbrick.regressor import PredictionError

# Instantiate the linear model and visualizer
linear_model.Ridge = linear_model.Ridge()
visualizer = PredictionError(linear_model.Ridge)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# ### Model #2: ElasticNet

# In[22]:


#Setting the hyperparameters 
grid_param = [{'alpha':[0.001,0.01,0.1,1.],
          'l1_ratio': [0.4,0.5,0.6,0.7,0.8,0.9],
          'max_iter':[1000,2000,5000,10000],
          'selection':['cyclic','random']}]


# Create grid search
gs = GridSearchCV(estimator = ElasticNet(random_state=rs_const, normalize=False), param_grid=grid_param, cv=5)

#Fit grid search
gs.fit(X_train, y_train)


#View hyperparameters of best elastic net: best parameters and best score 
print('Best params: {}'.format(gs.best_params_))
print('Best score : {}'.format(gs.best_score_))

#print('')
model_enet = gs.best_estimator_
print("RMSE train: {}".format(rmse(y_train, model_enet.predict(X_train))))
print("RMSE test : {}".format(rmse(y_test,  model_enet.predict(X_test))))

#Best params: {'alpha': 0.001, 'l1_ratio': 0.8, 'max_iter': 1000, 'selection': 'cyclic'}
#Best score : 0.8874114596083972
#RMSE train: 0.11251600848320001
#RMSE test : 0.11083347167509701


# In[16]:


from sklearn.linear_model import ElasticNet

from yellowbrick.regressor import PredictionError

# Instantiate the linear model and visualizer
ElasticNet = ElasticNet()
visualizer = PredictionError(ElasticNet)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# ### Model #3: Gradient Boost

# In[17]:


#Start
#1. param_dist = {'max_features' : ['sqrt'],
 #'max_depth' : [3, 5, 7, 10],
 #'min_samples_leaf' : [5, 10,15, 20],
 #'min_samples_split': [2, 5, 10, 15,20],    # 10      
 #'n_estimators': [1000,1200, 1500],
 #'learning_rate' : [0.0001,0.001,0.01,0.05,0.1,0.3],
 #'loss' : ['ls','huber'],
 #'subsample':[0.6,0.7,0.75,0.8,0.9],
 #}

#2. param_dist2 = {'max_features' : ['sqrt'],
 #'max_depth' : [2,3],  # 2 and 3
 #'min_samples_leaf' : [19,20], 
 #'min_samples_split': [14,15],     
 #'n_estimators': [1400, 1500],
 #'learning_rate' : [0.01,0.02,0.05], 
 #'loss' : ['huber'],
 #'subsample':[0.78,0.8],  
 #}

#3. param_dist3 = {'max_features': 'sqrt',
 #'max_depth': 3,     # max depth 5 better
 #'min_samples_leaf': 19, 
 #'min_samples_split': 14,  # best is 14     
#'n_estimators': 1400, #1500 is the best estimator
 #'learning_rate': 0.02,
 #'loss' : 'huber',
 #'subsample': 0.78,
 #}
    

#Setting the hyperparameters 
grid_param = [{'n_estimators': [1400],
              'max_features': ['sqrt'],
              'max_depth':[3],
              'learning_rate':[0.02],
              'loss':['huber'],
              'min_samples_split': [14],
              'min_samples_leaf': [19],
              'subsample': [0.78]}]


# Create grid search
gs = GridSearchCV(estimator = GradientBoostingRegressor(random_state=rs_const), param_grid=grid_param, cv=5)

#Fit grid search
gs.fit(X_train, y_train)

#View hyperparameters of best gradient boost: best parameters and best score 
print('Best params: {}'.format(gs.best_params_))
print('Best score : {}'.format(gs.best_score_))

#print('')

model_gbt = gs.best_estimator_
print("RMSE train: {}".format(rmse(y_train, model_gbt.predict(X_train))))
print("RMSE test : {}".format(rmse(y_test, model_gbt.predict(X_test))))


#Best params: {'learning_rate': 0.02, 'loss': 'huber', 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 19, 'min_samples_split': 14, 'n_estimators': 1400, 'subsample': 0.8}
#Best score : 0.8921460458464987
#RMSE train: 0.10404951521718483
#RMSE test : 0.11068459534560152
#MSE Training set = 0.01082630161693118, MSE Validation set = 0.010087653543690065, score Training Set = 0.932355335303073, score on Validation Set = 0.9287313699400164


# ### Model #4: Light Gradient Boost

# In[18]:


#Start

#1.
#'num_leaves':[4,6,8,10],
#'learning_rate':[0.001, 0.005, 0.01, 0.05, 0.1], 
#'n_estimators':[1300, 1500, 1800, 2000],
#'max_bin': [60, 66, 76,100], 
#'bagging_fraction': [0.5,0.6, 0.7, 0.8, 0.9],
#'bagging_freq': [3,4, 5, 6, 7, 8], 
#'feature_fraction': [0.2, 0.4, 0.6, 0.8],
#'feature_fraction_seed':[2,4,9,10], 
#'bagging_seed': [4,9,12],
#'min_sum_hessian_in_leaf': [1,11,20],
#'min_data_in_leaf':[4,6,8,10]

#2.
#'num_leaves': [4,8],
#'learning_rate':[0.005,0.01,0.005], #,  try #0.006, 0.007,0.1
#'n_estimators':[1300,1500],  #try 1400, 1500
#'max_bin': [66,76],          # 66
#'bagging_fraction': [0.6,0.7], # 0.8, 0.85, 0.9
#'bagging_freq': [8],  #4
#'feature_fraction': [0.6,0.7], # 0.8
#'feature_fraction_seed':[12], 
#'bagging_seed': [9],
#'min_data_in_leaf': [21,22], # 19,20
#'min_sum_hessian_in_leaf': [1,2] #19,20


#3. 
# 'num_leaves': [7,8],
#'learning_rate':[0.006,0.01],  
#'n_estimators':[1500],  
#'max_bin': [66],          
#'bagging_fraction': [0.2], 
#'bagging_freq': [8],  
#'feature_fraction': [0.2], 
#'feature_fraction_seed':[2], 
#'bagging_seed': [12],
#'min_data_in_leaf': [5], 
#'min_sum_hessian_in_leaf': [1] 




#Setting the hyperparameters
grid_param = [{'num_leaves': [8],
'learning_rate': [0.006], 
'n_estimators':[1493],
'max_bin':[66], 
'bagging_fraction':[0.2],
'bagging_freq': [8], 
'feature_fraction':[0.2],
'feature_fraction_seed':[2], 
'bagging_seed': [12],
'min_data_in_leaf':[5], 
'min_sum_hessian_in_leaf':[1]}]


# Create grid search
gs = GridSearchCV(estimator = lgb.LGBMRegressor(random_state=rs_const), param_grid=grid_param, cv=5)

#Fit grid search

gs.fit(X_train, y_train)

#View hyperparameters of best gradient boost: best parameters and best score  
print('Best params: {}'.format(gs.best_params_))
print('Best score : {}'.format(gs.best_score_))

#print('')

model_lgb = gs.best_estimator_
print("RMSE train: {}".format(rmse(y_train, model_lgb.predict(X_train))))
print("RMSE test : {}".format(rmse(y_test,  model_lgb.predict(X_test))))
#print_mse(gs, X_train,X_valid,y_train,y_valid)

#Best params: {'bagging_fraction': 0.2, 'bagging_freq': 8, 'bagging_seed': 12, 'feature_fraction': 0.2, 'feature_fraction_seed': 2, 'learning_rate': 0.006, 'max_bin': 66, 'min_data_in_leaf': 5, 'min_sum_hessian_in_leaf': 1, 'n_estimators': 1493, 'num_leaves': 8}
#Best score : 0.8913414601865287
#RMSE train: 0.09900935536017762
#RMSE test : 0.1163715169101849
#MSE Training set = 0.009802852448837931, MSE Validation set = 0.009799535050139148, score Training Set = 0.9387500283625891, score on Validation Set = 0.9307669087540118


# In[19]:


from sklearn.linear_model import Lasso

from yellowbrick.regressor import PredictionError

# Instantiate the linear model and visualizer
lgb.LGBMRegressor = lgb.LGBMRegressor()
visualizer = PredictionError(lgb.LGBMRegressor)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# ### Model #5: Multi-layer Perceptron Model

# In[29]:


#Setting the hyperparameters
grid_param = [{'hidden_layer_sizes': [8,9,10,20,30,40,50,60,70,80,90,100],
'activation': ['relu'],
'solver':['lbfgs'], 'alpha':[10,1,0.1,0.01],
'batch_size':['auto'], 'learning_rate':['constant'],
'learning_rate_init':[0.001], 'max_iter':[500]}]

# Create grid search
gs = GridSearchCV(estimator = MLPRegressor(random_state=rs_const), param_grid=grid_param, cv=5)

#Fit grid search
gs.fit(X_train, y_train)

#View hyperparameters of best mlp: best parameters and best score 
print('Best params: {}'.format(gs.best_params_))
print('Best score : {}'.format(gs.best_score_))
#print('')
model_nn = gs.best_estimator_
print("RMSE train: {}".format(rmse(y_train, model_nn.predict(X_train))))
print("RMSE test : {}".format(rmse(y_test,  model_nn.predict(X_test))))

est params: {'activation': 'relu', 'alpha': 10, 'batch_size': 'auto', 'hidden_layer_sizes': 9, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_iter': 500, 'solver': 'lbfgs'}
Best score : 0.8726359334640122
RMSE train: 0.1189130303889917
RMSE test : 0.12200019103394265


# ### Model: Random Forest

# In[34]:


#Setting the hyperparameters
grid_param = [{'n_estimators':[1500],
              'max_features': [436],
              'min_samples_split': [4],
              'min_samples_leaf': [2]
              }]
 
    
# Create grid search
gs = GridSearchCV(estimator = RandomForestRegressor(random_state=rs_const), param_grid=grid_param, cv=5)


gs.fit(X_train, y_train)

#View hyperparameters of best mlp: best parameters and best score
print('Best params: {}'.format(gs.best_params_))
print('Best score : {}'.format(gs.best_score_))

#print('')

model_rf = gs.best_estimator_
print("RMSE train: {}".format(rmse(y_train, model_rf.predict(X_train))))
print("RMSE test : {}".format(rmse(y_test,  model_rf.predict(X_test))))

#Best params: {'max_features': 436, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 1500}
#Best score : 0.867086034798973
#RMSE train: 0.06391324003979328
#RMSE test : 0.13998381038891858


# In[20]:


from yellowbrick.regressor import PredictionError

# Instantiate the linear model and visualizer
RandomForestRegressor = RandomForestRegressor()
visualizer = PredictionError(RandomForestRegressor)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[21]:


varImp = pd.DataFrame({'index':X_train.columns, 'feature_importance':RandomForestRegressor.feature_importances_})
varImp.sort_values(by='feature_importance', ascending=False, inplace=True)
f, ax = plt.subplots(1, 1, figsize=[12, 9])
sns.barplot(x = 'feature_importance', y = 'index', data = varImp.iloc[:30,], ax = ax)


# In[23]:


model_df = pd.DataFrame({'Lasso':[0.1078, 0.1138, 0.8855],                         'Ridge':[0.1091, 0.1101, 0.8870],                        'ElasticNet':[0.1125, 0.1108, 0.8874],                        'GBT':[0.1044, 0.1101, 0.8918],                        'LGB':[0.0990, 0.1163, 0.8913],                        'RF':[0.0639, 0.1399, 0.8670],                        'NN':[0.1189, 0.1220, 0.8670]},                        columns=['Lasso', 'Ridge','ElasticNet','GBT', 'LGB','RF','NN'],                        index=['RMSE Training', 'RMSE Testing', 'Score'])
model_df


# In[39]:


# The best single model selection (accuracy): gradient boosting regressor
model_gb = GradientBoostingRegressor(n_estimators = 1440,max_depth = 3, max_features = 'sqrt',
                                     min_samples_leaf = 19, min_samples_split =14,
                                     subsample= 0.78,random_state=rs_const)


model_gb.fit(X_train,y_train)
y_pred_gb = model_gb.predict(X_train)
y_pred_gb_test = model_gb.predict(X_test)
y_pred_gb_test = np.exp(y_pred_gb_test)


#Generate submission for single best model (gradient boosting regressor)
#generate_submission(gb, X_test)


# ### Ensembling all the Regressor

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_lr = linear_model.LinearRegression(n_jobs = -1)\nmodel_stack = StackingRegressor(regressors=[model_rf, model_gbt, model_nn, model_enet, model_lasso,model_ridge, model_lgb], meta_regressor=model_lr)\n\n# Fit the model on our data\nmodel_stack.fit(X_train, y_train)\ny_pred_stack = model_stack.predict(X_train)\nscore_stack = rmse(y_train, y_pred_stack)\n\nprint("StackingRegressor Score(On Training DataSet) : ",score_stack)')


# In[ ]:


#Generate submission for ensemble (gradient boosting regressor)
#generate_submission(model_stack, X_test)

