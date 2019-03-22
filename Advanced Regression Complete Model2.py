#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Let us load all the necessary libraries.
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


# In[2]:



#and write a function get_data() to provide us with a dataset to work through this recipe:

#Let us write the following three functions.The function build _model, which implements the Gradient Boosting routine.
#The functions view_model and model_worth, which are used to inspect the model that we have built:

#

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


# In[5]:


#Finally, we will write our main function, which is used to invoke all the preceding functions:
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


# In[6]:




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

    


# In[7]:


# #########################################################################################
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


# In[ ]:


#We proceed to divide the data into the train and test sets using the train_test_split function from Scikit library. 
#We reserve 30 percent of our dataset for testing.


# In[8]:


df = combined[:n_train]
df_test = combined[n_train:]

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.10,
                                  stratify=stratify_col,shuffle = True,random_state=20)

stratify_X_train = stratify_col[:X_train.shape[0]].copy()
X_train.shape,X_test.shape,y_train.shape,y_test.shape, stratify_X_train.shape


# In[9]:


X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.10,
                                  stratify=stratify_X_train,shuffle = True,random_state=20)
X_train.shape,X_valid.shape,y_train.shape,y_valid.shape


# In[10]:


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


# Let us proceed to build our model:


# #### Random Forest

# In[ ]:


# First we instantiate and train the  random forest
#We will apply RandomizedSearchCV to identify an optimal set of parameters regarding an ensemble of regression trees:
#rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,y_train)


# In[11]:


#Fit the model with the training data
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,y_train)


# In[13]:


##set up a parameter distribution for the grid search.
#We will use 5-fold cross-validation and populate the parameter 
#grid with values for the key configuration settings:
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
params = {'n_estimators':[300,500,800,1100,1500,1800],
              "max_features": randint(80,680),
              "min_samples_split": randint(2, 11),
              "min_samples_leaf": randint(1, 11)
         }


# In[14]:


# Configure the grid search
randomSearch_rf = RandomizedSearchCV(RandomForestRegressor(warm_start=True),
                                     param_distributions=params,cv=kfold,
                                     n_jobs=6, n_iter=20)  


# In[16]:


#Train the multiple ensemble models defined by the parameter grid:
#grid_fit = randomSearch_rf.fit(X_train, y_train)


rf_bst = randomSearch_rf.fit(X_train,y_train)
#print_mse(rs_gbt_bst, X_train,X_valid,y_train,y_valid)


# In[ ]:


#rf_opt = grid_fit.best_estimator_


# In[17]:


# Display the results of the RandomizedSearchCV of the random forest as follows:
report(randomSearch_rf.cv_results_)


# In[18]:


print_mse(randomSearch_rf, X_train,X_valid,y_train,y_valid)


# In[20]:


randomSearch_rf_predict = rf_bst.predict(X_test)


# In[ ]:


#rf_bst_preds = rf_bst.predict(X_test)


# In[ ]:





# In[ ]:


# fit the model
#randomSearch_rf.fit(X_train, y_train)
#y_predict = randomSearch_rf.predict(X_test)


# In[ ]:


#optrf_r2 = r2_score(y_test, rf_opt_preds)
#optrf_mse = mean_squared_error(y_test, rf_opt_preds)


# In[ ]:


#y_pred = randomSearch_rf.predict(X_test)
#print ("R-squared",r2_score(y_test, y_pred))
#print ("MSE : ",mean_squared_error(y_test, y_pred))


# In[ ]:


# Get the estimator best parameters as:
#grid_fit.best_params_


# In[ ]:


#Finally, measure the performance on a test set. The algorithm does not perform as well as others, but we
#can possibly use it as part of a stacking aggregator later:


# ##### Gradient Boosting

# In[ ]:


### Initializing and Fitting Model
# Bullet: We initialize the model and fit it on the Train dataset.


# In[ ]:


#GBR = GradientBoostingRegressor()
#GBR.fit(X_train,Y_train)


# In[ ]:


### Predict and Check Accuracy
# The above model is used to predict the values of the dependent variable in the Test dataset and check its accuracy.


# In[ ]:


#GBR_test_pred = GBR.predict(X_test)
#metrics.r2_score(Y_test, GBR_test_pred)

#GBR_test_pred = GBR.predict(X_test)
#metrics.r2_score(Y_test, GBR_test_pred)


# In[ ]:


## Bullet Tuning Hyperparameters
#Here we tune 6 parameters: max_depth, max_features, min_samples_split, 
#min_samples_leaf, learning_rate and n_estimators.


# In[ ]:





# In[ ]:


#Create a parameter distribution for the gradient boosting trees:


# In[ ]:





# In[21]:


param_dist1 = {'max_features' : ['sqrt'],
 'max_depth' : [3, 5, 7, 10],
 'min_samples_leaf' : [5, 10,15, 20],
 'min_samples_split': [2, 5, 10, 15,20],    # 10      
 'n_estimators': [1000,1200, 1500],
 'learning_rate' : [0.0001,0.001,0.01,0.05,0.1,0.3],
 'loss' : ['ls','huber'],
 'subsample':[0.6,0.7,0.75,0.8,0.9],
 'random_state': [5]
 }


# In[ ]:


#Now let's run the grid search to find the best parameters with 30 iterations:


# In[22]:


rs_gb1 = RandomizedSearchCV(GradientBoostingRegressor(warm_start=True),
 param_distributions = param_dist1,
 cv=5,
 n_iter = 30, n_jobs=-1)
rs_gb1.fit(X_train, y_train)   # cv =4


# In[23]:


# Create a function to view the report have been wrapped so that they can be used more times:
import numpy as np
import pandas as pd

def get_grid_df(fitted_gs_estimator):
    res_dict = fitted_gs_estimator.cv_results_
 
    results_df = pd.DataFrame()
    for key in res_dict.keys():
         results_df[key] = res_dict[key]
 
    return results_df

def group_report(results_df):
      param_cols = [x for x in results_df.columns if 'param' in x and x is not 'params']
      focus_cols = param_cols + ['mean_test_score']
 
      print ("Grid CV Report \n")
 
      output_df = pd.DataFrame(columns = ['param_type','param_set',
 'mean_score','mean_std'])
      cc = 0
      for param in param_cols:
         for key,group in results_df.groupby(param):
              output_df.loc[cc] = (param, key, group['mean_test_score'].mean(), group['mean_test_score'].std())
              cc += 1
      return output_df


# In[24]:


results_df = get_grid_df(rs_gb1)
group_report(results_df)


# In[25]:


param_dist2 = {'max_features' : ['sqrt'],
 'max_depth' : [2,3],  # 2 and 3 . The best being 3. Before: [5, 6, 7],
 'min_samples_leaf' : [19,20], # 19 and 20. The best is 20
 'min_samples_split': [14,15],   # right   
 'n_estimators': [1400, 1500],
 'learning_rate' : [0.01,0.02,0.05], # right
 'loss' : ['huber'],
 'subsample':[0.78,0.8],   #  use #0.78, 0.8. The best 0.8 ... old [0.7,0.72,0.75],
 'random_state': [5]
 }

rs_gb2 = RandomizedSearchCV(GradientBoostingRegressor(warm_start=True),
param_distributions = param_dist2,
cv=5,
n_iter = 30, n_jobs=-1)
rs_gb2.fit(X_train, y_train)    #cv = 3


# In[26]:


#Display the dataframe to show how gradient boosting trees performed with various parameter settings:

results_df = get_grid_df(rs_gb2)
group_report(results_df)


# In[27]:


param_dist3 = {'max_features': 'sqrt',
 'max_depth': 3,     # max depth 5 better
 'min_samples_leaf': 19, 
 'min_samples_split': 14,  # best is 14     
 'n_estimators': 1400, #1500 is the best estimator
 'learning_rate': 0.02,
 'loss' : 'huber',
 'subsample': 0.78,
 'random_state':  5
 }


# In[28]:


rs_gbt_bst = GradientBoostingRegressor(warm_start=True,
 max_features = 'sqrt',
 max_depth = 3,
 min_samples_leaf = 19,
 min_samples_split= 14,       
 n_estimators = 1400,
 learning_rate = 0.02,
 loss = 'huber',
 subsample = 0.78,
 random_state =  5)
 

rs_gbt_bst.fit(X_train,y_train)
print_mse(rs_gbt_bst, X_train,X_valid,y_train,y_valid)


# In[29]:


rs_gbt_bst_predict = rs_gbt_bst.predict(X_test) 


# In[ ]:





# ### Light Gradient Boost Algorithm

# In[30]:


# Create parameters to search


# In[31]:


###### 2222222222222
# Initiating LGBMRegressor model ##### THIS IS THE BEST SO FAR
#Set the range for each parameter
#Gentle reminder: try to make the range as narrow as possible
param_dist1 ={
                              'num_leaves':[4,6,8,10],
                              'learning_rate':[0.001, 0.005, 0.01, 0.05, 0.1], 
                              'n_estimators':[1300, 1500, 1800, 2000],
                              'max_bin': [60, 66, 76,100], 
                              'bagging_fraction': [0.5,0.6, 0.7, 0.8, 0.9],
                              'bagging_freq': [3,4, 5, 6, 7, 8], 
                              'feature_fraction': [0.2, 0.4, 0.6, 0.8],
                              'feature_fraction_seed':[2,4,9,10], 
                              'bagging_seed': [4,9,12],
                              'min_sum_hessian_in_leaf': [1,11,20],
                              'min_data_in_leaf':[4,6,8,10]
}

#'min_data_in_leaf':list(range(1,100,20)) 
#{'num_leaves': [4,34, 64, 94, 124, 154, 184],
#'min_data_in_leaf':list(range(1,100,20)) 


# Initiating LGBMRegressor model ##### THIS IS THE BEST SO FAR
#rs_gbt= lgb.LGBMRegressor(objective='regression',
                              #num_leaves=4,
                              #learning_rate=0.01, 
                              #n_estimators=1300,   #best at 1314
                              #max_bin=66, 
                              #bagging_fraction=0.76,
                              #bagging_freq=4, 
                              #feature_fraction=0.2,
                              #feature_fraction_seed=9, 
                              #bagging_seed=9,
                              #min_data_in_leaf=4, 
                              #min_sum_hessian_in_leaf=12).fit(X_train,y_train)


# In[32]:


# Create classifier to use. Note that parameters have to be input manually
# not as a dict!


# In[33]:


#33333333333
# Run the grid search to find the best parameters. Perform a randomized search with 30 iterations:
# Create the grid
rs_lgbm1 = RandomizedSearchCV(lgb.LGBMRegressor(warm_start=True),
 param_distributions = param_dist1,
 cv=5,
 n_iter = 30, n_jobs=-1)
#Run the grid
rs_lgbm1.fit(X_train, y_train)   # cv =4


# In[34]:



#4444444444444444444444444444444444444444444
# Now look at the report in dataframe form. The functions to view the report have been wrapped 
#so that they can be used more times:

# Create a function to view the report have been wrapped so that they can be used more times:
# Remove the skew
#
import numpy as np
import pandas as pd

def get_grid_df(fitted_gs_estimator):
    res_dict = fitted_gs_estimator.cv_results_
 
    results_df = pd.DataFrame()
    for key in res_dict.keys():
         results_df[key] = res_dict[key]
 
    return results_df

def group_report(results_df):
      param_cols = [x for x in results_df.columns if 'param' in x and x is not 'params']
      focus_cols = param_cols + ['mean_test_score']
 
      print ("Grid CV Report \n")
 
      output_df = pd.DataFrame(columns = ['param_type','param_set',
 'mean_score','mean_std'])
      cc = 0
      for param in param_cols:
         for key,group in results_df.groupby(param):
              output_df.loc[cc] = (param, key, group['mean_test_score'].mean(), group['mean_test_score'].std())
              cc += 1
      return output_df


# In[35]:



#5555555555555555555555555555555555
#View the dataframe that shows how gradient boosting trees performed with various parameter settings:
results_df = get_grid_df(rs_lgbm1)
group_report(results_df)


# In[36]:


param_dist2 ={'num_leaves': [4,8],
                              'learning_rate':[0.005,0.01,0.005], #,  try #0.006, 0.007,0.1
                              'n_estimators':[1300,1500],  #try 1400, 1500
                              'max_bin': [66,76],          # 66
                              'bagging_fraction': [0.6,0.7], # 0.8, 0.85, 0.9
                              'bagging_freq': [8],  #4
                              'feature_fraction': [0.6,0.7], # 0.8
                              'feature_fraction_seed':[12], 
                              'bagging_seed': [9],
                              'min_data_in_leaf': [21,22], # 19,20
                              'min_sum_hessian_in_leaf': [1,2] #19,20
}
    
# Create the grid
rs_lgbm2 = RandomizedSearchCV(lgb.LGBMRegressor(warm_start=True),
 param_distributions = param_dist2,
 cv=5,
 n_iter = 30, n_jobs=-1)
#Run the grid
rs_lgbm2.fit(X_train, y_train)   # cv =4


# In[37]:


results_df = get_grid_df(rs_lgbm2)
group_report(results_df)


# In[38]:


param_dist3 ={'num_leaves': [7,8],
                              'learning_rate':[0.006,0.01],  #0.006, 0.01,0.1
                              'n_estimators':[1500],  #1400, 1600,2000
                              'max_bin': [66],          # 66
                              'bagging_fraction': [0.2], # 0.8, 0.85, 0.9
                              'bagging_freq': [8],  #4
                              'feature_fraction': [0.2], # 0.8
                              'feature_fraction_seed':[2], 
                              'bagging_seed': [12],
                              'min_data_in_leaf': [5], # 19,20
                              'min_sum_hessian_in_leaf': [1] #19,20
}


# In[39]:


# Create the grid
rs_lgbm3 = lgb.LGBMRegressor(warm_start=True,
num_leaves = 8,
learning_rate = 0.006, 
n_estimators = 1493,
max_bin = 66, 
bagging_fraction = 0.2,
bagging_freq = 8, 
feature_fraction =  0.2,
feature_fraction_seed = 2, 
bagging_seed = 12,
min_data_in_leaf = 5, 
min_sum_hessian_in_leaf = 1)


rs_lgbm3.fit(X_train, y_train)
print_mse(rs_lgbm3, X_train,X_valid,y_train,y_valid)


# In[40]:


# LGBM with tuned parameters
rs_lgbm_bst = lgb.LGBMRegressor(warm_start=True,
num_leaves = 8,
learning_rate = 0.006, 
n_estimators = 1493,
max_bin = 66, 
bagging_fraction = 0.2,
bagging_freq = 8, 
feature_fraction =  0.2,
feature_fraction_seed = 2, 
bagging_seed = 12,
min_data_in_leaf = 5, 
min_sum_hessian_in_leaf = 1)


# In[41]:


# Predict and check accuracy
rs_lgbm_bst.fit(X_train, y_train)
rs_lgbm_bst_predict = rs_lgbm_bst.predict(X_test) #prediction for lgbm


# In[ ]:





# ### Elastic Net

# In[42]:


ElasticNet()


# In[43]:


kfold = KFold(n_splits=5, shuffle=True, random_state=0)


# In[45]:


params = {'alpha':[0.001,0.01,0.1,1.],
          'l1_ratio': [0.4,0.5,0.6,0.7,0.8,0.9],
          'max_iter':[1000,2000,5000,10000],
          'selection':['cyclic','random']
         }

randomSearch_elastic = RandomizedSearchCV(ElasticNet(warm_start=True),param_distributions=params,
                                          cv=kfold,n_jobs=6, n_iter=100)        
randomSearch_elastic.fit(X_train,y_train)
print_mse(randomSearch_elastic, X_train,X_valid,y_train,y_valid)


# In[47]:


report(randomSearch_elastic.cv_results_)


# In[48]:


# Elastic net with tuned parameters
randomSearch_elastic_bst = ElasticNet(alpha=0.001,
                                   selection='cyclic',
                                   max_iter=1000,
                                   l1_ratio=0.8
                                   ).fit(X_train,y_train)
print_mse(randomSearch_elastic_bst,X_train,X_valid,y_train,y_valid)


# In[49]:


#randomSearch_elastic_bst.fit(X_train, y_train)
randomSearch_elastic_bst_predict = randomSearch_elastic_bst.predict(X_test) #enet predictions


# In[ ]:





# In[ ]:





# ### Ensemble

# In[51]:


# Random Forest
#rf_r2 = r2_score(y_test, y_predict)
#rf_mse = mean_squared_error(y_test, y_predict)


rf_r2 = r2_score(y_test, randomSearch_rf_predict)
rf_mse = mean_squared_error(y_test, randomSearch_rf_predict)


# In[52]:


# Gradient boosting
rs_gbt_bst_r2 = r2_score(y_test, rs_gbt_bst_predict)
rs_gbt_bst_mse = mean_squared_error(y_test, rs_gbt_bst_predict)


# In[53]:


#Light gradient boosting
rs_lgbm_bst_r2 = r2_score(y_test, rs_lgbm_bst_predict)
rs_lgbm_bst_mse = mean_squared_error(y_test, rs_lgbm_bst_predict)


# In[54]:


# Enet
randomSearch_elastic_bst_r2 = r2_score(y_test, randomSearch_elastic_bst_predict)
enet_bst_mse = mean_squared_error(y_test, randomSearch_elastic_bst_predict)


# In[55]:


randomSearch_elastic_bst_r2


# In[56]:


enet_bst_mse


# In[ ]:





# In[ ]:


# Let's compare them using a dataframe
d = {'1. RF': [rf_r2, rf_mse], 
     '2. GB': [rs_gbt_bst_r2, rs_gbt_bst_mse], 
     '3. LGB': [rs_lgbm_bst_r2, rs_lgbm_bst_mse], 
     '4. Enet': [randomSearch_elastic_bst_r2, enet_bst_mse]}
d_i = ['R2', 'Mean Squared Error']
df_results = pd.DataFrame(data=d, index = d_i)
df_results


# In[ ]:


# Let's compare them using a dataframe
d = {'1. RF': [rf_mse], 
     '2. GB': [rs_gbt_bst_mse], 
     '3. LGB': [rs_lgbm_bst_mse], 
     '4. Enet': [enet_bst_mse]}
d_i = ['Mean Squared Error']
df_results = pd.DataFrame(data=d, index = d_i)
df_results


# In[ ]:


df_results = results.sort_values(by='Mean Squared Error', ascending=True).reset_index(drop=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from scipy.optimize import minimize


# In[ ]:


# finding the optimum weights
clfs = [randomSearch_elastic_bst,rs_gbt_bst] # let's focus on our top two performers: XGBoost (Boosting Method) & Lasso (Linear Regression method)
predictions = []
for clf in clfs:
    predictions.append(clf.predict(X_test)) # listing all our predictions


# In[ ]:


def mse_func(weights):
    # scipy minimize will pass the weights as a numpy array
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    return mean_squared_error(y_test, final_prediction)


# In[ ]:


starting_values = [0.5]*len(predictions) # minimize need a starting value
bounds = [(0,1)]*len(predictions) # weights are bound between 0 and 1
res = minimize(mse_func, 
               starting_values,
               bounds = bounds, 
               method='SLSQP'
              )
print('Result Assessment: {message_algo}'.format(message_algo = res['message']))
print('Ensemble Score: {best_score}'.format(best_score = res['fun']))
print('Best Weights: {weights}'.format(weights = res['x']))


# In[ ]:


# these are the weights that minimize MSE for our stacked model
randomSearch_elastic_bst_weight = res['x'][0]
rs_gbt_bst_weight = res['x'][1]


# In[ ]:


# Get the predictions for df_test from each model
#y_pred_xgb = xgb_opt.predict(df_test)
#y_pred_lasso = lasso_opt.predict(df_test)
# Blend the results of the three regressors using our model weights
#y_pred = (xgb_opt_weight*y_pred_xgb + lasso_opt_weight*y_pred_lasso)
# Lets not forget to apply the exponential functions to our results as we applied log earlier in our data prep
#y_pred_final = np.exp(y_pred)


# In[ ]:





# In[57]:


# Creating a table of results, ranked highest to lowest
results = pd.DataFrame({
    'Model': ['Random Forest', 
              'Gradient Boosting Regressor',
              'Light Gradient Boosting Regressor',
              'Enet'],
    'MSE': [rf_mse,
              rs_gbt_bst_mse,
              rs_lgbm_bst_mse,
              enet_bst_mse]})

# Build dataframe of values
result_df = results.sort_values(by='MSE', ascending=True).reset_index(drop=True)
result_df.head(8)


# In[58]:


# Creating a table of results, ranked highest to lowest
results = pd.DataFrame({
    'Model': ['ElasticNet', 
              'Gradient Boosting Regressor',
              'Light Gradient Boosting Regressor',
              'Random Forest'],
    'MSE': [enet_bst_mse,
              rs_gbt_bst_mse,
              rs_lgbm_bst_mse,
              rf_mse]})

# Build dataframe of values
result_df = results.sort_values(by='MSE', ascending=True).reset_index(drop=True)
result_df.head(8)


# In[59]:


# Plotting model performance
f, ax = plt.subplots(figsize=(10, 6))
plt.xticks(rotation='90')
sns.barplot(x=result_df['Model'], y=result_df['MSE'])
plt.xlabel('Models', fontsize=15)
plt.ylabel('Model performance', fontsize=15)
plt.ylim(0.0122, 0.02)
plt.title('MSE', fontsize=15)


# In[60]:


#mod1 = RandomForestRegressor()
mod1 = GradientBoostingRegressor()
mod2 = lgb.LGBMRegressor()
mod3 = ElasticNet()
rf = RandomForestRegressor()


# In[61]:


#mod1 = RandomForestRegressor()
#mod1 = GradientBoostingRegressor()
mod1 = lgb.LGBMRegressor()
mod2 = ElasticNet()
GB = GradientBoostingRegressor()


# In[62]:


sr = StackingRegressor(regressors=[mod1, mod2,GB], 
                          meta_regressor=GB)


# In[63]:


sr.fit(X_train,y_train)


# In[64]:


sr_pred = sr.predict(X_test)
metrics.r2_score(y_test,sr_pred)
metrics.mean_squared_error(y_test,sr_pred)


# In[65]:


# Create stacked model
stacked = (randomSearch_elastic_bst_predict + rs_lgbm_bst_predict + rs_gbt_bst_predict + randomSearch_rf_predict) / 4


# In[66]:


# Setting up competition submission
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = stacked
sub.to_csv('house_price_predictions.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:


from mlxtend.regressor import StackingRegressor
from mlxtend.data import boston_housing_data
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


# In[ ]:


# Setting up competition submission
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = stacked
sub.to_csv('house_price_predictions.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:


#We creat a function for calculating the mean_squared_erros
from sklearn.metrics import mean_squared_error

def mse (true_data, predicted_data):
    return np.sqrt(mean_squared_error(true_data, predicted_data))


# In[ ]:





# In[ ]:


randfr = RandomForestRegressor(random_state = 42) #random_state to avoid the result from fluctuating


# In[ ]:


param_grid = { 
    'n_estimators': [50,250,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2, 4, 6, 8, 10],
}


# In[ ]:


randfr_cv = GridSearchCV(estimator=randfr, param_grid=param_grid, cv= 5)  #cv = 5 to specify the number of folds(5 in this case)  in a stratified Kfold
randfr = randfr_cv.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:


######## TRY ANOTHER KIND OF ESEMBLE OR STACKING   SYRINGE


# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X_train, y_train):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X_train, y_train)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X_train):
        predictions = np.column_stack([
            model.predict(X_train) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[ ]:





# In[ ]:


score = mse_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# In[ ]:





# In[ ]:





# In[ ]:


averaged_models = AveragingModels(models = (ElasticNet, GradientBoostingRegressor,RandomForestRegressor))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:





# In[ ]:





# In[ ]:


#putting it lall together

score_vals = [LR_score, R_score, L_score, EN_score, RF_score]
rmse_vals = [LR_rmse, R_rmse, L_rmse, EN_rmse, RF_rmse]
labels = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest']

rmse_df = pd.Series(score_vals, index=labels).to_frame()
rmse_df.rename(columns={0: 'SCORES'}, inplace=1)
rmse_df['RMSE'] = rmse_vals
rmse_df


# In[ ]:


#Run this cell, to do HP optimisation. To save time opt_parameters was directly hardcoded below.


# In[ ]:





# In[ ]:


# Using parameters already set above, replace in the best from the grid search


# In[ ]:





# In[ ]:


# Plot importance
    #lgb.plot_importance(gbm)
    #plt.show()


# In[ ]:


print(opt_params)


# In[ ]:


print(opt_params)


# In[ ]:


#Let us proceed to check performance of the ensemble on the training data:


# In[ ]:


#As you can see, our boosting ensemble has fit the training data perfectly.

#The model_worth function prints some more details of the model. They are as follows:


# In[ ]:


Finally, we can also see the importance associated with each feature:

#print "\n Feature Importance"
   # print "======================\n"
    #for i,score in enumerate(model.feature_importances_):
        #print "\tFeature %d Importance %0.3f"%(i+1,score)


# In[ ]:


#Let us see how the features are stacked against each other.

