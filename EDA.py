#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load the Python libraries
import operator
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import re
import scipy
from scipy.stats import norm, skew, probplot


# In[3]:


pd.set_option('display.max_columns', 85)
pd.set_option('display.max_rows', 85)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Load the data and have a quick look
house = pd.read_csv('train.csv', index_col='Id')
house_test = pd.read_csv('test.csv', index_col='Id')

house.SalePrice = house.SalePrice.apply(np.log)
print(house.shape)
print(house_test.shape)


# In[5]:


house.describe()


# In[6]:


house.SalePrice.describe()


# In[10]:


# examining the target variable
mu, sigma = norm.fit(house['SalePrice'])
sns.distplot(house['SalePrice'], fit=norm)
plt.legend(
    ['Normal distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Histogram of SalePrice')

fig = plt.figure()
res = probplot(house['SalePrice'], plot=plt)


# In[104]:


# See which features have missing values
missing = house.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
missing


# In[105]:


# Numerical data
quantitative = [f for f in house.columns if house.dtypes[f] != 'object']
qualitative = [f for f in house.columns if house.dtypes[f] == 'object']


# In[106]:


quantitative


# In[107]:


f = pd.melt(house, value_vars=quantitative)
g = sns.FacetGrid(f, col='variable', col_wrap=6, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")


# In[108]:


quantitative2 = house.select_dtypes(include=['float64', 'int64'])
quantitative2.head()


# In[109]:


quantitative2.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# In[110]:


features = quantitative

standard = house[house['SalePrice'] < 12]
pricey = house[house['SalePrice'] >= 12]

diff = pd.DataFrame()
diff['feature'] = features
diff['difference'] = [(pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean())/(standard[f].fillna(0.).mean())
                      for f in features]

sns.barplot(data=diff, x='feature', y='difference')
x = plt.xticks(rotation=90)


# In[112]:


# categorical data

for c in qualitative:
    house[c] = house[c].astype('category')
    if house[c].isnull().any():
        house[c] = house[c].cat.add_categories(['MISSING'])
        house[c] = house[c].fillna('MISSING')


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)


f = pd.melt(house, id_vars=['SalePrice'], value_vars=qualitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")


# In[113]:


categorical_features = [a for a in quantitative[:-1] + house.columns.tolist()
                        if (a not in quantitative[:-1]) or (a not in house.columns.tolist())]
df_categ = house[categorical_features]
df_categ.head()


# In[114]:


df_not_num = df_categ.select_dtypes(include=['O'])


# In[115]:


fig, axes = plt.subplots(round(len(df_not_num.columns) / 3), 3, figsize=(12, 30))


# In[116]:


corr = house.corr()
corr.describe()


# In[118]:


corrmat = house.corr()
plt.figure(figsize=(8, 6))
k = 10
cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index
cm = np.corrcoef(house[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[119]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[120]:


plt.figure(1)
f, axarr = plt.subplots(3, 2, figsize=(10, 9))
price = house.SalePrice.values
axarr[0, 0].scatter(house.GrLivArea.values, price)
axarr[0, 0].set_title('GrLiveArea')
axarr[0, 1].scatter(house.GarageArea.values, price)
axarr[0, 1].set_title('GarageArea')
axarr[1, 0].scatter(house.TotalBsmtSF.values, price)
axarr[1, 0].set_title('TotalBsmtSF')
axarr[1, 1].scatter(house['1stFlrSF'].values, price)
axarr[1, 1].set_title('1stFlrSF')
axarr[2, 0].scatter(house.TotRmsAbvGrd.values, price)
axarr[2, 0].set_title('TotRmsAbvGrd')
axarr[2, 1].scatter(house.MasVnrArea.values, price)
axarr[2, 1].set_title('MasVnrArea')
f.text(-0.01, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize=12)
plt.tight_layout()
plt.show()


# In[121]:


ord_cols = ['ExterQual', 'ExterCond', 'BsmtCond', 'HeatingQC', 'KitchenQual',
            'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
ord_dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}


# In[122]:


ord_df = house.copy()

for col in ord_cols:
    ord_df[col] = ord_df[col].map(lambda x: ord_dic.get(x, 0))
ord_df.head()


# In[123]:


individual_features_df = []
for i in range(0, len(house.columns) - 1):  # -1 because the last column is SalePrice
    tmpDf = house[[house.columns[i], 'SalePrice']]
    tmpDf = tmpDf[tmpDf[house.columns[i]] != 0]
    individual_features_df.append(tmpDf)

all_correlations = {feature.columns[0]: feature.corr()['SalePrice'][0]
                    for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))


# In[124]:


golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5]
print("There is {} strongly correlated values with SalePrice:\n{}".format(
    len(golden_features_list), golden_features_list))


# In[ ]:


# differentiate the NA and missing in train data
miss = list(house.columns[house.isnull().sum(axis=0) >= 1])
miss.remove('LotFrontage')
miss.remove('MasVnrArea')
miss.remove('GarageYrBlt')
miss.remove('MasVnrType')

for i in range(len(miss)):
    house[miss[i]][house[miss[i]].isnull() == True] = 'NA'
house['MasVnrType'][house['MasVnrType'].isnull() == True] = 'CBlock'
