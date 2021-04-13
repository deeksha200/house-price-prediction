#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GridSearchCV
from sklearn.metrics import mean_absolute_error
pd.pandas.set_option("display.max_columns",None)
print("all necessary libraries are imported")


# In[3]:


train=pd.read_csv('C:\\Users\\Deeksha Rai\\Desktop\\projects\\train.csv')
test=pd.read_csv('C:\\Users\\Deeksha Rai\\Desktop\\projects\\test.csv')


# In[4]:


train.shape,test.shape


# In[5]:


test.tail()


# In[6]:


y=train['SalePrice'].values
train.drop('SalePrice',axis=1,inplace=True)
train.head()


# In[7]:


train.shape


# In[8]:


# checking null in training data


# In[9]:


col_train=list(train.columns)


# In[10]:


for feature in col_train:
    if(train[feature].isnull().any()):
        print(f'{feature} : {train[feature].isnull().sum()}')
    else:
         print(f'{feature} : {0}')
        


# In[11]:


# checking null in testing data
col_test=list(test.columns)
for feature in col_test:
    if(test[feature].isnull().any()):
        print(f'{feature} : {test[feature].isnull().sum()}')
    else:
         print(f'{feature} : {0}')


# In[12]:


# concatenating the train and test data to remove null values
data=pd.concat([train,test],axis=0,ignore_index=True)
data.tail()


# In[13]:


# checking null values in test+train data
col_train_test=list(data.columns)
for feature in col_train_test:
    if(data[feature].isnull().any()):
        print(f'{feature} : {data[feature].isnull().sum()}')
    


# In[14]:


col_num_feature=data.select_dtypes(exclude='object')
col_num_feature=col_num_feature.columns
col_num_feature=list(col_num_feature)
col_num_feature


# In[15]:


col_obj_feature=data.select_dtypes(include='object')
col_obj_feature=col_obj_feature.columns
col_obj_feature


# In[16]:


num_nan_feature=[]
for feature in col_num_feature:
    if(data[feature].isnull().any()):
        num_nan_feature.append(feature)
        


# In[17]:


num_nan_feature


# In[18]:


# replace all the null values with 0
for feature in num_nan_feature:
    data[feature].fillna(value=0,inplace=True)


# In[19]:


obj_nan_feature=[]
for feature in col_obj_feature:
    if(data[feature].isnull().any()):
        obj_nan_feature.append(feature)


# In[20]:


obj_nan_feature


# In[21]:


# replace all the null values with 'missing'
for feature in obj_nan_feature:
    data[feature].fillna(value='missing',inplace=True)


# In[22]:


dummy=pd.get_dummies(data[col_obj_feature],prefix=col_obj_feature)
dummy


# In[23]:


data.drop(col_obj_feature,axis=1,inplace=True)


# In[24]:


data_final=pd.concat([data,dummy],axis=1)


# In[25]:


data_final.drop(['Id'],axis=1,inplace=True)
print(data_final.head())


# In[26]:


# taking logarithm to remove skewness of data


# In[27]:


# for feature in data_final.columns:
#     data_final[feature].hist(bins=30)
#     plt.show()


# In[28]:


numeric_features=['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
          'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
          '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
          'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
          'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
          'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']


# In[29]:


# col_num_feature.remove('Id')
for feature in numeric_features :
    data_final[feature]=np.log(data_final[feature]+1)
print(data_final.head())
print(y.shape)
y=np.log(y+1)
print(y)


# In[30]:


data_final.shape


# In[31]:


for i in range(len(data_final.iloc[0,:])):
    p=[i+1,data_final.iloc[:,i].isnull().sum()]
    print(p)


# In[ ]:





# In[32]:


# removing correlated features by one method
cor_matrix = data_final.corr().abs()
# print(cor_matrix)
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
# print(upper_tri)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(); print(to_drop)
df1 = data_final.drop(to_drop, axis=1)
df1.shape


# In[33]:


# removing correlated features by another method
def get_corr(da_ta,threshold):
    corr_col=set()
    corr_mat=da_ta.corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if abs(corr_mat.iloc[i,j])>threshold:
                col_name=corr_mat.columns[i]
                corr_col.add(col_name)
    return corr_col
corr_features=get_corr(data_final,0.95)
dataset_final=data_final.drop(labels=corr_features,axis=1)
print(dataset_final.shape)
tr=dataset_final.iloc[0:1460,:]
print(tr.shape)
te=dataset_final.iloc[1460:2919,:]


# In[34]:


scaling=StandardScaler()
data_s_tr = scaling.fit_transform(tr)
data_s_te=scaling.transform((te))
Data_s_tr=pd.DataFrame(data_s_tr)
Data_s_te=pd.DataFrame(data_s_te)


# In[35]:


Parameters1=[{'reg_lambda': [0.4,0.5], 'reg_alpha': [0.9,1], 'n_estimators': [650,700], 'min_child_weight': [2.5,3],
             'max_depth': [4,3], 'learning_rate': [0.03,0.02], 'gamma': [0.00001,0.0001], 'booster': ['dart']}
               ]
scores1 = ['neg_mean_squared_error']

reg1 = GridSearchCV(xgb.XGBRegressor(), Parameters1, scoring='neg_root_mean_squared_error', verbose=2, cv=5)
reg1.fit(Data_s_tr.iloc[:,:].values,y)


# print(reg1.best_params_)

y_pred11 = reg1.predict(Data_s_tr.iloc[:,:].values)
y_pred1 = np.exp(reg1.predict(Data_s_te.iloc[:,:].values)).round(2)
# print(mean_absolute_error(y_pred11,Y))

Parameters2=[{'num_leaves':[31,32], 'max_depth':[3,4], 'learning_rate':[0.1,0.2],
            'n_estimators':[500,400]}]
scores = ['neg_mean_squared_error']

reg2 = GridSearchCV(LGBMRegressor(), Parameters2, scoring='neg_root_mean_squared_error', verbose=2, cv=5
                         )
reg2.fit(Data_s_tr.iloc[:,:].values,y)


# print(reg2.best_params_)

y_pred12 = reg2.predict(Data_s_tr.iloc[:,:].values)
y_pred2 = np.exp(reg2.predict(Data_s_te.iloc[:,:].values)).round(2)
# print(mean_absolute_error(y_pred12,Y))



Parameters3=[{'cache_size': [185,180],
              'tol': [0.0011,0.0013], 'kernel': ['rbf'],'gamma': [0.00009,0.0001],'epsilon': [0.011,0.013]
              }]
scores = ['neg_mean_squared_error']

reg3 = GridSearchCV(SVR(), Parameters3, scoring='neg_root_mean_squared_error', verbose=2, cv=5
                         )
reg3.fit(Data_s_tr.iloc[:,:].values,y)


# print(reg3.best_params_)

y_pred13 = reg3.predict(Data_s_tr.iloc[:,:].values)
y_pred3 = np.exp(reg3.predict(Data_s_te.iloc[:,:].values)).round(2)
# print(mean_absolute_error(y_pred13,Y))


# In[37]:


params_stack={
               'lgbmregressor__learning_rate': [0.01,0.02], 'lgbmregressor__max_depth': [3,4],
               'lgbmregressor__n_estimators': [500,600], 'lgbmregressor__num_leaves': [4,3],

              'xgbregressor__max_depth': [4,6],
              'xgbregressor__reg_lambda': [0.4,0.3],
              'xgbregressor__reg_alpha': [0.9],
              'xgbregressor__n_estimators': [500],
              'xgbregressor__min_child_weight': [2.5],
              'xgbregressor__learning_rate': [0.01,0.03],
              'xgbregressor__gamma': [0.00001],
              'xgbregressor__booster': ['dart'], 'svr__C': [75],
              'svr__cache_size': [185],
              'svr__tol': [0.0011], 'svr__kernel': ['rbf'],'svr__gamma': [0.00009],'svr__epsilon': [0.011],
              'svr__degree': [4],
              'meta_regressor__C': [75],
              'meta_regressor__cache_size': [185],
              'meta_regressor__tol': [0.0011],
              'meta_regressor__kernel': ['rbf'],
              'meta_regressor__gamma': [0.00009],
              'meta_regressor__epsilon': [0.011],
              'meta_regressor__degree': [4]
        }
xg_boost=xgb.XGBRegressor()
s_vr=SVR()
lgbm=LGBMRegressor()

regs=[xg_boost,s_vr,lgbm]

stack_reg=StackingRegressor(regressors=regs, meta_regressor=s_vr)

stack_gen=GridSearchCV(stack_reg,params_stack,cv=5,refit=True,verbose=2)
stack_gen.fit(Data_s_tr.iloc[:,:].values,y)
y_pred14=stack_gen.predict(Data_s_tr.iloc[:,:].values)
y_pred4=np.exp(stack_gen.predict(Data_s_te.iloc[:,:].values)).round(2)
# print(stack_gen.best_params_)
# print(mean_absolute_error(y_pred14,Y))


# In[42]:


y_pred4


# In[43]:


df_submission=pd.DataFrame({'Id':test['Id'],'Saleprice':y_pred4})


# In[44]:


df_submission


# In[45]:


df_submission.to_csv('submission_new.csv',index=False)

