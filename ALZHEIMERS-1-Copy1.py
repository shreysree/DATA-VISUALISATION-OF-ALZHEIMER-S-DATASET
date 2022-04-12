#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

sns.set(style="darkgrid")

import os

alz = pd.read_csv(r"d:\Desktop\oasis_longitudinal.csv")
# Dataset is now stored in a Pandas Dataframe
alz.head()#first 5 rows


# In[2]:


alz.describe().loc[['count','min','max']]


# In[3]:


alz=alz.drop(['Subject ID','MRI ID','Hand'],axis=1)#axis =1 means columns are going to be removed
alz.head()
alz.isna().sum() #detecting null values


# In[4]:


alz.head(3)


# In[5]:


def univariate_mul(var):
    #fig =
    plt.figure(figsize=(20,10))
    #cmap=plt.cm.Reds
    #cmap1=plt.cm.coolwarm_r
    #ax1 = fig.add_subplot(221)
    #ax2 = fig.add_subplot(212)
    ax1=plt.subplot(221)
    ax2=plt.subplot(222)
    alz[var].plot(kind='hist',ax=ax1, grid=False,color = "crimson", ec="green")
    ax1.set_title('Histogram of '+var, fontsize=14)
    
    ax2=sns.distplot(alz[[var]],hist=False,color = "crimson")
    ax2.set_title('Distribution of '+ var)
    plt.show()
# lets see the distribution of SES to decide which value we can impute in place of missing values.
univariate_mul('SES')


# In[6]:


alz['SES'].median() #the median of the column SES


# In[7]:


# imputing missing value in SES with median
alz['SES'].fillna((alz['SES'].median()), inplace=True)
alz.isna().sum() #detecting null values


# In[8]:


univariate_mul('MMSE')


# In[9]:


alz['MMSE'].median() #the median of the column MMSE


# In[10]:


# imputing missing value in SES with median
alz['MMSE'].fillna((alz['MMSE'].median()), inplace=True)
alz.isna().sum() #detecting null values


# In[11]:


# Defining function to create pie chart and bar plot as subplots
def plot_piechart(var):
  plt.figure(figsize=(20,10))
  plt.subplot(221)
  label_list = alz[var].unique().tolist()
  print(label_list)
  alz[var].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("colorblind",7),labels=label_list,
  shadow =False)
  plt.title("Distribution of "+ var +"  variable")

  plt.subplot(222)
  ax = alz[var].value_counts().plot(kind="barh",color="tomato")

  for i,j in enumerate(alz[var].value_counts().values):
    ax.text(.7,i,j,weight = "bold",fontsize=20)

  plt.title("Count of "+ var +" cases")
  plt.show()
plot_piechart('Group')


# In[12]:


# Categorizing feature CDR
def cat_CDR(n):
    if n == 0:
        return 'Normal'
    
    else:                                         # As we have no cases of sever dementia CDR score=3
        return 'Dementia'

alz['CDR'] = alz['CDR'].apply(lambda x: cat_CDR(x))
plot_piechart('CDR')

# Plotting CDR with other variable
def univariate_percent_plot(cat):
    fig = plt.figure(figsize=(18,12))
    cmap=plt.cm.Blues
    cmap1=plt.cm.coolwarm_r
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    
    result = alz.groupby(cat).apply (lambda group: (group.CDR == 'Normal').sum() / float(group.CDR.count())
         ).to_frame('Normal')
    result['Dementia'] = 1 -result.Normal
    result.plot(kind='bar', stacked = True,colormap=cmap1, ax=ax1, grid=True)
    ax1.set_title('stacked Bar Plot of '+ cat +' (in %)', fontsize=14)
    ax1.set_ylabel('% Dementia status (Normal vs Dementia)')
    ax1.legend(loc="lower right")
    group_by_stat = alz.groupby([cat, 'CDR']).size()
    group_by_stat.unstack().plot(kind='bar', stacked=True,ax=ax2,grid=True)
    ax2.set_title('stacked Bar Plot of '+ cat +' (in %)', fontsize=14)
    ax2.set_ylabel('Number of Cases')
    plt.show()


# In[13]:


plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="M/F", y="EDUC",hue="CDR",split=True, data=alz)
plt.show()


# In[14]:


def cat_CDR(n):
    if n == 'Nondemented':
        return 'Nondemented'
    
    else:                                         # converting 'converted' to 'demented'
        return 'Demented'

alz['Group'] = alz['Group'].apply(lambda x: cat_CDR(x))


# In[15]:


plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="M/F", y="EDUC",hue="Group",split=True, data=alz,palette="Blues")
plt.show()


# In[16]:


#CDR Vs MMSE BAR PLOT
sns.barplot(data=alz,x='CDR',y='MMSE',palette='Greens')


# In[17]:


# Categorizing feature MMSE
def cat_MMSE(n):
    if n >= 24:
        return 'Normal'
    elif n <= 9:
        return 'Severe'
    elif n >= 10 and n <= 18:
        return 'Moderate'
    elif n >= 19 and n <= 23:                                        # As we have no cases of sever dementia CDR score=3
        return 'Mild'

alz['MMSE'] = alz['MMSE'].apply(lambda x: cat_MMSE(x))


# In[18]:


plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="MMSE", y="Visit",split=True, data=alz)
plt.show()


# In[19]:


plot_piechart('MMSE')


# In[20]:


print(alz['SES'].value_counts())


# In[21]:


print(alz['SES'].unique().tolist())


# In[22]:


ses_count = alz['SES'].value_counts()
ses_indexes = list(ses_count.index)
print(ses_indexes)


# In[23]:


alz.hist(figsize=(8,6),layout=(2,4))


# In[24]:


sns.pairplot(data=alz)


# In[25]:


plt.figure(figsize=(6,5))
sns.countplot(alz['Group'])
plt.title('Distribution of CDR Levels')
plt.xlabel('CDR LEVEL')
plt.ylabel('COUNT')
plt.savefig('CDR_distribution.png')


# In[26]:


alz["Group"].replace({"Demented": 1, "NonDemented": 0}, inplace=True)


# In[27]:


alz["M/F"].replace({"M": 1, "F": 0}, inplace=True)


# In[28]:


ax=sns.barplot(data=alz,x='M/F',y='Age',hue='Group',palette='Blues_d')
#for i,j in enumerate(alz['M/F'].value_counts().values):
    #ax.text(.7,i,j,weight = "bold",fontsize=20)


# In[29]:


alz.shape


# In[30]:


#SES Vs CDR
sns.barplot(data=alz,x='CDR',y='SES',palette='Greens')


# In[31]:


#CDR VS eTIV
sns.catplot(data=alz,x='CDR',y='eTIV',hue='M/F')


# In[32]:


#CDR Vs ASF
sns.scatterplot(data=alz,x='CDR',y='ASF',hue='M/F',palette='winter')


# In[33]:


#nWBV VS CDR
plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="CDR", y="nWBV",split=True, data=alz,palette="Oranges")
plt.title("CDR Vs nWBV")
plt.show()


# In[34]:


alz.shape


# In[ ]:




