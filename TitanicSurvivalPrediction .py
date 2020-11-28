#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing needed libraries
import pandas as pd #data processing
import numpy as np #numerical analysis
import seaborn as sns #data visualization

#matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style

#Algorithms
from sklearn import linear_model
from sklearn.linear_model import LinearRegression #Linear Regression
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.ensemble import RandomForestClassifier #Randon Forest Classifier
from sklearn.model_selection import GridSearchCV #parameter tuning
from sklearn.svm import SVC, LinearSVC #Linear Support Vector
from sklearn.naive_bayes import GaussianNB #Naive Bayes'


# In[2]:


#read the csv datasets
train_df = pd.read_csv("titanic/train.csv") #csv file location
test_df = pd.read_csv("titanic/test.csv")


# In[3]:


#backup the datasets
backup_df = train_df
tbackup_df = test_df


# # Exploratory Data Analysis

# In[4]:


train_df.head()


# In[5]:


train_df.shape


# In[6]:


train_df.info()


# In[7]:


train_df.dtypes


# In[8]:


train_df.columns.values


# ##### From the 11 columns + target variable Survived, we can correlate everything with a high survival rate except 'PassengerId', 'Name', 'Ticket'

# In[9]:


train_df.describe()


# In[10]:


#checking missing values
train_df.isna().sum()


# In[11]:


#three columns has missing values. 
#find the percentage of missing values and create a dataframe.
total = train_df.isna().sum().sort_values(ascending=False)
percent = train_df.isna().sum()/train_df.isna().count()*100
rd_percent = (round(percent, 2)).sort_values(ascending=False)
missing_data = pd.concat([total, rd_percent], axis=1, keys=['Total','%'])
missing_data.head(3)


# ### Age and Sex:

# In[12]:


survived = 'Survived'
not_survived = 'Not_Survived'

#plot the histogram for gender feature according to the survival data
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']

ag = sns.histplot(women[women['Survived']==1].Age.dropna(), bins=18, label=survived, color='#5D6D7E', ax=axes[0], kde=False)
ag = sns.histplot(women[women['Survived']==0].Age.dropna(), bins=40, label=not_survived, color='#AEB6BF', ax=axes[0], kde=False)
ag.legend()
ag.set_title('Female')

ag = sns.histplot(men[men['Survived']==1].Age.dropna(), bins=18, label=survived, color='#5D6D7E', ax=axes[1], kde=False)
ag = sns.histplot(men[men['Survived']==0].Age.dropna(), bins=40, label=not_survived, color='#AEB6BF', ax=axes[1], kde=False)
ag.legend()
ag.set_title('Male')


#     Women have high survival chances between the ages 14-40 and men have high survival chances between the ages 18-40. However men have less survival chances in the similar age range. Infants have higher probability of survival.

# ### Embarked, Pclass and Sex:

# In[13]:


#Plot the graph on each port of embarkation and corelate with gender
FacetGrid = sns.FacetGrid(train_df, row='Embarked', height=4.5, aspect=2)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='Paired',  order=None, hue_order=None )
FacetGrid.add_legend()


#     Embarked seems to be correlated with survival, depending on the gender. Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton

# ### Pclass:

# In[14]:


#set bar plot for the Pclass(ticket class)
sns.barplot(x='Pclass', y='Survived', data=train_df, saturation=0.50)


#     Pclass is corelated with the survival chance of a person. pcalss and Fare are somewhat similar. A person in Upper Class, has a high probability of survival.

# In[15]:


#Use multiplot grid to plot the conditional relationship of Pclass and Survived feature
pclass_plot = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=2)
pclass_plot.map(plt.hist, 'Age', alpha=0.5, bins=20)
pclass_plot.add_legend()


#     Upper Class has a high probability of survival and Lower Class has less probability of survival. 

# ### SibSp and Parch:

#     These columns represent the no. of relatives of a passenger. SibSp represent the no. of siblings or spouse of a passenger. Parch represent the no. of parents, children of a passenger.

# In[16]:


#set new columns for relatives and not_alone by taking the data from SibSp and Parch
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
    
train_df['not_alone'].value_counts(sort=True)


# In[17]:


#plot the graph of relative and Survived using categorical plot
relative_plot = sns.catplot('relatives','Survived', data=train_df, kind='point', aspect=2)


#     A passenger has high chance of survival, if they have no. of relatives in the range of 1-3 and less chance of survial, if they have no relatives or more than 3 relatives, except the no. of relatives = 6.

# # Data Preprocessing 

# In[18]:


#drop passengerid from the dataset
train_df = train_df.drop(['PassengerId'], axis=1)


# ### Missing Data:

# In[19]:


#taking only the first three rows of missing data because it was found that Age, Cabin, Embarked has missing values.
missing_data.head(3)


#     Age:

# Upon checking the two datasets train_df and test_df, there are 177 and 86 missing values in Age column.

# In[20]:


mean = train_df['Age'].mean()
std = train_df['Age'].std()
null = train_df['Age'].isna().sum()
# compute random numbers between the mean, std deviation and null
rand_age = np.random.randint(mean - std, mean + std, size = null)
# fill NaN values in Age column with random values generated
age_slice = train_df['Age'].copy()
age_slice[np.isnan(age_slice)] = rand_age
train_df['Age'] = age_slice
train_df['Age'] = train_df['Age'].astype(int)
train_df['Age'].isna().sum()


# In[21]:


t_mean = test_df['Age'].mean()
t_std = test_df['Age'].std()
t_null = test_df['Age'].isna().sum()
# compute random numbers between the t_mean, t_std and t_null
trand_age = np.random.randint(t_mean - t_std, t_mean + t_std, size = t_null)
# fill NaN values in Age column with random values generated
age_slicet = test_df['Age'].copy()
age_slicet[np.isnan(age_slicet)] = trand_age
test_df['Age'] = age_slicet
test_df['Age'] = test_df["Age"].astype(int)
test_df["Age"].isna().sum()


#     Embarked:

# Embarked feature has only 2 missing values, we can find the mode of Embarked feature.

# In[22]:


train_df['Embarked'].describe()


# In[23]:


mode_value = train_df['Embarked'].mode()
mode_value


# In[24]:


#fill mode value into missing values
train_df['Embarked'] = train_df['Embarked'].fillna(mode_value[0])
train_df['Embarked'].isna().sum()


# In[25]:


testmode_value = train_df['Embarked'].mode()
testmode_value


# In[26]:


test_df['Embarked'] = test_df['Embarked'].fillna(mode_value[0])
test_df['Embarked'].isna().sum()


#     Cabin:

# In[27]:


#Check the missing values in test dataset.
test_df.isna().sum()


# In[28]:


#fill the missing values in Cabin column using an unknown value, U0
train_df['Cabin'] = train_df['Cabin'].fillna('U0')
test_df['Cabin'] = test_df['Cabin'].fillna('U0')


# In[29]:


train_df['Deck'] = train_df['Cabin'].apply(lambda x : x[0][0])
test_df['Deck'] = test_df['Cabin'].apply(lambda x : x[0][0])


# In[30]:


#for creating a deck column
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}


# In[31]:


train_df['Deck'] = train_df['Deck'].map(deck)
train_df['Deck'] = train_df['Deck'].fillna(0) #there's one missing value in train_df

test_df['Deck'] = test_df['Deck'].map(deck)


# In[32]:


train_df['Deck'] = train_df['Deck'].astype(int)


# In[33]:


test_df['Deck'] = test_df['Deck'].fillna(0)


# In[34]:


test_df['Deck'] = test_df['Deck'].astype(int)


# In[35]:


#drop the cabin column from the datasets
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# In[36]:


train_df.head()


# In[37]:


test_df.head()


# ### Converting Features: 

# In[38]:


train_df.info()


#     ‘Fare’ is a float and also other 4 categorical features: Name, Sex, Ticket and Embarked aren't int type.

# In[39]:


test_df.info()


#     Fare:

# In[40]:


train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)


#     Name:

# In[41]:


titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}


# In[42]:


#extract titles
train_df['Title'] = train_df.Name.str.extract('([A-Za-z]+)\.', expand=False)


# In[43]:


train_df['Title'].value_counts()


#     There were other titles apart from the categories in title. Replace those titles and assign into Rare

# In[44]:


train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev','Sir', 'Jonkheer', 'Dona'], 'Rare')


# In[45]:


train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')


# In[46]:


train_df['Title'] = train_df['Title'].map(titles)


# In[47]:


test_df['Title'] = test_df.Name.str.extract('([A-Za-z]+)\.', expand=False)


# In[48]:


test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev','Sir', 'Jonkheer', 'Dona'], 'Rare')


# In[49]:


test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')


# In[50]:


test_df['Title'] = test_df['Title'].map(titles)


# In[51]:


#deleting Name feature
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)


#     Sex:

# In[52]:


gender = {"male": 0, "female": 1}


# In[53]:


train_df['Sex'] = train_df['Sex'].map(gender)
test_df['Sex'] = test_df['Sex'].map(gender)


# In[54]:


train_df['Ticket'].describe()


#     Ticket attribute has 681 unique tickets, it will be difficult to convert the feature into useful category.

# In[55]:


#deleting Ticket feature
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


#     Embarked:

# In[56]:


port = {"S": 0, "C": 1, "Q": 2}


# In[57]:


train_df['Embarked'] = train_df['Embarked'].map(port)
test_df['Embarked'] = test_df['Embarked'].map(port)


# ### Creating Categories:

#     Age:

# In[58]:


train_df['Age'].describe()


#     Categorise the Age feature by forming groups, that have similar distribution.

# In[59]:


train_df.loc[train_df['Age'] <= 15, 'Age'] = 0
train_df.loc[(train_df['Age'] > 15) & (train_df['Age'] <= 20), 'Age'] = 1
train_df.loc[(train_df['Age'] > 20) & (train_df['Age'] <= 25), 'Age'] = 2
train_df.loc[(train_df['Age'] > 25) & (train_df['Age'] <= 29), 'Age'] = 3
train_df.loc[(train_df['Age'] > 29) & (train_df['Age'] <= 35), 'Age'] = 4
train_df.loc[(train_df['Age'] > 35) & (train_df['Age'] <= 41), 'Age'] = 5
train_df.loc[(train_df['Age'] > 41)] = 6


# In[60]:


train_df['Age'].value_counts()


# In[61]:


test_df.loc[test_df['Age'] <= 15, 'Age'] = 0
test_df.loc[(test_df['Age'] > 15) & (test_df['Age'] <= 20), 'Age'] = 1
test_df.loc[(test_df['Age'] > 20) & (test_df['Age'] <= 25), 'Age'] = 2
test_df.loc[(test_df['Age'] > 25) & (test_df['Age'] <= 29), 'Age'] = 3
test_df.loc[(test_df['Age'] > 29) & (test_df['Age'] <= 35), 'Age'] = 4
test_df.loc[(test_df['Age'] > 35) & (test_df['Age'] <= 41), 'Age'] = 5
test_df.loc[(test_df['Age'] > 41)] = 6


# In[62]:


test_df['Age'].value_counts()


# In[63]:


pd.qcut(train_df['Fare'], 6)


# In[64]:


train_df


# In[65]:


train_df.loc[train_df['Fare'] <= 6.0, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 6.0) & (train_df['Fare'] <= 7.0), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 7.0) & (train_df['Fare'] <= 9.0), 'Fare'] = 2
train_df.loc[(train_df['Fare'] > 9.0) & (train_df['Fare'] <= 17.33), 'Fare'] = 3
train_df.loc[(train_df['Fare'] > 17.33) & (train_df['Fare'] <= 34.0), 'Fare'] = 4
train_df.loc[(train_df['Fare'] > 34.0) & (train_df['Fare'] <= 512.0), 'Fare'] = 5


# In[66]:


train_df['Fare'].value_counts()


# In[67]:


test_df.loc[test_df['Fare'] <= 6.0, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 6.0) & (test_df['Fare'] <= 7.0), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 7.0) & (test_df['Fare'] <= 9.0), 'Fare'] = 2
test_df.loc[(test_df['Fare'] > 9.0) & (test_df['Fare'] <= 17.33), 'Fare'] = 3
test_df.loc[(test_df['Fare'] > 17.33) & (test_df['Fare'] <= 34.0), 'Fare'] = 4
test_df.loc[(test_df['Fare'] > 34.0) & (test_df['Fare'] <= 512.0), 'Fare'] = 5


# In[68]:


test_df['Fare'].value_counts()


# # Creating new Features

#     Age times Class:

# In[69]:


train_df['Age_Class'] = train_df['Age']*train_df['Pclass']
test_df['Age_Class'] = test_df['Age']*test_df['Pclass']


#     Fare per Person:

# In[70]:


train_df['Fare_Per_Person'] = train_df['Fare']/(train_df['relatives'] + 1)
train_df['Fare_Per_Person'] = train_df['Fare_Per_Person'].astype(int)


# In[71]:


test_df['Fare_Per_Person'] = test_df['Fare']/(test_df['relatives'] + 1)
test_df['Fare_Per_Person'] = test_df['Fare_Per_Person'].astype(int)


# In[72]:


train_df.head(10)


# In[73]:


test_df.head(10)


# In[74]:


test_sur_df = test_df['Survived']


# In[ ]:





# # Building ML Models

# In[75]:


X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']


# In[76]:


X_test = test_df.drop(['PassengerId','Survived'], axis=1).copy()


# In[77]:


X_test


# In[78]:


X_train


# ### Linear Regression:

# In[79]:


lin_reg = LinearRegression()


# In[80]:


lin_reg.fit(X_train, Y_train)


# In[81]:


Y_pred = lin_reg.predict(X_test)


# In[82]:


lin_acc = round(lin_reg.score(X_train, Y_train) * 100, 2)


# In[83]:


lin_acc


# ### Logistic Regression:

# In[84]:


log_reg = LogisticRegression(max_iter=500)


# In[85]:


log_reg.fit(X_train, Y_train)


# In[86]:


Y_pred = log_reg.predict(X_test)


# In[87]:


log_acc = round(log_reg.score(X_train, Y_train) * 100, 2)
log_acc


# ### Random Forest Classifier:

# In[88]:


rfc = RandomForestClassifier(n_estimators=100)


# In[89]:


rfc.fit(X_train, Y_train)


# In[90]:


Y_pred = rfc.predict(X_test)


# In[91]:


rfc.score(X_train, Y_train)


# In[92]:


rfc_acc = round(rfc.score(X_train, Y_train) * 100, 2)
rfc_acc


#     Parameter Tuning:

# In[93]:


cls=RandomForestClassifier()
n_estimators=[25,50,75,100] #number of decision trees in the forest.
criterion=['gini','entropy'] #criteria for choosing nodes
max_depth=[3,5,10] #maximum number of nodes in a tree 
parameters={'n_estimators': n_estimators,'criterion':criterion,'max_depth':max_depth} #this will undergo 4*1*3 = 12 iterations
RFC_cls = GridSearchCV(cls, parameters)
RFC_cls.fit(X_train,Y_train)


# In[94]:


RFC_cls.best_params_


# In[95]:


cls = RandomForestClassifier(n_estimators=75,criterion='gini',max_depth=10)


# In[96]:


cls.fit(X_train, Y_train)


# In[97]:


Y_pred = cls.predict(X_test)


# In[98]:


cls.score(X_train, Y_train)


# ### Linear Support Vector Machine:

# In[99]:


lin_svc = LinearSVC()


# In[100]:


lin_svc.fit(X_train, Y_train)


# In[101]:


Y_pred = lin_svc.predict(X_test)


# In[102]:


lin_svc_acc = round(lin_svc.score(X_train, Y_train) * 100, 2)
lin_svc_acc


# ### Gaussian Naive Bayes: 

# In[103]:


gaussian = GaussianNB()


# In[104]:


gaussian.fit(X_train, Y_train)


# In[105]:


Y_pred = gaussian.predict(X_test)


# In[106]:


gaussian_acc = round(gaussian.score(X_train, Y_train) * 100, 2)
gaussian_acc


# # Best Model

# In[107]:


results = pd.DataFrame({'Model': ['Linear Regression', 'Logistic Regression', 'Random Forest', 'Support Vector Machines', 'Naive Bayes'], 
                        'Score': [lin_acc, log_acc, rfc_acc, lin_svc_acc, gaussian_acc]})


# In[108]:


result_df = results.sort_values(by='Score', ascending=False)


# In[109]:


result_df = result_df.set_index('Score')


# In[110]:


result_df


#     Got high accuracy in Linear Regression.

# In[111]:


#END

