
# coding: utf-8

# ### Initial Dataset Analysis
# Import libraries and dataset  <br>
# Inspect the head of the data and key statistics.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# In[2]:


data = pd.read_csv('~/Desktop/house_data.csv')
data.head()


# In[3]:


data.describe()


# **Distirbuted Data due to the following : **
# 
# - maximum square feet of 13,540 and a minimum of 290.
# - largest number of bedrooms is 33; very large house included in our data set along with the existence of a house with no bedrooms.

# **Most attractive bedroom-wise house**
# 
# Find the most common house for a house buyer. First thing most people take into account are the number of bedrooms that a house consists of.

# In[4]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom(s)')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# The above bar chart shows that 3 bedroom houses are sold most commonly, with the 4 bedroom houses coming up next. 
# It could be useful for a starting point of a house builder and/or seller towards attracting a higher number of buyers.

# ** Locality of houses **
# 
# Now that the highest selling bedroom-wise houses are known, next thing to look at is the location of houses based on the given latitude and longitude data.

# In[5]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=14)
plt.xlabel('Latitude', fontsize=14)
plt.show()
sns.despine


# The above visualization depicts the concentration and placement of data, from which we can infer that there is a higher concentration of houses within the interception of ranges -47.7 and -47.8 for latitude and -122.2 to -122.4 for longitude, making this area the ideal location to build and sell a house.

# ### Test Variable analysis prior to feature engineering and cleaning of the data

# In[6]:


data['price'].describe()


# In[7]:


# Histogram of price
sns.distplot(data['price'] , fit=norm);

print("Skewness: %f" % data['price'].skew())
print("Kurtosis: %f" % data['price'].kurt())


# ### Multivariable analysis is carried out towards some underlying relations
# 

# In[8]:


categorical = len(data.select_dtypes(include=['object']).columns)
numerical = len(data.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', categorical+numerical, 'features')


# Convert the data found within the 'date' column to 1’s and 0’s so our data are less influenced.

# In[9]:


converted_dates = [1 if int(str(value)[:4]) == 2014 else 0 for value in data.date]
data['date'] = converted_dates


# In order to discover the underlying relations between these features and house prices,<br> a **correlation matrix** is constructed below using the traning set.

# In[10]:


matrix = data.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(matrix, vmax=0.7, square=True)


# **The top 5 correlated variables to price** are calculated below and examined for containing any outliers, in which case they will be removed; pre-cleaning on each of these top features is carried out.

# In[11]:


topcols = matrix.nlargest(5, 'price')['price'].index
top = pd.DataFrame(topcols)
top.columns = ['Most Correlated Features']
top


# The square feet living variable is the most ***predicting variable***, which is a continuous variable, expressing that the larger square footage of the house is, the higher its price will be. Next variables to follow are the grade, the square footage of house apart from basement and the living room area in 2015 which implies some renovations.
# 

# A joint plot is constructed of the price against each of the above variables to visually **detect and remove any outliers** <br>towards a higher ** Pearson correlation coefficient value **.

# In[12]:


# Living Area vs Sale Price
g = sns.jointplot(x=data['sqft_living'], y=data['price'], kind='reg')
g.ax_joint.legend_.remove()

df1 = data[['price','sqft_living']]
print (df1.corr(method='pearson'))


# In[13]:


data = data.drop(data[(data['sqft_living']>12000) 
                         & (data['price']<3000000)].index).reset_index(drop=True)

sns.jointplot(x=data['sqft_living'], y=data['price'], kind='reg')
#g.ax_joint.legend_.remove()

df1 = data[['price','sqft_living']]
print (df1.corr(method='pearson'))


# In[14]:


df1 = data[['price','grade']]
print (df1.corr(method='pearson'))

ax = sns.boxplot(x="grade", y="price", data=df1)


# In[15]:


# Grade vs Sale Price after removal of some outliers

data = data.drop(data[(data['grade']<5)].index).reset_index(drop=True)

data = data.drop(data[(data['grade']==13) 
                         & (data['price']>5550000)].index).reset_index(drop=True)
data = data.drop(data[(data['grade']==12) 
                         & (data['price']>4000000)].index).reset_index(drop=True)
data = data.drop(data[(data['grade']==11) 
                         & (data['price']>2800000)].index).reset_index(drop=True)
data = data.drop(data[(data['grade']==10) 
                         & (data['price']>2000000)].index).reset_index(drop=True)
data = data.drop(data[(data['grade']==9) 
                         & (data['price']>1300000)].index).reset_index(drop=True)
data = data.drop(data[(data['grade']==8) 
                         & (data['price']>1100000)].index).reset_index(drop=True)
data = data.drop(data[(data['grade']==7) 
                         & (data['price']>700000)].index).reset_index(drop=True)
data = data.drop(data[(data['grade']==6) 
                         & (data['price']>600000)].index).reset_index(drop=True)
data = data.drop(data[(data['grade']==5) 
                         & (data['price']>300000)].index).reset_index(drop=True)

df1 = data[['price','grade']]

ax = sns.boxplot(x="grade", y="price", data=df1)

plt.ylim(0, 8000000)
print (df1.corr(method='pearson'))


# In[16]:


# Sqft above vs Sale Price

g = sns.jointplot(x=data['sqft_above'], y=data['price'], ylim=(0, 8000000), kind='reg')
g.ax_joint.legend_.remove()

df = data[['price','sqft_above']]

print (df.corr(method='pearson'))


# In[17]:


# Sqft above vs Sale Price after removal of some outliers
data = data.drop(data[(data['sqft_above']<3000) 
                         & (data['price']>2000000)].index).reset_index(drop=True)

g = sns.jointplot(x=data['sqft_above'], y=data['price'],ylim=(0, 8000000), kind='reg')
g.ax_joint.legend_.remove()

df1 = data[['price','sqft_above']]
print (df1.corr(method='pearson'))


# In[18]:


# Living Area 15 vs Sale Price

g = sns.jointplot(x=data['sqft_living15'], y=data['price'],ylim=(0, 8000000), kind='reg')
g.ax_joint.legend_.remove()

df1 = data[['price','sqft_living15']]


print (df1.corr(method='pearson'))


# In[19]:


# Living Area 15 vs Sale Price after removal of some outliers
data = data.drop(data[(data['price']>2600000)].index).reset_index(drop=True)

g = sns.jointplot(x=data['sqft_living15'], y=data['price'],ylim=(0, 8000000), kind='reg')
g.ax_joint.legend_.remove()

df1 = data[['price','sqft_living15']]

print (df1.corr(method='pearson'))


# **Change in Pearson r coefficient after removal of outliers**
# 
# square footage living: 0.7020 --> 0.7022 <br>
# grade: 0.6672 --> 0.7372 <br>
# square footage above: 0.6242 --> 0.6272 <br>
# square footage living 15: 0.6038 --> 0.6098

# ### Test Variable (price) Analysis following feature engineering and cleaning of the data
# 

# In[20]:


data['price'].describe()


# In[21]:


# Plot Histogram of price
sns.distplot(data['price'] , fit=norm);

print("Skewness: %f" % data['price'].skew())
print("Kurtosis: %f" % data['price'].kurt())


# Looking now at the **kurtosis score**, it has decreased from 37.16 to 7.50, which makes it closer to the expected value of 3. Even though the skewness remains high, it has decreased from  4.13 to 2.24.

# ### Machine Learning Techniques
# 
# Three different **Machine Learning models ** are applied to the processed data using a random seed of 42. <br>
# A prediction score; the accuracy of each prediction, is calculated for each model. <br>
#     **-  Linear Regression** <br>
#     **-  Gradient Boosting Regression**<br>
#     **-  Random Forest Regression**

# In[ ]:


#Splitting the data to train & test sets
labels = data['price']
train1 = data.drop(['id', 'price'],axis=1)
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state = 42)


# In[22]:


reg = LinearRegression()

reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

print (reg.score(x_test,y_test))

# Build a plot
plt.scatter(y_pred, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()


# In[23]:


reg = GradientBoostingRegressor()

reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

print (reg.score(x_test,y_test))

# Build a plot
plt.scatter(y_pred, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()


# In[24]:


reg = RandomForestRegressor()

reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

print (reg.score(x_test,y_test))

# Build a plot
plt.scatter(y_pred, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')

plt.show()


# ### Results & Conclusion

# In[26]:


model = ['LR', 'GBR', 'RFR']
score = [75.6, 88.7, 89.2]
df = pd.DataFrame(score, model)
df.reset_index(level=0, inplace=True)
df.columns = ['model', 'score']

#make the bar plot
ax = sns.barplot(x = 'score', y = 'model', data = df)
plt.xlim(0, 100)
plt.title('Prediction Scores')
plt.xlabel('Prediction Score')
plt.ylabel('Prediction Model')

for p in ax.patches:
    width = p.get_width()
    if width >= 85 :
        plt.text(5+p.get_width(), p.get_y()+0.55*p.get_height(),
                 '{:1.2f}'.format(width),
                 ha='center', va='center')
        p.set_facecolor('green')
    else:
        plt.text(5+p.get_width(), p.get_y()+0.55*p.get_height(),
                '{:1.2f}'.format(width),
                 ha='center', va='center')
        p.set_facecolor('orange')


# Based on the above barchart, which depicts the corresponding prediction model used and the generated prediction score, it can be concluded that ** compared to the threshold of 85% accuracy, the Gradeint Boosting (GBR) and Random Forest Regression (RFR) have outperformed the Linear Regression (LR) model by at least 13.1% **. <br>
# Both **GBR and RFR have a higher accuracy** than 85%, of 88.70% and 89.70% respectively. 
