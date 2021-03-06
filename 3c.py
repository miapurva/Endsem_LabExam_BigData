import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle 
import time
from nltk.tokenize import TweetTokenizer
import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the BlackFriday dataset
df=pd.read_csv('big_BlackFriday.csv')
df.dropna(inplace=True)
#df.drop(df.columns[0],axis=1,inplace=True)
non_cat = [f for f in df.columns if df.dtypes[f] != 'object']
cat = [f for f in df.columns if df.dtypes[f] == 'object']

def treat_missing_numeric(df,columns,how = 'mean'):
    '''
    Function to treat missing values in numeric columns
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns need to be imputed
        - how = valid values are 'mean', 'mode', 'median','ffill', numeric value
    Expected Output -
        - Pandas dataframe with imputed missing value in mentioned columns
    '''
    if how == 'mean':
        for i in columns:
            print("Filling missing values with mean for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mean())
            
    elif how == 'mode':
        for i in columns:
            print("Filling missing values with mode for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mode())
    
    elif how == 'median':
        for i in columns:
            print("Filling missing values with median for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].median())
    
    elif how == 'ffill':
        for i in columns:
            print("Filling missing values with forward fill for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(method ='ffill')
    
    elif type(how) == int or type(how) == float:
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how,i))
            df.ix[:,i] = df.ix[:,i].fillna(how)
    else:
        print("Missing value fill cannot be completed")
    return df

def treat_missing_categorical(df,columns,how = 'mode'):
    '''
    Function to treat missing values in numeric columns
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns need to be imputed
        - how = valid values are 'mode', any string or numeric value
    Expected Output -
        - Pandas dataframe with imputed missing value in mentioned columns
    '''
    if how == 'mode':
        for i in columns:
            print("Filling missing values with mode for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mode()[0])
    elif type(how) == str:
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how,i))
            df.ix[:,i] = df.ix[:,i].fillna(how)
    elif type(how) == int or type(how) == float:
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how,i))
            df.ix[:,i] = df.ix[:,i].fillna(str(how))
    else:
        print("Missing value fill cannot be completed")
        return df
treat_missing_numeric(df,non_cat,how = 'mean')
treat_missing_categorical(df,cat,how = 'mode')

#print(df)

from sklearn.model_selection import train_test_split
#X = df.copy()

X = df['Purchase']
#y = df['Age']
y, z = pd.factorize(df['Age'])
X, y = X.tolist(), y.tolist()
#print(X, y)
x = [[0 for i in range(2)] for j in range(len(X))]
for i in range(len(X)):
    x[i][0] = X[i]
    x[i][1] = y[i]
# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
#y = df['Age']
#print (x)
def holdout_cv(x,y,size = 0.3, seed = 1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = size, random_state = seed)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = holdout_cv(x, y, size = 0.3, seed = 1)

reg = LinearRegression().fit(x_train, y_train)
print "\n Score:", reg.score(x, y)
print "\n Coefficient:", reg.coef_
print "\n Intercept:", reg.intercept_ 
#print "\n Fit:", reg.fit(list(x_test),list(y_test))
print "\n Predict:", reg.predict(list(x_test))


# Plot outputs
plt.scatter(X,y, color='black')
#plt.plot(X,y, color='blue', linewidth=3)
plt.title('LinearRegression model')
labels=['X-axis:Purchase,Y-axis:Age']
plt.legend(labels,loc=3)
plt.xticks(())
plt.yticks(())
plt.show()
