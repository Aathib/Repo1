# CIS fraud detection competition

import pandas as pd
data = pd.read_csv(r'C:\Users\aathi\Downloads\ieee-fraud-detection\train_transaction.csv')
data.head()
from sklearn.model_selection import train_test_split


#downsizing the majority class
from sklearn.utils import resample
df_majority = data[data.isFraud==0]
df_minority = data[data.isFraud==1]

df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=20663,     # to match minority class
                                 random_state=123) # reproducible results

df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled.isFraud.value_counts()

#filling the missing values with mean & mode
df_downsampled.isna().sum()
df_downsampled.fillna(df_downsampled.mean(), inplace = True)
df_downsampled.fillna(df_downsampled.mode(), inplace = True)
df_downsampled.to_csv('df_downsampled.csv')
(df_downsampled.isnull().sum()/len(df_downsampled))*100

#dropping email columns
df_downsampled.drop(['P_emaildomain','R_emaildomain'],axis = 1, inplace = True)

#dropping low variance columns
numeric  = df_downsampled.columns
var = df_downsampled.var()
variable = [ ]
for i in range(0,len(var)):
    if var[i]>=10:   #setting the threshold as 10%
       variable.append(numeric[i+1])
df1 = df_downsampled[variable]

df1.columns
dfcorr = df1.drop('isFraud',1)
dfcorr.corr()

y = df_downsampled.isFraud
X = df_downsampled.drop('isFraud', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf_2 = LogisticRegression().fit(X_train, y_train)
