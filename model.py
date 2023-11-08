import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
data=pd.read_csv(r'C:\Users\SREELAKSHMI\Downloads\credit.csv')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
obj_list = data.select_dtypes(include="object").columns
for obj in obj_list:
    data[obj] = le.fit_transform(data[obj].astype(str))
    
    
X = data.drop(['Credit_Score','Name','Customer_ID','SSN','Payment_Behaviour','Occupation'], axis=1)
y = data['Credit_Score']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf=rf.fit(X_train,y_train)

pickle.dump(rf,open('model.pkl','wb'))
