import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

df=pd.read_csv(r'C:\Users\himanshimehta\Downloads\cardio\CardioVascularDisease\cardio_train.csv',sep=';')

df.drop(columns=['id'],axis=1,inplace=True)
df['age']=(df['age']/365).round().astype(int)
df['female']=df['gender']==1
df['sex']=np.where(df['female']==True,1,0)
df.drop(columns=['gender','female'],axis=1,inplace=True)

df.drop(df[(df['height'] > df['height'].quantile(0.975)) | (df['height'] < df['height'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['weight'] > df['weight'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index,inplace=True)

df=df[['age','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','sex','active','cardio']]
X=df.iloc[:,:11]
y=df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42,shuffle=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


kfold = KFold(n_splits=10, random_state=42)

dt = DecisionTreeClassifier(criterion = 'gini', random_state=100,max_depth=10)
dt.fit(X_train,y_train)
# y_pred = dt.predict(X_test) 
pickle.dump(dt, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[63,172,80,150,80,1,1,1,1,1,1,32]]))