import pandas as pd
import numpy as np
import pickle
import os

DATA_PATH=os.path.join(os.path.dirname(__file__),'..','data','churn.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..','models', 'rf_model.pkl')

df=pd.read_csv(DATA_PATH)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].median())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Churn']=le.fit_transform(df['Churn']) # Yes → 1, No → 0

X=df.drop(['Churn','customerID'],axis=1)
y=df['Churn']

cat_cols=X.select_dtypes(include='object').columns.to_list()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

##Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(
    [
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat_cols),
        ('num',StandardScaler(),num_cols)
    ]
)

##Model Pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
rf_model=Pipeline(steps=[
    ('preprocess',preprocessor),
    ('classifier',RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ))
])

##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

rf_model.fit(X_train,y_train)
y_pred=rf_model.predict(X_test)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(rf_model, f)

ENCODER_PATH = os.path.join(os.path.dirname(__file__),'..', 'models', 'label_encoder.pkl')
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(le, f)

print(f"\n🎯 Model saved to {MODEL_PATH}")
print(f"🔠 LabelEncoder saved to {ENCODER_PATH}")

