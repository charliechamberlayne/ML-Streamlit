import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('Machine Learning App 🤖')

st.info('This app builds a machine learning model')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data visualisation'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# data prep
with st.sidebar:
  st.header('Input Features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgerson'))
  bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
  flipper_lenth_mm = st.slider('Flipper length (mm)',  172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))


  #create df for input features
  data= {'island': island,
       'bill_length_mm': bill_length_mm,
       'bill_depth_mm': bill_depth_mm,
       'flipper_lenth_mm': flipper_lenth_mm,
       'body_mass_g': body_mass_g,
       'sex': gender}

  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins

  #Encode x
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
input_row = df_penguins[:1]

  #encode y
target_mapper = {'Adelie': 0,
                   'Chinstrap':1,
                   'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)
y
y_raw

with st.expander('Data preparation'):
  st.write('**Encoded X (Input penguin)**')
  input_row
  st.write('**Encoded y**')
  y


#model training and inference

##train the ML model
clf= RandomForestClassifier()
clf.fit(X_raw, y)

#apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

prediction_proba


  


                            
