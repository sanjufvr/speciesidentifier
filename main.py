import streamlit as st
import pandas as pd
import numpy as np
from os import path
import pickle

st.title("flower species predictor")
petal_length = st.number_input("please choose a petal length between 1.0 to 6.9",placeholder="please enter the petal length",min_value=1.0,max_value=6.9,value=None)
petal_width = st.number_input("please choose a petal width between 0.1 to 2.5",placeholder="please enter the petal width",min_value =0.1, max_value=2.5,value=None)
sepal_length = st.number_input("please choose a sepal length between 4.3 to 7.9",placeholder="please enter the sepal length",min_value =4.3, max_value=7.9,value=None)
sepal_width = st.number_input("please choose a sepal width between 0.1 to 2.5",placeholder="please enter the sepal width",min_value =0.1, max_value=2.5,value=None)

user_input = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)

#using the .pkl file, creating an ML model named 'iris_classifier'
st.write(user_input)
model_path = path.join("model","iris_classifier.pkl")
with open (model_path,'rb') as file:
    iris_predictor=pickle.load(file)

species={0:'setosa',1:'versicolor',2:'virginica'}

if st.button("predict species"):
      if((petal_length == None) or (petal_width == None)or
        (sepal_length == None) or (sepal_width == None)):
         st.write("please fill all the values")
         #will be executed when any value is not executed properly
      else:
          #prediction can be done here we are expecting a dataframe
          predicted_species = iris_predictor.predict(user_input)
          #predicted_species[0] will give us the value in the dataframe
          #we use that value to find the corresponding species from the the
          #dictionary 'species'

          st.write("the species is ",species[predicted_species[0]])