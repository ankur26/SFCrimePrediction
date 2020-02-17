import streamlit as st
import numpy as np 
import pandas as pd
from catboost import *
import json
#print("Imports done")
@st.cache()
def load_model():
    model = CatBoost()
    model.load_model(fname = "C:\\Users\\Ankur Bhatkalkar\\Documents\\GitHub\\SFCrimePrediction\\models\\final_model")
    return model
@st.cache()
def load_location():
    data = pd.read_csv("C:\\Users\\Ankur Bhatkalkar\\Documents\\GitHub\\SFCrimePrediction\\deployment\\location.csv",keep_default_na=False)
    return data.drop("Unnamed: 0",axis=1)

@st.cache()
def load_max_min_location():
    file = open("C:\\Users\\Ankur Bhatkalkar\\Documents\\GitHub\\SFCrimePrediction\\deployment\\location_dict.txt","r")
    return json.load(file)

st.title("Know what crimes happen in your area!!")
st.header("Using San Francisco Open Crime Data this a model designed to predict which crimes are most likely to happen depending on factors used in the dataset")
model = load_model()
locations = load_location()
location_dict= load_max_min_location()
st.subheader("Enter the parameters")
st.slider(label="Longitude",min_value = float(location_dict["min_longitude"][0]),
          value=float(location_dict["min_longitude"][0]),max_value = float(location_dict["max_longitude"][0]),step=0.0001,key="Longitude")

st.slider(label="Latitude",min_value = float(location_dict["min_latitude"][0]),
          value=float(location_dict["min_latitude"][0]),max_value = float(location_dict["max_latitude"][0]),step=0.0001,key="Latitude")

