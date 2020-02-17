import streamlit as st
import numpy as np 
import pandas as pd
from catboost import *
import json
from sklearn.metrics.pairwise import euclidean_distances
#print("Imports done")
@st.cache()
def load_model():
    model = CatBoost()
    model.load_model(fname = "C:\\Users\\Ankur Bhatkalkar\\Documents\\GitHub\\SFCrimePrediction\\models\\final_model")
    return model

def load_location():
    data = pd.read_csv("C:\\Users\\Ankur Bhatkalkar\\Documents\\GitHub\\SFCrimePrediction\\deployment\\location.csv",keep_default_na=False)
    return data.drop("Unnamed: 0",axis=1)

@st.cache()
def load_max_min_location():
    file = open("C:\\Users\\Ankur Bhatkalkar\\Documents\\GitHub\\SFCrimePrediction\\deployment\\location_dict.txt","r")
    return json.load(file)

st.title("Know what crimes happen in your area!!")
st.header("Using San Francisco Open Crime Data this model has been trained designed to predict which crimes are most likely to happen depending on factors used in the dataset")
model = load_model()
locations = load_location()
location_dict= load_max_min_location()
st.subheader("Enter the parameters")
longitude = st.slider(label="Longitude",min_value = float(location_dict["min_longitude"]),
          value=float(location_dict["min_longitude"]),
          max_value = float(location_dict["max_longitude"]),
          step=0.0001,key="Longitude",
          format = "%2.4f")

latitude = st.slider(label="Latitude",min_value = float(location_dict["min_latitude"]),
          value=float(location_dict["min_latitude"]),
          max_value = float(location_dict["max_latitude"]),
          step=0.0001,key="Latitude",
          format = "%2.4f")

time = st.time_input(label="Time",key="Time")
date = st.date_input(label="Date",key="Date")

locations["distances"] = ((locations["X"]-longitude)**2 + (locations["Y"]-latitude)**2)**0.5
st.write("The closest estimated area and paramters are")
min_distance_df = locations.iloc[locations["distances"].argmin()]
st.dataframe(min_distance_df)
min_distance_df["Day"] = date.day
min_distance_df["Year"] = date.year
min_distance_df["Month"] = date.month
min_distance_df["DayOfWeek"] = date.strftime("%A").upper()
min_distance_df["Hour"] = time.hour
min_distance_df["Minute"] = time.minute
st.write("Updating it with date gives us")
st.dataframe(min_distance_df)
st.write("With our dataframe ready to predict we can make an inference")
predictions = pd.DataFrame(model.predict())