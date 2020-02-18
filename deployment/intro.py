import streamlit as st
import numpy as np 
import pandas as pd
from catboost import *
import json
import altair as alt
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
df = pd.DataFrame(np.array([longitude,latitude]).reshape(1,2),columns=["lon","lat"])
st.map(df,zoom=12)
time = st.time_input(label="Time",key="Time")
date = st.date_input(label="Date",key="Date")

locations["distances"] = ((locations["X"]-longitude)**2 + (locations["Y"]-latitude)**2)**0.5
st.write("The closest estimated area and paramters are")
min_distance_df = pd.DataFrame([locations.iloc[locations["distances"].argmin()]],columns=locations.columns.values)
st.dataframe(min_distance_df.head())
min_distance_df["Day"] = int(date.day)
min_distance_df["Year"] = int(date.year)
min_distance_df["Month"] = int(date.month)
min_distance_df["DayOfWeek"] = date.strftime("%A").upper()
min_distance_df["Hour"] = int(time.hour)
min_distance_df["Minute"] = int(time.minute)
min_distance_df["block_present"] = min_distance_df["block_present"].astype(int)
min_distance_df["block_number"] = min_distance_df["block_number"].astype(int)
min_distance_df["X"] = min_distance_df["X"].astype(float)
min_distance_df["Y"] = min_distance_df["Y"].astype(float)
min_distance_df.drop("distances",axis=1,inplace=True)
st.write("Updating it with the other parameters gives us")
st.dataframe(min_distance_df.head())
st.write("With our dataframe ready here is the inferred output")

features = min_distance_df.dtypes.reset_index()
cat_features = features.loc[features[0] == "object","index"].values
test = Pool(data = min_distance_df,cat_features=cat_features)

predictions = pd.DataFrame(np.array(model.predict(test,prediction_type="Probability")).T*100,columns=["Percentage"])
predictions["Crime"] = model.classes_
predictions["Crime"] = predictions["Crime"].astype(str)
predictions_sorted = predictions.sort_values("Percentage",ascending=False)
st.dataframe(predictions_sorted)
st.write("You can see the top 5 crimes happening are in the "+", ".join(predictions_sorted["Crime"][:5].values)+" Category")
st.write("Here is graph indicating all the crime percentages in a sorted format")
c = alt.Chart(predictions_sorted).mark_bar().encode(x=alt.X("Crime",sort="-y"),y="Percentage")
st.altair_chart(c)

