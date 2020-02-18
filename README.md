# SFCrimePrediction
Crime Classification in San Francisco with a focus on live model deployment. 

My focus for this project is to deploy a live model which serves a front end giving me some experience in having a live model in production.

**Libraries used**: Catboost, Pandas, Numpy


The progress of the project currently is as follows:
* Part 1: Initial Data Exploration and Feature Engineering-<a href="https://github.com/ankur26/SFCrimePrediction/blob/master/notebooks/01_data_exploration_and_feature_engineering.ipynb">**link**<a>
* Part 2: Model selection and hyperparameter optimization-<a href="https://github.com/ankur26/SFCrimePrediction/blob/master/notebooks/02_model_creation.ipynb">**link**<a>
* Part 3: Model deployment process. <a href = "https://github.com/ankur26/SFCrimePrediction/blob/master/deployment/intro.py">**Link to streamlit code**</a>
  
**App Link** : <a href ="http://52.144.46.109:8501">**Link**</a>
Please use the server in moderation as it costs me money and if there are any problems report it on my
<a href="mailto:ab7869@nyu.edu">email</a> or message me on <a href="https://www.linkedin.com/in/ankur-bhatkalkar">LinkedIn</a> 

## Observations :
  * We see that a majority of the crime conducted is indeed Larceny and Theft
  * A good amount of crimes take place in the Central Southern and Northern districts as seen by Case report counts.
  * Most of the cases go unresolved as a huge margin of 69 % cases are left with no resolution.
## A note on the libraries used
  * Catboost is a very strong and numerically optimized boosting framework and is especially strong in dealing with categorical datatypes in a very efficient manner. Also the added GPU training support cut training times from a single day to a matter of a single hour.
  * Streamlit also is a very effective library to quickly iterate on and display your work as it is based on python so there is no need to worry or focus on the nitty gritty design details of HTML or Javascript(Though it is always an asset to have)


References : 
* Streamlit.io : <a href="www.streamlit.io">Link</a>
* Deploying a streamlit app on AWS EC2 : <a href="https://towardsdatascience.com/how-to-deploy-a-streamlit-app-using-an-amazon-free-ec2-instance-416a41f69dc3">Link</a>
* Catboost-A high-performance open source library for gradient boosting on decision trees : <a href="https://catboost.ai/">Link</a>
* San Francisco Crime Classification: <a href="https://www.kaggle.com/c/sf-crime">Link</a>
* How to use OpenStreetMap image to visualize map information: <a href="https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db">Link</a>
