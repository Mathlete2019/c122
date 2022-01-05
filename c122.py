#1.import cv2 - This is the library with which we are going to use our computer's camera.
import cv2
#2.import numpy as np - This is so that we can perform complex mathematical/list operations.
import numpy as np
#3.import pandas as pd - This is so that we can treat our data as DataFrames. We already know how helpful they are.
import pandas as pd
#4.import seaborn as sns - This is a python module to prettify the charts that we draw with matplotlib. We have used it a couple of times.
import seaborn as sns
#5.import matplotlib.pyplot as plt -This library is used to draw the charts.
import matplotlive.pyplot as plt
#6.from sklearn.datasets import fetch_openml - This function allows us to retrieve a data set by name from OpenML, a public repository for machine learning data and experiments.
from sklearn.datasets import fetch_openml
#7.from sklearn.model_selection import train_test_split - This is to split our data into training and testing.
from sklearn.model_selection import train_test_split
#8.from sklearn.linear_model import LogisticRegression - This is for creating a Logistic Regression Classifier.
from sklearn.linear_model import LogisticRegression 
#9.from sklearn.metrics import accuracy_score - This is to measure the accuracy score of the model.
from sklearn.metrics import accuracy_score

from PIL import Image
import PIL.ImageOps

#setting and https context to fetch data from OpenML
if(not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    getattr(ssl,'_create_unverified_context', None)):
    ssl._create_default_https_context=ssl._create_unverified_context
#fetching the data
import os, ssl, time
X,y=fetch_openml('mnist_784',version=1, return_X_y=True)
print(pd.Series(y).value_count())
classes=['0','1','2','3','4','5','6','7','8','9'] 
nclasses=len(classes)
