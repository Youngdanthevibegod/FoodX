# FoodX
These are the steps I took to do exploratory 
data analysis on Jupiter Notebook.
(Attached is also a file called FoodX where I did my coding.=)

%pip install pyodide-http
import pyodide_http
pyodide_http.patch_all()
import matplotlib
import pandas

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('Downloads/XTern 2024 Artificial Intelegence Data Set - Xtern_TrainData.csv')
data.head()

X = np.array(data['Time']).reshape(1,-1)                 
y = np.array(data['Order'])

plt.scatter(X,y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=0)

lr =LinearRegression()

lr.fit(X_train,y_train)


Visualization and Findings
* I used a graph to look at the data from a graphical perspective.
* I was able to find out from the line graphs and bar graphs that Indiana University is the dominant school in the data. Also, the Fried Catfish Basket was ordered the most.

Business use cases for the data
*  This data can also be used to determine which or how many colleges are more likely to do online orders. Then FoodX can target those colleges to bring in more income.
*   This data can also be used to identify what order college students like the most. The favorite food on the menu is more likely to attract more customers back to FoodX. Giving FoodX long-term customers.


Implications of data collection, storage, and data biases

1. The ethical implications of these factors are privacy concerns and transparency. It's important that the data collection of college students respect privacy and adhere to data protection regulations. They must also be informed of what will happen to the data collected, how it is stored, and its use.
   
3. Business outcome implications of these factors are the data quality and biased data. Inaccurate data can lead to incorrect analysis and decisions that affect business outcomes. Also, biased data can lead to unfair treatment of a certain group or population.
   
5. Technical implications of these factors are data security and data retention. It is very important that the data is stored securely to avoid access, breaches, and cyberattacks. Also, it is crucial to know data retention policies to avoid storing unnecessary data for extended periods. This can help reduce attacks and potential security risks.

For my model, I chose the Linear Regression model. I chose this model because Linear regression is a statistical modeling technique used to describe a continuous response variable as a linear function of one or more predictor variables. Because linear regression models are simple to interpret and easy to train, they are often the first models to try when working with a new data set. While building a Linear Regression model to predict a customer's order from their available information, I came upon a huge wall. I tried searching for a solution from all the resources that I used but to no avail, I was not able to complete my model and get to the prediction part. I am truly disappointed that I couldn't get to that part but there's only so much that I can do and know.

However, I think that given the work required to bring a solution like this to maturity and its performance, the consideration I would make to determine if this is a suitable course of action is to see how well the AI is able to keep predicting a customer's order. Because that determines the outcome of FoodX's future.
