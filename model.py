import sys
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# surpress warnings regarding sklearn
warnings.filterwarnings("ignore", category = UserWarning, module = "sklearn")




#SECTION 1   Getting the data and splitting it

# fetch data
data = pd.read_csv('./housing.csv')

# add price feature as MEDV and all other features listed as 'features'
prices = data['MEDV']
features = data.drop('MEDV',axis=1)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.1, random_state=42)




#SECTION 2 making the models and testing them (decision tree regressor is fixed in its predictability, but grb
#           will automatically adjust how it learns and test its results based on the user provided parameters)

# the model functions
tree = DecisionTreeRegressor(max_depth=5)
modelgbr = GradientBoostingRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,criterion='friedman_mse')


# determine how well the models perform
tree_model = tree.fit(X_train,y_train)
tree_score = tree.score(X_train,y_train)
print(f"tree training score: {tree_score:.5f}")
print(f"tree test score {tree.score(X_test,y_test):.5f}")

gbr_model = modelgbr.fit(X_train,y_train)
gbr_score = modelgbr.score(X_train,y_train)
print(f"gradient boosting training score: {gbr_score:.5f}")
print(f"gradient boosting test score {modelgbr.score(X_test,y_test):.5f}")




# SECTION 3  predicting on completely new data

# new data with # of rooms, student/pupil, poverty level (%)
client_data =  [[3, 6, 7],
                [9, 20, 2],
                [18,  2, 20]]

# use tree model to predict on new data
for i, price in enumerate (tree.predict(client_data)):
    print(f"Tree predicted selling price for Client {i+1}'s home: ${price:.2f}")


# use gbr model to predict on new data
for i, price in enumerate (modelgbr.predict(client_data)):
    print(f"Gradient Boosting predicted selling price for Client {i+1}'s home: ${price:.2f}")




# SECTION 4  pickeling the tree and gradient boosting models

pickle.dump(modelgbr, open('chosen_gbr_model.pkl', 'wb'))
pickle.dump(tree, open('chosen_tree_model.pkl', 'wb'))
