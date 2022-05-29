import json
import pickle
import boto3
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def lambda_handler(event, context):
      #1. Parse out query string params
      rooms = event['queryStringParameters']['rooms']
      student_teacher_ratio = event['queryStringParameters']['str']
      poverty_percent = event['queryStringParameters']['pov']

      #if testing, comment above queries and uncomment/adjust below features
      #rooms = 5
      #student_teacher_ratio=18
      #poverty_percent=12
  
      print('rooms ' + str(rooms))
      print('student/teacher ' + str(student_teacher_ratio))
      print('poverty percent '+ str(poverty_percent))

      
  # load model:
      s3 = boto3.resource('s3')
      model = pickle.loads(s3.Bucket("housemodelgbr1823").Object("chosen_tree_model.pkl").get()['Body'].read())

      # fixes a Decision tree regressor error:
      setattr(model,'n_features_',3)

      # make prediction:
      new_vals = [[rooms, student_teacher_ratio, poverty_percent]]
      price = round(model.predict(new_vals)[0],2)


      #2. Construct the body of the response object
      modelResponse = {}
      modelResponse['rooms'] = rooms
      modelResponse['student/teacher'] = student_teacher_ratio
      modelResponse['poverty percent'] = poverty_percent
      modelResponse['predicted home price'] = price


      #3. Construct http response object
      responseObject = {}
      responseObject['statusCode'] = 200
      
      responseObject['headers'] = {}
      responseObject['headers']['Content-Type'] = 'application/json'
      
      responseObject['body'] = json.dumps(modelResponse)
      print('returning object')


      #4 Return the response object
      return responseObject
