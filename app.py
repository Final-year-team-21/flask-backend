import numpy as np
from flask import Flask, request, jsonify, render_template, session, Response
import pickle
import pandas as pd
import numpy as np
from model import model
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

app = Flask(__name__)
app.secret_key = "covid-model"
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/symptom-list", methods=['GET', 'POST'])
def SymptomList():
    data = pd.read_csv('data.csv')
    data = data.replace(np.nan, '', regex=True)
    if request.method == 'GET':
        location=jsonify(data['gender'].unique().tolist())
        symptom1=data['symptom1'].unique().tolist()
        symptom2=data['symptom2'].unique().tolist()
        symptom3=data['symptom3'].unique().tolist()
        symptom4=data['symptom4'].unique().tolist()
        symptom5=data['symptom5'].unique().tolist()
        symptom6=data['symptom6'].unique().tolist()
        list=symptom1+symptom2+symptom3+symptom4+symptom5+symptom6
        list = [x for x in list if x]
        unique_list = get_unique_list(list)
        return jsonify(unique_list)
    else:
        session['symptom'] = request.json["symptom"]
        print(session.get('symptom'))
        response = Response(status=200)
        response.data = "Symptom Set to %s" % session.get('symptom')
        return response

@app.route('/location',methods=['POST'])
def LocationChange():
    session['location'] = request.json["location"]
    print(session.get('location'))
    response = Response(status=200)
    return response

@app.route('/country',methods=['POST'])
def CountryChange():
    session['country'] = request.json["country"]
    print(session.get('country'))
    response = Response(status=200)
    return response

@app.route('/gender',methods=['POST'])
def GenderChange():
    session['gender'] = request.json["gender"]
    print(session.get('gender'))
    response = Response(status=200)
    return response

@app.route('/age',methods=['POST'])
def AgeChange():
    session['age'] = request.json["age"]
    print(session.get('age'))
    response = Response(status=200)
    return response

@app.route('/predict')
def Predict():
    tdata = {
        'location':session.get('location'),
        'country':session.get('country'),
        'gender':session.get('gender'),
        'age':int(session.get('age')),
        'vis_wuhan':0,
        'from_wuhan':0,
        'symptom1':session.get('symptom1'),
        'symptom2':np.nan,
        'symptom3':np.nan,
        'symptom4':np.nan,
        'symptom5':np.nan,
        'symptom6':np.nan,
        'diff_sym_hos':0
    }
    tdata['location'] = encoder.fit_transform(tdata['location'])
    tdata['country'] = encoder.fit_transform(tdata['country'])
    tdata['gender'] = encoder.fit_transform(tdata['gender'])
    tdata[['symptom1']] = encoder.fit_transform(tdata['symptom1'])
    tdata[['symptom2']] = encoder.fit_transform(tdata['symptom2'])
    tdata[['symptom3']] = encoder.fit_transform(tdata['symptom3'])
    tdata[['symptom4']] = encoder.fit_transform(tdata['symptom4'])
    tdata[['symptom5']] = encoder.fit_transform(tdata['symptom5'])
    tdata[['symptom6']] = encoder.fit_transform(tdata['symptom6'])
    X = tdata[['location','country','gender','age','vis_wuhan','from_wuhan','symptom1','symptom2','symptom3','symptom4','symptom5','symptom6','diff_sym_hos']]
    print(model.predict(X))

def get_unique_list(list):

    list_of_unique_list = []

    unique_list = set(list)

    for number in unique_list:
        list_of_unique_list.append(number)

    return list_of_unique_list

if __name__ == "__main__":
    app.run(debug=True)
