from flask import Flask, request, jsonify, render_template, session, Response
import pickle
import pandas as pd
from model import *
from csv import writer

data = pd.read_csv('train.csv')
data = data.replace(np.nan, '', regex=True)

app = Flask(__name__)
app.secret_key = "covid-model"

@app.route("/")
def home():
    return model_hello()

@app.route("/symptom1", methods=['GET', 'POST'])
def Symptom1():
    if request.method == 'GET':
        symptom1=data['symptom1'].unique().tolist()
        return jsonify(symptom1)
    else:
        session['symptom1'] = request.json["symptom1"]
        print(session.get('symptom1'))
        response = Response(status=200)
        return response

@app.route("/symptom2", methods=['GET', 'POST'])
def Symptom2():
    if request.method == 'GET':
        symptom2=data['symptom2'].unique().tolist()
        return jsonify(symptom2)
    else:
        session['symptom2'] = request.json["symptom2"]
        print(session.get('symptom2'))
        response = Response(status=200)
        return response

@app.route("/symptom3", methods=['GET', 'POST'])
def Symptom3():
    if request.method == 'GET':
        symptom3=data['symptom3'].unique().tolist()
        return jsonify(symptom3)
    else:
        session['symptom3'] = request.json["symptom3"]
        print(session.get('symptom3'))
        response = Response(status=200)
        return response

'''@app.route("/symptom4", methods=['GET', 'POST'])
def Symptom4():
    if request.method == 'GET':
        symptom4=data['symptom4'].unique().tolist()
        return jsonify(symptom4)
    else:
        session['symptom4'] = request.json["symptom4"]
        print(session.get('symptom4'))
        response = Response(status=200)
        return response

@app.route("/symptom5", methods=['GET', 'POST'])
def Symptom5():
    if request.method == 'GET':
        symptom5=data['symptom5'].unique().tolist()
        return jsonify(symptom5)
    else:
        session['symptom5'] = request.json["symptom5"]
        print(session.get('symptom5'))
        response = Response(status=200)
        return response

@app.route("/symptom6", methods=['GET', 'POST'])
def Symptom6():
    if request.method == 'GET':
        symptom6=data['symptom6'].unique().tolist()
        return jsonify(symptom6)
    else:
        session['symptom6'] = request.json["symptom6"]
        print(session.get('symptom6'))
        response = Response(status=200)
        return response'''


@app.route('/location',methods=['GET','POST'])
def LocationChange():
    if request.method == 'GET':
        location=data['location'].unique().tolist()
        return jsonify(location)
    else:
        session['location'] = request.json["location"]
        print(session.get('location'))
        response = Response(status=200)
        return response

@app.route('/country',methods=['GET','POST'])
def CountryChange():
    if request.method == 'GET':
        country=data['country'].unique().tolist()
        return jsonify(country)
    else:
        session['country'] = request.json["country"]
        print(session.get('country'))
        response = Response(status=200)
        return response

@app.route('/gender',methods=['GET','POST'])
def GenderChange():
    if request.method == 'GET':
        gender=data['gender'].unique().tolist()
        return jsonify(gender)
    else:
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
    columns = ['location','country','gender','age','vis_wuhan','from_wuhan','symptom1','symptom2','symptom3','symptom4','symptom5','symptom6','diff_sym_hos']
    tdata = [session.get('location'), session.get('country'), session.get('gender'),session.get('age'),1,0,session.get('symptom1'),session.get('symptom2'),session.get('symptom3'),'nan','nan','nan',0]
    df = pd.DataFrame([tdata],columns=columns)
    pvalue = model_prediction(df)
    pvalue = int(pvalue)
    return jsonify(pvalue)

def get_unique_list(list):

    list_of_unique_list = []

    unique_list = set(list)

    for number in unique_list:
        list_of_unique_list.append(number)

    return list_of_unique_list

if __name__ == "__main__":
    app.run(debug=True)


'''@app.route("/symptom-list", methods=['GET', 'POST'])
def SymptomList():
    if request.method == 'GET':
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
        return response'''
