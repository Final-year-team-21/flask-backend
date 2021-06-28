from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
import datetime as dt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score as rs
from sklearn.metrics import precision_score as ps
from sklearn.metrics import f1_score as fs

from sklearn.metrics import recall_score as rs
from sklearn.metrics import precision_score as ps
from sklearn.metrics import f1_score as fs

def model_hello():
    return "Model Prediction success"

def model_prediction(df):

    df_train = pd.read_csv('train.csv')
    le_l = LabelEncoder()    
    df_train['location'] = le_l.fit_transform(df_train['location'])

    le_c = LabelEncoder()
    df_train['country'] = le_c.fit_transform(df_train['country'])

    le_g = LabelEncoder()
    df_train['gender'] = le_g.fit_transform(df_train['gender'].astype(str))

    le_s1 = LabelEncoder()
    df_train['symptom1'] = le_s1.fit_transform(df_train['symptom1'].astype(str))

    print(df_train['symptom2'].astype(str))

    le_s2 = LabelEncoder()
    df_train['symptom2'] = le_s2.fit_transform(df_train['symptom2'].astype(str))

    le_s3 = LabelEncoder()
    df_train['symptom3'] = le_s3.fit_transform(df_train['symptom3'].astype(str))

    le_s4 = LabelEncoder()
    df_train['symptom4'] = le_s4.fit_transform(df_train['symptom4'].astype(str))

    le_s5 = LabelEncoder()
    df_train['symptom5'] = le_s5.fit_transform(df_train['symptom5'].astype(str))

    le_s6 = LabelEncoder()
    df_train['symptom6'] = le_s6.fit_transform(df_train['symptom6'].astype(str))

    #exporting the departure encoder
    output_location = open('Location_encoder.pkl', 'wb')
    pickle.dump(le_l, output_location)
    output_location.close()

    output_country = open('Country_encoder.pkl','wb')
    pickle.dump(le_c,output_country)
    output_country.close()

    output_gender = open('Gender_encoder.pkl','wb')
    pickle.dump(le_g,output_gender)
    output_gender.close()

    output_symptom1 = open('Symptom1_encoder.pkl','wb')
    pickle.dump(le_s1,output_symptom1)
    output_symptom1.close()

    output_symptom2 = open('Symptom2_encoder.pkl','wb')
    pickle.dump(le_s2,output_symptom2)
    output_symptom2.close()

    output_symptom3 = open('Symptom3_encoder.pkl','wb')
    pickle.dump(le_s3,output_symptom3)
    output_symptom3.close()

    output_symptom4 = open('Symptom4_encoder.pkl','wb')
    pickle.dump(le_s4,output_symptom4)
    output_symptom4.close()

    output_symptom5 = open('Symptom5_encoder.pkl','wb')
    pickle.dump(le_s5,output_symptom5)
    output_symptom5.close()

    output_symptom6 = open('Symptom6_encoder.pkl','wb')
    pickle.dump(le_s6,output_symptom6)
    output_symptom6.close()

    tdata = pd.read_csv('train.csv')
    print(tdata.head())

    tdata = pd.read_csv('train.csv')
    tdata = tdata.drop('id',axis=1)
    tdata = tdata.fillna(np.nan,axis=0)
    tdata['age'] = tdata['age'].fillna(value=tdata['age'].mean())
    tdata['location'] = le_l.fit_transform(tdata['location'].astype(str))
    tdata['country'] = le_c.fit_transform(tdata['country'].astype(str))
    tdata['gender'] = le_g.fit_transform(tdata['gender'].astype(str))
    tdata[['symptom1']] = le_s1.fit_transform(tdata['symptom1'].astype(str))
    tdata[['symptom2']] = le_s2.fit_transform(tdata['symptom2'].astype(str))
    tdata[['symptom3']] = le_s3.fit_transform(tdata['symptom3'].astype(str))
    tdata[['symptom4']] = le_s4.fit_transform(tdata['symptom4'].astype(str))
    tdata[['symptom5']] = le_s5.fit_transform(tdata['symptom5'].astype(str))
    tdata[['symptom6']] = le_s6.fit_transform(tdata['symptom6'].astype(str))

    tdata['sym_on'] = pd.to_datetime(tdata['sym_on'])
    tdata['hosp_vis'] = pd.to_datetime(tdata['hosp_vis'])
    tdata['sym_on']= tdata['sym_on'].map(dt.datetime.toordinal)
    tdata['hosp_vis']= tdata['hosp_vis'].map(dt.datetime.toordinal)
    tdata['diff_sym_hos']= tdata['hosp_vis'] - tdata['sym_on']

    tdata = tdata.drop(['sym_on','hosp_vis'],axis=1)
    print(tdata)

    #print(tdata.isna().sum())

    rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                        criterion='gini', max_depth=2, max_features='auto',
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=2, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=100,
                        n_jobs=None, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)
    classifier = AdaBoostClassifier(rf,50,0.01,'SAMME.R',10)

    X = tdata[['location','country','gender','age','vis_wuhan','from_wuhan','symptom1','symptom2','symptom3','symptom4','symptom5','symptom6','diff_sym_hos']]
    Y = tdata['death']

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
    classifier.fit(X_train,np.array(Y_train))

    pred = np.array(classifier.predict(X_test))

    recall = rs(Y_test,pred)
    precision = ps(Y_test,pred)
    f1 = fs(Y_test,pred)
    ma = classifier.score(X_test,Y_test)

    '''print('*** Evaluation metrics for test dataset ***\n')
    print('Recall Score: ',recall)
    print('Precision Score: ',precision)
    print('F1 Score: ',f1)
    print('Accuracy: ',ma)
    a = pd.DataFrame(Y_test)
    a['pred']= classifier.predict(X_test)
    print('\n\tTable 3\n')
    print(a.head())'''

    pkl = open('Location_encoder.pkl', 'rb')
    le_l = pickle.load(pkl)
    pkl.close()
    df['location'] = le_l.transform(df['location'])

    pkl = open('Country_encoder.pkl', 'rb')
    le_c = pickle.load(pkl)
    pkl.close()
    df['country'] = le_c.transform(df['country'])

    pkl = open('Gender_encoder.pkl', 'rb')
    le_g = pickle.load(pkl)
    pkl.close()
    df['gender'] = le_g.transform(df['gender'])

    pkl = open('Symptom1_encoder.pkl', 'rb')
    le_s1 = pickle.load(pkl)
    pkl.close()
    df['symptom1'] = le_s1.transform(df['symptom1'])

    pkl = open('Symptom2_encoder.pkl', 'rb')
    le_s2 = pickle.load(pkl)
    pkl.close()
    df['symptom2'] = le_s2.transform(df['symptom2'])

    pkl = open('Symptom3_encoder.pkl', 'rb')
    le_s3 = pickle.load(pkl)
    pkl.close()
    df['symptom3'] = le_s3.transform(df['symptom3'])

    pkl = open('Symptom4_encoder.pkl', 'rb')
    le_s4 = pickle.load(pkl)
    pkl.close()
    df['symptom4'] = le_s4.transform(df['symptom4'])

    pkl = open('Symptom5_encoder.pkl', 'rb')
    le_s5 = pickle.load(pkl)
    pkl.close()
    df['symptom5'] = le_s5.transform(df['symptom5'])

    pkl = open('Symptom6_encoder.pkl', 'rb')
    le_s6 = pickle.load(pkl)
    pkl.close()
    df['symptom6'] = le_s6.transform(df['symptom6'])

    #print(df)

    predicted_value = classifier.predict(df)
    print("pvalue:",predicted_value[0])
    return predicted_value[0]