import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler 

application=Flask(__name__)
app=application

# import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/ridge (1).pkl','rb'))
standard_scaler=pickle.load(open('models/scaler (1).pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Gender=float(request.form.get('Gender'))
        Age= float(request.form.get('Age'))
        Height=float(request.form.get('Height'))
        Weight=float(request.form.get('Weight'))
        Duration=float(request.form.get('Duration'))
        Heart_Rate=float(request.form.get('Heart_Rate'))
        Body_Temp=float(request.form.get('Body_Temp'))
        
        new_data_scaled=standard_scaler.transform([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])
        result=ridge_model.predict(new_data_scaled)
        
        return render_template('home.html',result=int(result[0]))
    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0')