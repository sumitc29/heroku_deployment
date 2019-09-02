from flask import Flask, escape, request,render_template
import numpy as np
import pandas as pd 
import pickle

app = Flask(__name__)

"""reading model"""
model=pickle.load(open("model.sav",'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hello1')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'


@app.route('/predict',methods=['POST'])    
def predict():
    """getting input data """
    features=[float(x) for x in request.form.values()]
    f_features=[np.array(features)]
    predictions=model.predict(f_features)
    
    return render_template('index.html',prediction_text ="type of flower is $ {}".format(predictions))
    
if __name__ == "__main__":
    app.run(debug=True)
