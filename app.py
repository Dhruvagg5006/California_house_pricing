import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
app=Flask(__name__)

# Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features=[float(x) for x in request.form.values()]
    # final_features=[np.array(int_features)]
    # prediction=model.predict(final_features)

    # output=round(prediction[0],2)

    # return render_template('home.html',prediction_text='Predicted House Price is $ {}'.format(output))
def predict_api():
    '''
    For direct API calls through request
    '''
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))

    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])
if __name__=="__main__":
    app.run(debug=True)
    
    # prediction=model.predict([np.array(list(data.values()))])

    # output=prediction[0]
    # return jsonify(output)
