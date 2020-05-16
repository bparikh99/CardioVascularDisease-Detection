from flask import Flask ,request, jsonify, render_template
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template('cardioweb.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    if(output==0):
        return render_template('cardioweb.html', prediction_text="You have Low Chances of Cardio Vascular Disease ")
    else:
        return render_template('cardioweb.html', prediction_text="You have High Chances of Cardio Vascular Disease ")

if __name__ == "__main__":
    app.run(host='localhost', debug=True)
  