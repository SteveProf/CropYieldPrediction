from flask import Flask as fk, render_template, request, url_for
import pickle
import numpy as np

app = fk(__name__)
@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/predicted", methods=["GET","POST"])
def predict():
    rainfall = request.form['rainfall']
    humidity = request.form['humidity']
    N = request.form['N']
    P = request.form['P']
    K=request.form["K"]
    temperature = request.form["temp"]
    ph = request.form["ph"]
    form = np.array([[N ,P,K,temperature,humidity,ph,rainfall]])
    model = pickle.load(open("model.pkl","rb"))
    prediction = model.predict(form)[0]
    prediction = prediction.capitalize()
    return render_template("predicted.html", result=prediction)

@app.route("/apidoc")
def apidoc():
    return render_template("apidoc.html")

if __name__=="__main__":
    app.run(debug=True)