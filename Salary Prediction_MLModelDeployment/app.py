from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
# create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict",methods = ["POST"])
def predict():
    sex = request.form['sex']
    designation = request.form['designation']
    age = request.form['age']
    unit = request.form['unit']
    leaves_used = request.form['leaves_used']
    ratings = request.form['ratings']
    past_exp = request.form['past_exp']
    years_worked = request.form['years_worked']

# creating a data frame with the input values

    input_data = {'Sex':[sex],
              'Designation':[designation],
              'Age':[age],
              'Unit':[unit],
              'LeavesUsed':[leaves_used],
              'Ratings':[ratings],
              'PastExp':[past_exp],
              'YearsWorked':[years_worked]}

    print(input_data)

    input_df = pd.DataFrame(input_data)
    print(input_df)

# making a prediction using a model
    prediction = model.predict(input_df)
    return render_template("index.html",prediction_text = " The predicted salary is {}".format(prediction[0]))
if __name__ == "__main__":
    flask_app.run(debug= True)