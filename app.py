import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from path import Path as p
from flask import Flask, render_template, request, flash, send_file, abort
from werkzeug.utils import secure_filename
from flask_cors import cross_origin
from preprocess import Cardiopreprocess
import time

# from flask_bootstrap import Bootstrap

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)  # initializing a flask app

app.secret_key = "12345"

# Setting paths for both windows and linux env
dirname = Path.cwd()
DOWNLOAD_FOLDER = (os.path.join(dirname, Path('csv/output')))
UPLOAD_FOLDER = os.path.join(dirname, Path("csv/input"))

# clean old files uploaded before 7 days.
def cleanup():
    dirname = Path.cwd()
    seven_days_ago = time.time() - 7 * 86400
    base = p(os.path.join(dirname, Path('csv/')))

    for file in base.walkfiles():
        if file.mtime < seven_days_ago:
            file.remove()

@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    cleanup()
    return render_template("index.html", static_url_path='/static')


@app.route('/predictsingle', methods=['POST'])
@cross_origin()
def predictsingle():
    try:
        #  reading the inputs given by the user

        age = int(request.form["age"])
        gender = int(request.form["Gender"])
        height = int(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = int(request.form["ap_hi"])
        ap_lo = int(request.form["ap_lo"])
        gluc = int(request.form["gluc"])
        smoke = int(request.form["smoke"])
        alco = int(request.form["alco"])
        active = int(request.form.get("active"))
        chol = int(request.form["chol"])

        columns = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco",
                   "active"]
        values = np.array([age, gender, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active])
        data = dict(zip(columns, values))
        data = pd.DataFrame(data, index=[0])
        print(data)
        cardiopreprocess = Cardiopreprocess()
        transformeddata = cardiopreprocess.preprocessdata(data)
        # predictions using the loaded model file
        loaded_model = joblib.load('model.pckl')
        prediction = loaded_model.predict(transformeddata)
        # showing the prediction results in a UI
        if prediction[0] == 0:
            msg = "Hurray ! There are no symptoms of heart Disease"
        else:
            msg = "Have heart Disease"
        return render_template('predict.html', prediction=msg)
    except Exception as e:
        return 'The Exception message is: ', e
        # return 'something is wrong'


@app.route('/bulkpredict', methods=['GET'])
@cross_origin()
def bulkpredict():
    return render_template('Bulkpredict.html')


def download_csv(report_filename):
    try:
        # report_filename = 'report.csv'
        report_filepath = str(Path(DOWNLOAD_FOLDER + '/' + report_filename))  # send file dosen't accept path objects
        return send_file(report_filepath, attachment_filename=report_filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)


def predictmultiple(filename):
    try:
        # reading the inputs given by the user
        datafilepath = (UPLOAD_FOLDER + '/' + filename)
        data = pd.read_csv(datafilepath, delimiter=';')
        print()
        id = data['id']
        data = data.drop('id', axis=1)
        if 'cardio' in data.columns:
            data.drop('cardio', inplace=True, axis=1)
        cardiopreprocess = Cardiopreprocess()
        transformeddata = cardiopreprocess.preprocessdata(data)

        # predictions using the loaded model file
        loaded_model = joblib.load('model.pckl')
        predictions = loaded_model.predict(transformeddata)
        # showing the prediction results in a UI
        predictions = pd.DataFrame({'id': id, 'predictions': predictions})

        # save predictions to file
        report_filename = filename + "_report.csv"
        report_filepath = Path(DOWNLOAD_FOLDER + '/' + report_filename)
        if os.path.exists(report_filename):
            os.remove(report_filename)
        predictions.to_csv(report_filepath, index=False)
        print(report_filename)
        return report_filename
    except Exception as e:
        return 'The Exception message is: ', e
        # return 'something is wrong'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


@app.route('/upload_csv', methods=['POST'])
@cross_origin()
def upload_csv():
    if request.method == 'POST':
        file = request.files['data']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template('Bulkpredict.html')
        if file and allowed_file(file.filename):
            data_filename = secure_filename(file.filename)
            data_filepath = Path(UPLOAD_FOLDER + '/' + data_filename)
            if os.path.exists(data_filepath):
                os.remove(data_filepath)
            file.save(data_filepath)
            report_filename = predictmultiple(data_filename)
            return download_csv(report_filename)
        else:
            return "Only csv format is supported"
    return render_template('Bulkpredict.html')

if __name__ == "__main__":
#    app.run(debug=True, host='0.0.0.0')
     app.run(debug=True, host='0.0.0.0',port=5001)
