from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
from classifier.main_predict import resnet, predict_class

from gevent.pywsgi import WSGIServer

import os

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # predict class
        model = resnet()
        img, y_pred, y_class = predict_class(file_path, model)
        result = str(y_class[0][0][1]) + ' \n \n' + str(round(y_class[0][0][2] * 100, 1)) + '% confidence'
        os.remove(file_path)
        return result
    return None


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()