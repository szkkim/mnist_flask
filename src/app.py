from flask import request, make_response, Flask, flash
from flask import render_template, session, redirect, url_for
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import pickle
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import constant as const
from PIL import Image
import pandas as pd
import numpy as np




app = Flask(__name__, template_folder=const.TEMPLATE_PATH)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'hard to guess string'


sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)

with open(const.CNN_MODEL, 'rb') as f:
    obj = pickle.load(f)
app.config['MODEL'] = obj


def process_model(f):
    img = Image.open(const.UPLOAD_PATH + f.filename).convert("L")
    img = np.resize(img, (28, 28, 1))
    im2arr = np.array(img)
    label = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    X = im2arr / 255.0
    X = X.reshape(-1, 28, 28, 1)
    model = app.config['MODEL']
    with graph.as_default():
        set_session(sess)
        predicted_prob = model.predict_proba(X)
    model_score = predicted_prob[:, 1]
    rec = str(model.predict_classes(X)[0])
    return rec

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in const.ALLOWED_EXTENSIONS

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route('/success', methods=['GET', 'POST'])
def success():
    session.pop('_flashes', None)
    rec = request.args['rec']
    return render_template('rec.html', rec=rec)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if request.files['file'].filename == '':
            flash("File not found", "danger")
            return redirect(url_for('index'))
        if f and allowed_file(f.filename):
            f.save(const.UPLOAD_PATH + secure_filename(f.filename))
            flash("file uploaded!", "success")
            rec = process_model(f)
            return redirect(url_for('success', rec=rec))
        else:
            flash("Incorrect File type, must be image", "warning")
    return redirect(url_for('index'))

if __name__ == '__main__':
    HOSTNAME = 'localhost'
    PORT = 8080
    app.run(host=HOSTNAME, port=PORT, debug=False)
