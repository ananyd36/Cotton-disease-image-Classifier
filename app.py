import sys
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, redirect, render_template,request
from werkzeug.utils import secure_filename
import cv2

IMG_SIZE = 100

app = Flask(__name__)

model_path = "model1.h5"
model = load_model(model_path)

def model_predict(img_path,model):
    categories = ["diseased cotton leaf","diseased cotton plant","fresh cotton leaf","fresh cotton plant"]
    img_array = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    new_arr = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    new_arr = new_arr/255
    pred = model.predict(new_arr.reshape(1,IMG_SIZE,IMG_SIZE,3))
    idx = np.argmax(pred)
    return categories[idx]


@app.route('/', methods=['GET'])
def index():
    return render_template('second.html')


@app.route('/uploaded', methods=['GET', 'POST'])
def uploaded():
    if request.method=="POST":
        f = request.files['file']
        filename = secure_filename(f.filename)
        # basepath = os.path.dirname(__file__)
        file_path = os.path.join(r"./uploads", filename)
        f.save(file_path)
        preds = model_predict(file_path,model)
        result = preds
        return render_template("uploaded.html",results = result, fname = filename)
    return None

if __name__ == '__main__':
    app.run(port=5000,debug=True)
