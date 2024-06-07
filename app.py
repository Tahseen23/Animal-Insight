from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import requests
import pandas as pd
from io import BytesIO

app = Flask(__name__)
model = load_model('vgg_model.h5')
model_yolo = YOLO('yolov8n.pt')
target_img = os.path.join(os.getcwd(), 'static/images')

# Ensure the static/images directory exists
if not os.path.exists(target_img):
    os.makedirs(target_img)

@app.route('/')
def index_view():
    return render_template('index.html')


ALLOWED_EXT = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def load_image(img):
    x = img.resize((150, 150))
    x = np.array(x)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def yolo(filename):
    img_path = os.path.join(target_img, filename)
    img = Image.open(img_path)
    out = model_yolo(img_path)
    if len(out[0].boxes.xyxy.numpy()) > 1:
        box = out[0].boxes.xyxy.numpy()[1]
    else:
        box = out[0].boxes.xyxy.numpy()[0]
    x_min, y_min, x_max, y_max = box
    box = (x_min, y_min, x_max, y_max)
    img = img.crop(box)
    return img

def get_dataframe():
    url = r"https://raw.githubusercontent.com/Tahseen23/Animal_Project/main/Project.animal.csv"
    df = pd.read_csv(url)
    return df

def get_name(pred, df):
    if '_id' not in df.columns:
        raise ValueError("The '_id' column is not present in the DataFrame")
    return df[df["_id"] == pred]['name'].iloc[0]

# def get_image(df, pred):
#     url = df[df["_id"] == pred]["url"].iloc[0]
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content))
#     return img

def get_description(df, pred):
    if '_id' not in df.columns:
        raise ValueError("The '_id' column is not present in the DataFrame")
    return df[df["_id"] == pred]['About'].iloc[0]

def favourable(df, pred):
    if '_id' not in df.columns:
        raise ValueError("The '_id' column is not present in the DataFrame")
    text = df[df["_id"] == pred]['Survival'].iloc[0]
    return ". ".join([sentence.strip() for sentence in text.replace("\\n", '').replace("**","").split('.') if sentence.strip()])

def issue(df, pred):
    if '_id' not in df.columns:
        raise ValueError("The '_id' column is not present in the DataFrame")
    text = df[df["_id"] == pred]['Problem'].iloc[0]
    return ". ".join([sentence.strip() for sentence in text.replace("\\n", '').replace("**","").split('.') if sentence.strip()])

def get_location(df,pred):
    return df[df["_id"] == pred]['location'].iloc[0]


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(target_img, filename)
            file.save(file_path)
            img = yolo(filename)
            final_img = load_image(img)
            pred = np.argmax(model.predict(final_img))
            df = get_dataframe()
            name = get_name(pred, df)
            about = get_description(df, pred)
            iss = issue(df, pred)
            fav = favourable(df, pred)
            loc=get_location(df,pred)
            img.save(os.path.join(target_img, 'output.jpg'))
            return render_template('prediction.html', name=name,
                                   About=about, Favourable=fav, Survival_Issue=iss,Location=loc,
                                   user_image='my_static/images/output.jpg')
        else:
            return "Failed"
    return "Method not allowed", 405

@app.route('/my_static/images/<filename>')
def send_image(filename):
    return send_from_directory(target_img, filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)


            



