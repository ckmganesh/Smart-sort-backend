
import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
import pandas as pd
from io import BytesIO
import tensorflow.keras
import numpy as np
from PIL import Image, ImageOps



# app = FastAPI()
# @app.get('/')
# def index():
#     return {'message': 'Hello, World'}

# @app.get('/{name}')
# def get_name(name: str):
#     return {'Welcome To Waste Segregator': f'{name}'}


# def read_imagefile(file) -> Image.Image:
#     image = Image.open(BytesIO(file))
#     return image

# def predict_class(image:Image.Image):

#     np.set_printoptions(suppress=True)
#     model = tensorflow.keras.models.load_model('keras_model.h5')
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=float)
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.ANTIALIAS)
#     image_array = np.asarray(image)
#     image.show()
#     normalized_image_array = (image_array.astype(float) / 127.0) - 1
#     data[0] = normalized_image_array
#     prediction = model.predict(data)
#     return prediction

# @app.post("/predict/image")
# async def predict_api(file: UploadFile = File(...)):
#     extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     if not extension:
#         return "Image must be jpg or png format!"
#     image = read_imagefile(await file.read())
#     prediction = predict_class(image)
    
#     return prediction

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

from flask import Flask, render_template, request
import numpy as np
from flask_cors import CORS, cross_origin

#names = ["daisy", "dandelon", "roses", "sunflowers", "tulips"]



# Process image and predict label
def predict_class(file_path):

    np.set_printoptions(suppress=True)
    model = tensorflow.keras.models.load_model('keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(file_path)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    image.show()
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    if(prediction[0][0]>0.5):
        return "Biodegradable"
    else:
        return "NonBiodegradable"


# Initializing flask application
app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def main():
    return """
        Application is working
    """

# About page with render template
# Process images
@app.route("/process", methods=["POST"])
def processReq():
    data = request.files["img"]
    data.save("img.jpg")
    resp = predict_class("img.jpg")
    return resp


if __name__ == "__main__":
    app.run(debug=True)