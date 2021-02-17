
import numpy as np
import tensorflow.keras
from PIL import Image, ImageOps
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin


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



app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def main():
    return """
        Application is working
    """


@app.route("/process", methods=["POST"])
def processReq():
    data = request.files["img"]
    data.save("img.jpg")
    resp = predict_class("img.jpg")
    return resp


if __name__ == "__main__":
    app.run(debug=True)