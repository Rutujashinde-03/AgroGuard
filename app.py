# from flask import Flask, request, render_template
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np
# import os

# # app = Flask(__name__)
# app = Flask(__name__, static_folder="static")
# model = load_model('agro_model.h5')
# classes = os.listdir("dataset")

# @app.route("/", methods=["GET", "POST"])
# def home():
#     prediction = ""
#     if request.method == "POST":
#         image = request.files["image"]
#         image.save("uploaded.jpg")
#         img = load_img("uploaded.jpg", target_size=(128, 128))
#         img = img_to_array(img) / 255.0
#         img = np.expand_dims(img, axis=0)

#         pred = model.predict(img)
#         class_index = np.argmax(pred)
#         prediction = f"Prediction: {classes[class_index]}"
#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import json

app = Flask(__name__, static_folder="static")

# Load trained model
model = load_model('agro_model.h5')

# Load class names from json
with open("class_names.json") as f:
    classes = list(json.load(f).keys())

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        image = request.files["image"]
        image.save("uploaded.jpg")
        img = load_img("uploaded.jpg", target_size=(128, 128))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        class_index = np.argmax(pred)

        # Clean up class name
        readable_name = classes[class_index].replace("___", " - ").replace("_", " ")
        prediction = f"Detected: {readable_name}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
