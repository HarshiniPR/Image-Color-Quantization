from flask import Flask, render_template, request, session
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.secret_key="sec123"
def compress_image_kmeans(image_path, k):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
    compressed_pixels = np.uint8(compressed_pixels)

    compressed_img = compressed_pixels.reshape(img.shape)
    compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_RGB2BGR)

    return compressed_img
from werkzeug.utils import secure_filename

@app.route("/", methods=["GET", "POST"])
def index():
    k = 10 #default val
    if request.method == "POST":
        k = int(request.form["k"])

        # FIRST upload
        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            filename = secure_filename(file.filename)

            original_path = os.path.join(app.config["UPLOAD_FOLDER"], "original_" + filename)
            file.save(original_path)

            session["original_image"] = original_path

        # REUSE original image
        original_path = session.get("original_image")

        if not original_path or not os.path.exists(original_path):
            return "Please upload an image first"

        compressed_img = compress_image_kmeans(original_path, k)

        output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"compressed_k{k}.png")
        cv2.imwrite(output_path, compressed_img)

        return render_template(
            "index.html",
            original=original_path,
            compressed=output_path,
            k=k
        )

    return render_template("index.html", k=k)

if __name__ == "__main__":
    app.run(debug=True)
