
from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load and process the data
    data = pd.read_csv(filepath)
    if "x" not in data.columns or "y" not in data.columns:
        return jsonify({"error": "File must contain 'x' and 'y' columns"}), 400

    # Generate the plot
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    plt.plot(data["x"], data["y"], color="cyan", linewidth=2)
    plt.title("Sci-Fi Themed Plot", fontsize=14)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plot_path = os.path.join(PLOT_FOLDER, "plot.png")
    plt.savefig(plot_path)
    plt.close()

    return jsonify({"plot_url": f"/{plot_path}"})


if __name__ == "__main__":
    app.run(debug=True)
