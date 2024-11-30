import matplotlib
matplotlib.use('Agg')  # setting the backend to 'Agg'

from flask import Flask, request, render_template, jsonify
import os
import matplotlib.pyplot as plt
import skrf as rf
import numpy as np  # make sure numpy is imported

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

uploaded_files = {}

@app.route("/")
def index():
    return render_template("index.html")

# time domain page
@app.route("/time-domain")
def time_domain():
    return render_template("time_domain.html")

# time gating page
@app.route("/time-gating")
def time_gating():
    return render_template("time_gating.html")

# handle file uploads (common for all tabs)
@app.route("/upload", methods=["POST"])
def upload_file():
    if "files" not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("files")

    if not files or files[0].filename == "":
        return jsonify({"error": "No selected files"}), 400

    uploads_info = []

    for file in files:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # load data using scikit-rf
        try:
            network = rf.Network(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to load file {filename}: {str(e)}"}), 400

        # detect available parameters in the file
        available_parameters = []
        if hasattr(network, 's11'):
            available_parameters.append('s11')
        if hasattr(network, 's21'):
            available_parameters.append('s21')
        if hasattr(network, 's12'):
            available_parameters.append('s12')
        if hasattr(network, 's22'):
            available_parameters.append('s22')

        if not available_parameters:
            return jsonify({"error": f"No S-parameters found in the file {filename}"}), 400

        # save info about the uploaded file and its parameters
        uploaded_files[filename] = {
            "filepath": filepath,
            "parameters": available_parameters
        }

        uploads_info.append({
            "filename": filename,
            "parameters": available_parameters
        })

    # return the list of uploaded files and their parameters to the client
    return jsonify({
        "uploads": uploads_info
    })


# handle plot updates for frequency domain
@app.route("/update-plot", methods=["POST"])
def update_plot():
    data = request.get_json()
    selections = data.get("selections")

    if not selections:
        return jsonify({"error": "No parameters selected"}), 400

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    for selection in selections:
        filename = selection.get("filename")
        parameter = selection.get("parameter")

        if filename not in uploaded_files:
            return jsonify({"error": f"File {filename} not found"}), 400

        filepath = uploaded_files[filename]["filepath"]

        # wczytaj dane za pomocą scikit-rf
        try:
            network = rf.Network(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to load file {filename}: {str(e)}"}), 400

        if parameter == "s11" and hasattr(network, 's11'):
            s11 = network.s11
            s11.plot_s_db(ax=ax, label=f'{filename} S11')
        elif parameter == "s21" and hasattr(network, 's21'):
            s21 = network.s21
            s21.plot_s_db(ax=ax, label=f'{filename} S21')
        elif parameter == "s12" and hasattr(network, 's12'):
            s12 = network.s12
            s12.plot_s_db(ax=ax, label=f'{filename} S12')
        elif parameter == "s22" and hasattr(network, 's22'):
            s22 = network.s22
            s22.plot_s_db(ax=ax, label=f'{filename} S22')
        else:
            return jsonify({"error": f"Parameter {parameter} not found in file {filename}"}), 400

    if not ax.lines:
        return jsonify({"error": "No valid parameters to plot"}), 400

    ax.set_title("Frequency Domain")
    # Adjust the plot area to make room for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # Reduce plot width to 75%

    # Place the legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    # Adjust the figure layout to prevent clipping
    fig.subplots_adjust(right=0.85)

    plot_path = os.path.join(PLOT_FOLDER, "frequency_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return jsonify({"plot_url": f"/{plot_path}"})


# route to get the list of uploaded files
@app.route("/get-uploaded-files", methods=["GET"])
def get_uploaded_files():
    # prepare data to send to the client
    data = {
        "uploaded_files": {}
    }
    for filename, info in uploaded_files.items():
        data["uploaded_files"][filename] = {
            "parameters": info["parameters"]
        }
    return jsonify(data)

# handle plot updates for time domain
@app.route("/update-time-domain", methods=["POST"])
def update_time_domain():
    data = request.get_json()
    selections = data.get("selections")

    if not selections:
        return jsonify({"error": "No parameters selected"}), 400

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    for selection in selections:
        filename = selection.get("filename")
        parameter = selection.get("parameter")

        if filename not in uploaded_files:
            return jsonify({"error": f"File {filename} not found"}), 400

        filepath = uploaded_files[filename]["filepath"]

        # wczytaj dane za pomocą scikit-rf
        try:
            network = rf.Network(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to load file {filename}: {str(e)}"}), 400

        # wybierz odpowiedni parametr S
        if parameter == "s11" and hasattr(network, 's11'):
            s_param = network.s11
        elif parameter == "s21" and hasattr(network, 's21'):
            s_param = network.s21
        elif parameter == "s12" and hasattr(network, 's12'):
            s_param = network.s12
        elif parameter == "s22" and hasattr(network, 's22'):
            s_param = network.s22
        else:
            continue  # pomiń, jeśli parametr nie istnieje

        # oblicz odpowiedź w dziedzinie czasu
        N = len(s_param.f)
        delta_f = s_param.f[1] - s_param.f[0]
        time = np.fft.fftfreq(N, d=delta_f)
        time = np.fft.fftshift(time) * 1e9  # konwersja na nanosekundy
        s_time = np.fft.fftshift(np.fft.ifft(s_param.s[:, 0, 0]))

        # filtruj, aby zachować tylko dodatni czas
        positive_time_mask = time >= 0
        time_positive = time[positive_time_mask]
        s_time_positive = s_time[positive_time_mask]

        # wykres modułu w dziedzinie czasu
        ax.plot(time_positive, np.abs(s_time_positive), label=f"{filename} {parameter.upper()}")

    if not ax.lines:
        return jsonify({"error": "No valid parameters to plot"}), 400

    ax.set_title('Time Domain (Magnitude)')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Magnitude")

    # Adjust the plot area
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    # Place the legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    # Adjust the figure layout
    fig.subplots_adjust(right=0.85)

    plot_path = os.path.join(PLOT_FOLDER, "time_domain_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return jsonify({"plot_url": f"/{plot_path}"})

from flask import send_from_directory

@app.route("/clean", methods=["POST"])
def clean_files():
    try:
        # Delete all files in the upload folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Clear the uploaded_files dictionary
        global uploaded_files
        uploaded_files = {}

        return jsonify({"success": True, "message": "All files deleted successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error while deleting files: {str(e)}"}), 500


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


# handle plot updates for time gating
@app.route("/update-time-gating", methods=["POST"])
def update_time_gating():
    data = request.get_json()
    selections = data.get("selections")
    center = data.get("center", 7.0)
    span = data.get("span", 3.0)

    if not selections:
        return jsonify({"error": "No parameters selected"}), 400

    center = float(center)
    span = float(span)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # Increased width

    for selection in selections:
        filename = selection.get("filename")
        parameter = selection.get("parameter")

        if filename not in uploaded_files:
            return jsonify({"error": f"File {filename} not found"}), 400

        filepath = uploaded_files[filename]["filepath"]

        # load data using scikit-rf
        try:
            network = rf.Network(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to load file {filename}: {str(e)}"}), 400

        if parameter == "s21" and hasattr(network, 's21'):
            s_param = network.s21
        elif parameter == "s11" and hasattr(network, 's11'):
            s_param = network.s11
        elif parameter == "s12" and hasattr(network, 's12'):
            s_param = network.s12
        elif parameter == "s22" and hasattr(network, 's22'):
            s_param = network.s22
        else:
            continue  # skip if parameter doesn't exist

        s_param_gated = s_param.time_gate(center=center, span=span)
        s_param_gated.name = f'Gated {parameter.upper()}'

        # compute time domain responses
        N = len(s_param.f)
        delta_f = s_param.f[1] - s_param.f[0]
        time = np.fft.fftfreq(N, d=delta_f)
        time = np.fft.fftshift(time) * 1e9  # convert to nanoseconds
        s_time = np.fft.fftshift(np.fft.ifft(s_param.s[:, 0, 0]))
        s_gated_time = np.fft.fftshift(np.fft.ifft(s_param_gated.s[:, 0, 0]))

        # filter to keep only positive time
        positive_time_mask = time >= 0
        time_positive = time[positive_time_mask]
        s_time_positive = s_time[positive_time_mask]
        s_gated_time_positive = s_gated_time[positive_time_mask]

        # plot in frequency domain (magnitude)
        s_param.plot_s_mag(ax=axes[0], label=f"{filename} {parameter.upper()} Original")
        s_param_gated.plot_s_mag(ax=axes[0], label=f"{filename} {parameter.upper()} Gated")
        axes[0].set_title('Frequency Domain (Magnitude)')
        axes[0].legend()

        # plot in time domain (original vs gated)
        axes[1].plot(time_positive, np.abs(s_time_positive), label=f"{filename} {parameter.upper()} Original")
        axes[1].plot(time_positive, np.abs(s_gated_time_positive), label=f"{filename} {parameter.upper()} Gated")
        axes[1].set_title('Time Domain (Original vs Gated - Magnitude)')
        axes[1].axvspan(center - span / 2, center + span / 2, color='red', alpha=0.2, label="Gating Window")
        axes[1].legend()

        # plot in time domain (gated only)
        axes[2].plot(time_positive, np.abs(s_gated_time_positive), label=f"{filename} {parameter.upper()} Gated")
        axes[2].set_title('Time Domain (Gated Only - Magnitude)')
        axes[2].set_xlabel("Time (ns)")
        axes[2].set_ylabel("Magnitude")
        axes[2].legend()

    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    fig.subplots_adjust(right=0.85)

    plot_path = os.path.join(PLOT_FOLDER, "time_gating_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return jsonify({"plot_url": f"/{plot_path}"})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment or default to 5000
    app.run(host="0.0.0.0", port=port)  # Bind to 0.0.0.0 to allow external connections

