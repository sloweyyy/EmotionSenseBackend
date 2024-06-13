from flask import Flask, request, jsonify, send_file, render_template_string, make_response
from PIL import Image
import io
from predict import predict_with_hog, predict_without_hog  

app = Flask(__name__)

@app.route('/')
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except Exception as e:
        app.logger.error(f"Error loading index.html: {str(e)}")
        return f"Error loading index.html: {str(e)}", 500
    
@app.route('/predict_with_hog', methods=['POST'])
def predict_with_hog_route():
    if 'file' not in request.files:
        app.logger.error("No file part")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        app.logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream)
    except Exception as e:
        app.logger.error(f"Error loading image: {str(e)}")
        return jsonify({"error": f"Error loading image: {str(e)}"}), 400

    try:
        emotion, face_img = predict_with_hog(img)
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500

    if face_img is None:
        return jsonify({"emotion": emotion}), 200

    byte_arr = io.BytesIO()
    face_img.save(byte_arr, format='PNG')
    byte_arr.seek(0)

    response = make_response(send_file(byte_arr, mimetype='image/png'))
    response.headers['Emotion'] = emotion
    return response

@app.route('/predict_without_hog', methods=['POST'])
def predict_without_hog_route():
    if 'file' not in request.files:
        app.logger.error("No file part")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        app.logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream)
    except Exception as e:
        app.logger.error(f"Error loading image: {str(e)}")
        return jsonify({"error": f"Error loading image: {str(e)}"}), 400

    try:
        emotion, face_img = predict_without_hog(img)
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500

    if face_img is None:
        return jsonify({"emotion": emotion}), 200

    byte_arr = io.BytesIO()
    face_img.save(byte_arr, format='PNG')
    byte_arr.seek(0)

    response = make_response(send_file(byte_arr, mimetype='image/png'))
    response.headers['Emotion'] = emotion
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
