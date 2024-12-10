import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from utils import extract_keypoints, get_signs, mediapipe_detection
import mediapipe as mp
import pyttsx3 
import threading 

app = Flask(__name__)

def speak_text(text):
    """
    Habla el texto proporcionado usando un hilo separado.
    """
    def run():
        # Inicializar el motor de voz dentro del hilo
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Velocidad de la voz
        engine.setProperty('volume', 0.9)  # Volumen de la voz
        engine.say(text)
        engine.runAndWait()
    
    # Crear y ejecutar un hilo separado
    threading.Thread(target=run).start()

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('../backend/train_model.keras')
mp_holistic = mp.solutions.holistic

# Obtener las acciones activas desde la base de datos
actions_data = get_signs()
actions = [action['word'] for action in actions_data if action['status']]

@app.route('/')
def index():
    """
    Renderiza la página principal.
    """
    return render_template('index.html', data={'titulo': 'LumiSign', 'signs': actions})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('frames')
        if not data or len(data) < 20:  # Cambiar el mínimo requerido según sea necesario
            return jsonify({'error': 'No se recibieron suficientes frames'}), 400

        sequence = []
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for encoded_frame in data:
                decoded_data = base64.b64decode(encoded_frame)
                np_image = np.frombuffer(decoded_data, dtype=np.uint8)
                image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

                _, results = mediapipe_detection(image, holistic)
                keypoints = extract_keypoints(results)

                if keypoints is not None and not np.all(keypoints == 0):
                    sequence.append(keypoints)

        # Completar hasta 60 frames con ceros si es necesario
        while len(sequence) < 60:
            sequence.append(np.zeros(1629))  # Rellenar con arrays de ceros

        # Tomar solo los últimos 60 frames si hay más
        sequence = sequence[-60:]

        sequence_array = np.expand_dims(np.array(sequence), axis=0)
        prediction = model.predict(sequence_array)[0]

        probabilities = prediction.tolist()
        action = actions[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Hablar la palabra predicha en un hilo separado
        speak_text(action)

        return jsonify({
            'action': action,
            'confidence': round(float(confidence), 4),
            'probabilities': probabilities,
            'actions': actions
        })
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({'error': 'Ocurrió un error en el servidor'}), 500


if __name__ == '__main__':
    app.run(debug=True)