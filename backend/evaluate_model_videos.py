import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import threading
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, get_signs

# Cargar acciones desde MongoDB
actions_data = get_signs()
if not actions_data:
    print("No se encontraron acciones en la base de datos. Finalizando.")
    exit()

# Filtrar palabras con status=True
actions = [action['word'] for action in actions_data if action['status']]
if not actions:
    print("No hay palabras con status=True para entrenar. Finalizando.")
    exit()

# Generar colores para cada acción
colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(actions))]

# Cargar el modelo entrenado
model = tf.keras.models.load_model('train_model_videos.keras')

# Parámetros de visualización
threshold = 0.5

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def predict_action(sequence, model):
    sequence = np.expand_dims(sequence, axis=0)
    prediction = model.predict(sequence)[0]
    return prediction

def speak_text(text):
    """
    Habla el texto proporcionado usando un hilo separado.
    """
    def run():
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Velocidad de la voz
        engine.setProperty('volume', 0.9)  # Volumen de la voz
        engine.say(text)
        engine.runAndWait()
    
    # Crear y ejecutar un hilo separado para evitar bloqueos
    threading.Thread(target=run).start()

def run_evaluation():
    cap = cv2.VideoCapture(0)
    sequence = []
    sentence = []
    predictions = []
    prediction = None  # Inicializar prediction

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Realizar detecciones y extraer keypoints
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-60:]  # Mantener solo los últimos 60 frames

            # Predecir solo si tenemos 60 frames en la secuencia
            if len(sequence) == 60:
                prediction = predict_action(sequence, model)
                action = actions[np.argmax(prediction)]
                predictions.append(np.argmax(prediction))

                # Manejar la lógica de la secuencia y oración
                if np.unique(predictions[-10:])[0] == np.argmax(prediction):
                    if prediction[np.argmax(prediction)] > threshold:
                        if len(sentence) > 0:
                            if action != sentence[-1]:
                                sentence.append(action)
                                speak_text(action)  # Hablar la nueva palabra predicha
                        else:
                            sentence.append(action)
                            speak_text(action)  # Hablar la nueva palabra predicha
            
            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualizar probabilidades solo si se realiza una predicción
            if prediction is not None:
                image = prob_viz(prediction, actions, image, colors)

            cv2.rectangle(image, (0, 0), (1280, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Mostrar en pantalla
            cv2.imshow('LumiSign: Traductor de LSCH', image)

            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_evaluation()
