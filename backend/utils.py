import cv2
import numpy as np
import mediapipe as mp
from pymongo import MongoClient

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    """
    Extrae keypoints de pose, rostro y manos, garantizando una longitud fija de 1629.
    """
    # Pose landmarks: 33 puntos * 3 coordenadas = 99
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 3)
    
    # Face landmarks: 468 puntos * 3 coordenadas = 1404
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    
    # Left hand landmarks: 21 puntos * 3 coordenadas = 63
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    
    # Right hand landmarks: 21 puntos * 3 coordenadas = 63
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    # Concatenar keypoints
    keypoints = np.concatenate([pose, face, lh, rh])

    # Validar longitud
    expected_length = 1629
    if keypoints.shape[0] != expected_length:
        print(f"Advertencia: Longitud inesperada ({keypoints.shape[0]}). Rellenando...")
        keypoints = np.zeros(expected_length)

    # Normalizar valores a [-1, 1]
    keypoints = np.clip((keypoints - 0.5) * 2, -1, 1)

    return keypoints

def get_signs():
    """
    Obtiene las palabras (acciones) desde la base de datos MongoDB.
    """
    try:
        client = MongoClient('localhost', 27017)
        database = client['actions']
        collection = database['signs']

        # Consulta a MongoDB: Obtener solo las palabras ('word')
        signs = collection.find({}, {'_id': 0, 'word': 1, 'status': 1})
        return [{'word': sign['word'], 'status': sign.get('status', False)} for sign in signs]
    except Exception as ex:
        print(f"Error al conectar a MongoDB: {ex}")
        return []
    finally:
        print("Conexión finalizada a MongoDB.")


#Función para actualizar el estado de la Acción
def update_status(action, status):
    client = MongoClient('localhost', 27017)
    database = client['actions']
    collection = database['signs']
    collection.update_one({'word': action}, {'$set': {'status': status}})
    print(f"Estado actualizado para '{action}' a {'capturado' if status else 'no capturado'}.")
