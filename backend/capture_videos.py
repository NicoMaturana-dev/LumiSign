import cv2
import os
import numpy as np
import mediapipe as mp
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, get_signs, update_status

VIDEOS_PATH = os.path.join('Videos')
DATA_PATH = os.path.join('MP_Data_Videos')
sequence_length = 60
max_sequences = 60
target_resolution = (1280, 720)

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))

def validate_saved_data(sequence_path):
    """
    Valida los archivos `.npy` guardados en una secuencia.
    """
    npy_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.npy')])
    for npy_file in npy_files:
        file_path = os.path.join(sequence_path, npy_file)
        data = np.load(file_path)

        # Validar forma esperada
        expected_shape = (1629,)
        if data.shape != expected_shape:
            print(f"Advertencia: Forma inesperada en {file_path}. Forma encontrada: {data.shape}, esperada: {expected_shape}")

        # Validar normalización
        if not np.all((data >= -1) & (data <= 1)):
            print(f"Advertencia: Valores fuera de rango en {file_path}. Rango esperado: [-1, 1]")

        # Validar valores NaN
        if np.isnan(data).any():
            print(f"Advertencia: Valores NaN detectados en {file_path}.")

def save_sequence(sequence, action_path, sequence_number):
    sequence_path = os.path.join(action_path, str(sequence_number))
    os.makedirs(sequence_path, exist_ok=True)
    for idx, frame in enumerate(sequence):
        npy_path = os.path.join(sequence_path, f"{idx}.npy")
        np.save(npy_path, frame)
    print(f"Secuencia {sequence_number} guardada en '{sequence_path}'.")
    validate_saved_data(sequence_path)

def process_action_videos(action, video_files, action_folder):
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
    sequence_counter = 1
    current_sequence = []

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        for video_file in video_files:
            video_path = os.path.join(action_folder, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error al abrir el video: {video_path}")
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = resize_frame(frame, *target_resolution)
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                keypoints = extract_keypoints(results)
                current_sequence.append(keypoints)

                if len(current_sequence) == sequence_length:
                    save_sequence(current_sequence, action_path, sequence_counter)
                    sequence_counter += 1
                    current_sequence = []

                    if sequence_counter > max_sequences:
                        cap.release()
                        return

            cap.release()

    return sequence_counter - 1

def process_videos_in_folders():
    if not os.path.exists(VIDEOS_PATH):
        print(f"La carpeta {VIDEOS_PATH} no existe.")
        return

    actions_list = get_signs()
    if not actions_list:
        print("No se encontraron acciones en la base de datos.")
        return

    # Extraer solo las palabras de las acciones
    actions = [action['word'] for action in actions_list]

    for action in actions:
        action_folder = os.path.join(VIDEOS_PATH, action)
        video_files = [f for f in os.listdir(action_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            print(f"No se encontraron videos en la carpeta '{action_folder}'.")
            continue
        
        # Procesar los videos de la acción
        processed_sequences = process_action_videos(action, video_files, action_folder)
        if processed_sequences:
            # Actualizar estado de la palabra en la base de datos
            update_status(action, True)
            print(f"✅ Se completó la captura para '{action}'.")
        else:
            print(f"⚠️ No se procesaron secuencias para la acción '{action}'.")

if __name__ == "__main__":
    process_videos_in_folders()
