import cv2
import os
import numpy as np
import mediapipe as mp
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, get_signs, update_status

#Ruta para guardar las secuencias
DATA_PATH = os.path.join('MP_Data')

# Configuración
no_sequences = 30
sequence_length = 60

# Función para mostrar texto con barra negra al fondo en la pantalla
def display_message(image, message, color=(255, 255, 255), size=2, thickness=3, bg_color=(0, 0, 0)):
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2

    top_left = (text_x - 10, text_y - text_size[1] - 10)
    bottom_right = (text_x + text_size[0] + 10, text_y + 10)

    cv2.rectangle(image, top_left, bottom_right, bg_color, -1)

    cv2.putText(image, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
    return image

def collect_data():
    """
    Recolecta datos para palabras que aún no han sido capturadas y actualiza su estado en la base de datos.
    """
    # Obtener acciones con su estado desde la base de datos
    actions = get_signs()
    print('Estas son las acciones:', actions)
    
    # Filtrar palabras que necesitan ser capturadas y las que ya están capturadas
    actions_to_capture = [action['word'] for action in actions if not action['status']]
    actions_completed = [action['word'] for action in actions if action['status']]

    # Validación: si no hay palabras pendientes de captura
    if not actions_to_capture:
        print("\n🎉 Todas las palabras ya han sido capturadas. Puedes proceder a entrenar el modelo.")

        # Ofrecer re-capturar palabras
        while True:
            user_input = input("¿Deseas re-capturar alguna palabra en particular? (S/N): ").strip().lower()
            if user_input == 'n':
                print("\n✅ Finalizando el proceso. Puedes proceder a entrenar el modelo.")
                return
            elif user_input == 's':
                print("\nPalabras disponibles para re-capturar:")
                for i, action in enumerate(actions_completed, start=1):
                    print(f"{i}. {action}")
                try:
                    selection = int(input("Selecciona el número de la palabra que deseas re-capturar: "))
                    if 1 <= selection <= len(actions_completed):
                        actions_to_capture.append(actions_completed[selection - 1])
                        break
                    else:
                        print("\n❌ Número inválido. Intenta nuevamente.")
                except ValueError:
                    print("\n❌ Entrada inválida. Por favor, ingresa un número válido.")
            else:
                print("\n❌ Entrada no válida. Por favor, responde con 'S' o 'N'.")

    # Configurar captura de datos
    print("\n📹 Iniciando la colección de datos para las palabras:")
    for action in actions_to_capture:
        print(f"- {action}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("\n❌ Error: No se pudo acceder a la cámara.")
        return

    # Captura de datos
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions_to_capture:
            print(f"\n⚙️ Procesando: {action}")
            for sequence in range(1, no_sequences + 1):
                print(f"   Secuencia {sequence}/{no_sequences}")
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("Error al capturar la imagen.")
                        continue

                    # Realizar detecciones
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Mostrar texto en pantalla
                    if frame_num == 0:
                        image = display_message(image, f"Capturando: {action}", size=1.5, color=(255, 255, 255), bg_color=(0, 0, 0))
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)

                    # Dibujar información
                    cv2.rectangle(image, (10, 10), (10 + 400, 60), (0, 0, 0), -1)
                    cv2.putText(image, f'{action} - Secuencia {sequence}', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                    # Exportar keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                    np.save(npy_path, keypoints)

                    # Salir si se presiona 'q'
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("\n❌ Captura interrumpida por el usuario.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

            # Actualizar estado de la palabra en la base de datos
            update_status(action, True)
            print(f"✅ Se completó la captura para '{action}'.")

            # Preguntar si el usuario desea continuar
            user_input = input(f"\n¿Deseas continuar con la siguiente palabra? (S/N): ").strip().lower()
            if user_input != 's':
                print("\n✅ Finalizando la recolección de datos.")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Colección de datos completada.")


if __name__ == "__main__":
    collect_data()
