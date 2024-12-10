import os
import numpy as np
from pymongo import MongoClient
from utils import get_signs

# Directorio donde se crearan las carpetas
DATA_PATH = os.path.join('MP_Data')

# Configuración
no_sequences = 30
sequence_length = 30

# Crear carpetas necesarias para las acciones y capturas
def setup_folders():
    # Verificar si la carpeta raíz existe
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # Obtener las acciones desde MongoDB
    actions_list = get_signs()
    if not actions_list:
        print("No se encontraron acciones en la base de datos.")
        return

    # Extraer solo las palabras de las acciones
    actions_words = [action['word'] for action in actions_list]

    # Identificar acciones que ya tienen carpetas
    existing_actions = {folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))}
    new_actions = [action for action in actions_words if action not in existing_actions]

    # Crear carpetas solo para las nuevas acciones
    for action in new_actions:
        action_path = os.path.join(DATA_PATH, action)
        os.makedirs(action_path)  # Crear carpeta de la acción
        print(f"Creando carpeta para nueva acción: {action}")

        # Crear carpetas para las capturas de cada secuencia
        for sequence in range(1, no_sequences + 1):
            sequence_path = os.path.join(action_path, str(sequence))
            os.makedirs(sequence_path)
            print(f"Creando carpeta para secuencia: {sequence} de {action}")

    if not new_actions:
        print("No hay nuevas acciones para agregar.")
    else:
        print(f"Se agregaron nuevas acciones: {', '.join(new_actions)}")


if __name__ == "__main__":
    # Configurar las carpetas con base en los datos obtenidos de MongoDB
    setup_folders()


