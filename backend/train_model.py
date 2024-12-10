import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard, EarlyStopping
from datetime import datetime
from utils import get_signs
from keras.regularizers import l2
from keras.layers import BatchNormalization
from matrix_confusion_model import evaluate_model_with_metrics 

#Ruta para exportar la data
DATA_PATH = os.path.abspath('MP_Data')  # Ruta absoluta para evitar conflictos
LOGS_PATH = os.path.abspath('Logs')  # Ruta absoluta para los logs
TEST_DATA_PATH = os.path.abspath('Test_Data')  # Carpeta para datos de prueba

# Crear la carpeta Test_Data si no existe
os.makedirs(TEST_DATA_PATH, exist_ok=True)

# Obtener acciones desde la base de datos (solo las que tienen status=True)
actions_data = get_signs()
if not actions_data:
    print("No se encontraron acciones en la base de datos. Finalizando.")
    exit()

# Filtrar palabras con status=True
actions = [action['word'] for action in actions_data if action['status']]
if not actions:
    print("No hay palabras con status=True para entrenar. Finalizando.")
    exit()

print(f"Acciones seleccionadas para entrenamiento: {actions}")

# Crear mapa de etiquetas
label_map = {label: num for num, label in enumerate(actions)}

# Configuración de secuencias dinámicas
max_sequence_length = 60  # Longitud máxima esperada para las secuencias
sequences, labels = [], []

def validate_normalization(data):
    """
    Valida si los datos están normalizados en el rango [-1, 1], y revisa NaN o Inf.
    """
    if not np.all((data >= -1) & (data <= 1)):
        print("Advertencia: Datos fuera del rango esperado [-1, 1]. Verifica la normalización.")
    if np.isnan(data).any():
        print("Error: Datos contienen valores NaN.")
    if np.isinf(data).any():
        print("Error: Datos contienen valores infinitos.")

def normalize_sequence(sequence):
    sequence = np.array(sequence)
    sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))  # Escala a [0, 1]
    sequence = sequence * 2 - 1  # Escala a [-1, 1]
    return sequence

# Extraer características y etiquetas
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"Advertencia: No se encontraron datos para la acción '{action}'.")
        continue

    for sequence in np.array(os.listdir(action_path)).astype(int):
        sequence_path = os.path.join(action_path, str(sequence))
        frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.npy')])

        # Verificar si hay suficientes frames para la secuencia
        if len(frame_files) < max_sequence_length:
            print(f"Advertencia: Secuencia {sequence} de '{action}' tiene menos frames de lo esperado.")
            continue

        # Cargar frames hasta la longitud máxima permitida
        window = []
        for frame_file in frame_files[:max_sequence_length]:
            res = np.load(os.path.join(sequence_path, frame_file))
            res = normalize_sequence(res)
            window.append(res)

        validate_normalization(np.array(window))

        # Asegurarse de que todas las secuencias tengan la misma longitud
        if len(window) == max_sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

# Convertir datos a arrays de NumPy
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Data Augmentation: Añadir ruido a las secuencias
def augment_sequence(sequence):
    noise = np.random.normal(0, 0.01, sequence.shape)  # Ruido pequeño
    return sequence + noise

X_augmented = np.array([augment_sequence(seq) for seq in X])
X = np.concatenate([X, X_augmented])
y = np.concatenate([y, y])

print(f"Datos después de la augmentación: {X.shape[0]} secuencias, {X.shape[1]} frames por secuencia, {X.shape[2]} características por frame.")

# Diagnóstico de datos
print("\n=== Diagnóstico de Normalización ===")
print("Datos normalizados - Valores mínimos y máximos:")
print(f"Min: {np.min(X)}, Max: {np.max(X)}")
print(f"Media: {np.mean(X)}, Desviación Estándar: {np.std(X)}")
print(f"\n¿Hay valores NaN? {np.isnan(X).any()}")
print(f"¿Hay valores infinitos? {np.isinf(X).any()}")

print("\n=== Diagnóstico de Longitudes de Secuencias ===")
unique_lengths = set([len(seq) for seq in X])
print(f"Tamaño único de secuencias: {unique_lengths}")

print("\n=== Diagnóstico de Clases ===")
unique_classes, class_counts = np.unique(np.argmax(y, axis=1), return_counts=True)
print(f"Distribución de clases: {dict(zip(unique_classes, class_counts))}")

# Dividir en conjuntos de entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)

# Guardar los datos de prueba en la carpeta Test_Data
np.save(os.path.join(TEST_DATA_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(TEST_DATA_PATH, 'y_test.npy'), y_test)
np.save(os.path.join(TEST_DATA_PATH, 'actions.npy'), np.array(actions))
print(f"Datos de prueba guardados en {TEST_DATA_PATH}")

# Construir modelo simplificado para diagnóstico inicial
model = Sequential()
model.add(LSTM(128, return_sequences=False, activation='relu', input_shape=(60, 1629)))
model.add(BatchNormalization())  # Normalización después de la LSTM
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))  # Dropout en capa Dense
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))  # Dropout en capa Dense
model.add(Dense(len(actions), activation='softmax'))

# Compilar modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Configurar carpeta de logs
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(LOGS_PATH, 'train', timestamp)
os.makedirs(log_dir, exist_ok=True)
print(f"Logs serán guardados en: {log_dir}")

# Configurar callbacks
tb_callback = TensorBoard(log_dir=log_dir)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
print("Iniciando el entrenamiento del modelo...")
model.fit(X_train, y_train, epochs=2000, batch_size=32, validation_split=0.1, callbacks=[tb_callback, early_stopping])

# Guardar el modelo
model.save('train_model.keras')
print(f"Entrenamiento completado. Modelo guardado como 'train_model.keras'. Logs en {log_dir}.")

# Evaluar en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida en prueba: {test_loss:.4f}, Precisión en prueba: {test_accuracy:.4f}")
