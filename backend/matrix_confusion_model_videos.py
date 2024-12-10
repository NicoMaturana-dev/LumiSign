import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from utils import get_signs

# Ruta para los datos de prueba
TEST_DATA_PATH = os.path.abspath('Test_Data_Videos')  # Carpeta donde están los datos de prueba

def plot_individual_confusion_matrices(cm, actions, output_dir='Matrix_Confusions_Videos'):
    """
    Guarda matrices de confusión individuales como archivos de imagen.
    """
    os.makedirs(output_dir, exist_ok=True)  # Crear el directorio si no existe

    for idx, matrix in enumerate(cm):
        plt.figure(figsize=(6, 5))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión para '{actions[idx]}'")
        plt.colorbar()

        classes = ['No', 'Yes']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # Anotaciones en las celdas
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, matrix[i, j], horizontalalignment="center", 
                         color="white" if matrix[i, j] > matrix.max() / 2 else "black")

        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        # Guardar la gráfica en un archivo
        filename = os.path.join(output_dir, f"confusion_matrix_{actions[idx]}.jpg")
        plt.savefig(filename)
        plt.close()  # Cierra la figura para evitar consumir demasiada memoria

        print(f"Matriz de confusión para '{actions[idx]}' guardada en {filename}.")

def evaluate_model_with_metrics(X_test, y_test, model, actions):
    """
    Evalúa el modelo en un conjunto de prueba y calcula las métricas, incluidas las matrices de confusión.
    """
    # Generar predicciones
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Matrices de confusión multilabel
    cm = multilabel_confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("\n=== Precisión Total ===")
    print(f"Precisión: {accuracy:.2f}")

    print("\n=== Reporte de Clasificación ===")
    print(classification_report(y_true, y_pred, target_names=actions))

    # Graficar matrices de confusión individuales
    plot_individual_confusion_matrices(cm, actions)

# Cargar los datos de prueba desde la carpeta Test_Data
X_test = np.load(os.path.join(TEST_DATA_PATH, 'X_test.npy'))
y_test = np.load(os.path.join(TEST_DATA_PATH, 'y_test.npy'))
actions = np.load(os.path.join(TEST_DATA_PATH, 'actions.npy')).tolist()  # Convertir a lista

# Cargar modelo entrenado
model = tf.keras.models.load_model('train_model_videos.keras')

# Evaluar el modelo
evaluate_model_with_metrics(X_test, y_test, model, actions)
