# Instrucciones para Configurar y Ejecutar el Proyecto

## Configuración Inicial

### Instalar el env en la carpeta principal del proyecto:
	- virtualenv -p python3 env

### Iniciar el env con:
	- .\env\Scripts\activate

### Instalar las dependencias necesarias
	- pip install -r requirements.txt

### Iniciar la base de datos de Mongodb en CMD con:
	- mongod


## BACKEND

### Pasos para Configurar y Ejecutar

### PRIMERO Debera crear o cargar la base de datos de SIGNS en MongoDB
 
### SEGUNDO Debera ejecutar el codigo data_collection.py para cargar las carpetas para las señas
	- python data_collection.py

### TERCERO Debera elegir si capturara o subira el video de las señas
- **Si desea capturarlos**
	- Ejecutar codigo capture.py
	- Se guardara la captura de frames en MP_Data
- **Si subio los videos a la carpeta VIDEOS**
	- Se pueden descargar del repositorio tambien		
	- Correr el codigo capture_videos.py
	- Se guardara la captura de frames en MP_Data_Videos
	
### CUARTO Deberas entrenar el modelo con:
 
- **Si capturo los videos**
	-  train_model.py
- **Si subio los videos**
	- train_model_videos.py

### Quinto: Evaluar y probar el modelo

- **Si capturaste las señas**

	- Se puede evaluar con matrix_confusion_model.py ESTE SUBE LA EVALUACION EN LA CARPETA MATRIX_CONFUSIONS
		- python matrix_confusion_model.py
	- Se puede probar con evaluate_model.py
		- python matrix_confusion_model.py

- *Si subiste los videos de las señas

	- Se puede evaluar con matrix_confusion_model.py ESTE SUBE LAS IMAGENES EN LA CARPETA MATRIX_CONFUSIONS_VIDEOS
		- python matrix_confusion_model_videos.py
	- Se puede probar con evaluate_model.py
		- python matrix_confusion_model_videos.py

	- RECOMENDACION: Puedes verificar el rendimiento del modelo con Tensorboard
		- tensorboard --logdir=Logs/train
		- tensorboard --logdir=Logs_Videos/train

## Frontend

### DEBERAS IRTE A LA CARPETA app
	- cd app

### DEBERAS CORRER EN LA TERMINAL app
	- python app.py

### Y LISTO!!