import cv2
import os
import numpy as np
from time import time

# Set the path to the 'Data' folder containing subfolders with face images
dataRuta = 'Data'
listaData = os.listdir(dataRuta)

# Initialize lists to store face images and their corresponding ids (labels)
rostrosData = []
ids = []
id_actual = 0

# Record the start time of the face data collection process
tiempoInicial = time()

# Loop through each subfolder in the 'Data' directory
for fila in listaData:
    # Build the complete path to the current subfolder
    rutacompleta = dataRuta + '/' + fila
    print('Iniciando lectura de carpeta:', fila)

    # Record the start time of reading images from the current subfolder
    tiempoinicialLecturaCarpeta = time()

    # Loop through each image in the current subfolder
    for archivo in os.listdir(rutacompleta):
        print('Imagenes:', fila + '/' + archivo)
        # Read the face image in grayscale
        rostro = cv2.imread(rutacompleta + '/' + archivo, 0)
        # Append the face image to the list 'rostrosData'
        rostrosData.append(rostro)
        # Append the current 'id_actual' (label) to the list 'ids'
        ids.append(id_actual)

    # Increment the 'id_actual' for the next subfolder (label for the next set of images)
    id_actual += 1

    # Record the end time of reading images from the current subfolder
    tiempofinalLecturaCarpeta = time()
    tiempoTotalLecturaCarpeta = tiempofinalLecturaCarpeta - tiempoinicialLecturaCarpeta
    print('Tiempo total lectura carpeta:', tiempoTotalLecturaCarpeta)

# Record the end time of the entire face data collection process
tiempofinalLectura = time()
tiempoTotalLectura = tiempofinalLectura - tiempoInicial
print('Tiempo total lectura:', tiempoTotalLectura)

# Create an instance of EigenFaceRecognizer for training
entrenamientoEigenFaceRecognizer = cv2.face.EigenFaceRecognizer_create()
print('Iniciando el entrenamiento... espere')

# Record the start time of the training process
tiempoinicialEntrenamiento = time()

# Train the EigenFaceRecognizer using the collected face images and their corresponding ids
entrenamientoEigenFaceRecognizer.train(np.array(rostrosData), np.array(ids))

# Record the end time of the training process
tiempofinalEntrenamiento = time()

# Calculate the total time taken for the training process
tiempoTotalEntrenamiento = tiempofinalEntrenamiento - tiempoinicialEntrenamiento
print('Tiempo entrenamiento total:', tiempoTotalEntrenamiento)

# Save the trained EigenFaceRecognizer model to a file named 'EntrenamientoEigenFaceRecognizer.xml'
entrenamientoEigenFaceRecognizer.write('EntrenamientoEigenFaceRecognizer.xml')
print('Entrenamiento concluido')
