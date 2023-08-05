import cv2 as cv
import os
import imutils

# Define the path to the data directory where training images are stored
dataRuta = 'Data'
# Get the list of files in the data directory
listaData = os.listdir(dataRuta)

# Load the pre-trained EigenFaceRecognizer model
entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer.read('EntrenamientoEigenFaceRecognizer.xml')

# Load the Haar cascade classifier for face detection
ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start capturing video from a file (e.g., "larreta.mp4")
camara = cv.VideoCapture('videoplayback.mp4')

while True:
    # Read a frame from the video
    respuesta, captura = camara.read()
    if not respuesta:
        break

    # Resize the captured frame to a width of 640 pixels to speed up processing
    captura = imutils.resize(captura, width=640)
    # Convert the frame to grayscale for face detection
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade classifier
    caras = ruidos.detectMultiScale(grises, scaleFactor=1.3, minNeighbors=8)

    for (x, y, e1, e2) in caras:
        # Crop the detected face region
        rostrocapturado = grises[y:y + e2, x:x + e1]
        # Resize the cropped face to 160x160 pixels for prediction
        rostrocapturado = cv.resize(rostrocapturado, (160, 160), interpolation=cv.INTER_CUBIC)

        # Perform face prediction using the pre-trained model
        resultado = entrenamientoEigenFaceRecognizer.predict(rostrocapturado)

        # Display the predicted label on the captured frame
        cv.putText(captura, '{}'.format(resultado), (x, y - 5), 1, 1.3, (0, 255, 0), 1, cv.LINE_AA)

        if resultado[1] < 6000:
            # If the prediction confidence is below 8000, display the recognized person's name
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x, y - 20), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)  # Draw a blue rectangle around the face
        else:
            # If the prediction confidence is above 8000, display "Not found"
            cv.putText(captura, "No encontrado", (x, y - 20), 2, 0.7, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)  # Draw a blue rectangle around the face

    # Show the captured frame with the results
    cv.imshow("Resultados", captura)

    # Exit the loop if the 's' key is pressed
    if cv.waitKey(1) == ord('s'):
        break

# Release the camera and close all windows
camara.release()
cv.destroyAllWindows()
