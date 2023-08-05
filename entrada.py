import cv2 as cv
import imutils
from pathlib import Path

def detect_faces_and_save(video_path, output_dir, model_name, max_images=300):
    # Create output directory if it doesn't exist
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize video capture
    camara = cv.VideoCapture(video_path)

    # Load HaarCascade classifier for face detection
    ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Variable to keep track of saved images
    saved_images = 0

    while True:
        # Read the next frame from the video
        response, frame = camara.read()

        if not response:
            # Break the loop if there are no more frames in the video
            break

        # Resize the frame for faster processing
        frame = imutils.resize(frame, width=640)

        # Convert the frame to grayscale for face detection
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame using HaarCascade
        faces = ruidos.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face and save the images
        for (x, y, w, h) in faces:
            # Check the size of the detected face region
            min_face_size = 60  # Adjust this value as needed
            if w >= min_face_size and h >= min_face_size:
                # Extract the face region of interest (ROI) and resize it
                face_roi = frame[y:y + h, x:x + w]
                face_roi = cv.resize(face_roi, (160, 160), interpolation=cv.INTER_CUBIC)

                # Generate a unique filename with a sequential ID
                image_filename = output_path / f'imagen_{saved_images}.jpg'

                # Save the image with the unique filename
                cv.imwrite(str(image_filename), face_roi)
                saved_images += 1

                # Draw a green rectangle around the detected face on the original frame
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the processed frame with rectangles around the faces
        cv.imshow("Face Detection", frame)

        # Break the loop if the specified number of images is reached
        if saved_images >= max_images:
            break

        # Break the loop if 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    camara.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # Set the input video file, output directory, and model name
    video_file = 'milei.mp4'
    output_directory = 'Data'
    model_name = 'Milei'

    # Call the detect_faces_and_save function with the specified parameters
    detect_faces_and_save(video_file, output_directory, model_name)
