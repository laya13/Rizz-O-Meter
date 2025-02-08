import cv2
import tkinter as tk
from tkinter import Label, Button, Toplevel, Text
from PIL import Image, ImageTk
import pathlib
import mediapipe as mp

cascade_path = pathlib.Path(cv2.__file__).parent.absolute()/"data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0

        # Open video source (by default this will try to open the computer webcam)
        self.vid = cv2.VideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # Button that lets the user process the image
        self.btn_process = Button(window, text="Process Image", width=50, command=self.open_popup)
        self.btn_process.pack(anchor=tk.CENTER, expand=True)

        # Create a label
        self.label = Label(self.window, text='', font=("Arial", 16))
        self.label.pack(pady=20)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def open_popup(self):
        # Create a new Toplevel window
        popup = Toplevel(self.window)
        popup.title("Instructions")
        popup.geometry("300x550")

        # Add a label to the popup window
        label = Label(popup, text="Press show button")
        label.pack(pady=10)
        label = Label(popup, text="Click on your facial features in this order:")
        label.pack(pady=10)
        label1 = Label(popup, text='1. outside of left eye in picture')
        label1.pack(pady=10)
        label2 = Label(popup, text='2. inside of left eye')
        label2.pack(pady=10)
        label3 = Label(popup, text='3. inside of right eye')
        label3.pack(pady=10)
        label4 = Label(popup, text='4. outside of right eye')
        label4.pack(pady=10)
        label5 = Label(popup, text='5. left side of nose')
        label5.pack(pady=10)
        label6 = Label(popup, text='6. right side of nose')
        label6.pack(pady=10)
        label7 = Label(popup, text='7. top of nose')
        label7.pack(pady=10)
        label8 = Label(popup, text='8. bottom of nose')
        label8.pack(pady=10)
        label9 = Label(popup, text='9. left side of mouth')
        label9.pack(pady=10)
        label10 = Label(popup, text='10. right side of mouth')
        label10.pack(pady=10)
        label11 = Label(popup, text='Press \'q\' to see your score!')
        label11.pack(pady=10)
        self.btn_process1 = Button(popup, text="Show Image", width=50, command=self.process_image)
        self.btn_process1.pack(anchor=tk.CENTER, expand=True)

    def snapshot(self):
        # Get a frame from the video source
        while True:
            ret, frame = self.vid.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = clf.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, width, height) in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (225, 225, 0), 2)

            #cv2.imshow("Faces", frame)

            if len(faces)>0: #Check if face is detected before capturing
                    x, y, width, height = faces[0]  # Get the first detected face
                    self.captured_face = frame[y:y + height, x:x + width].copy()
                    # newImage, landmarks = getLandmarks(self.captured_face)
                    # self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)))
                    # self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                    # self.vid.release()  # Release the video capture
                    return

    def process_image(self):
            self.newImage, landmarks = getLandmarks(self.captured_face)
            clicked_coordinates = self.show_image_with_clicks(self.newImage)
            score = getRatios(landmarks, clicked_coordinates)
            self.label.config(text=score)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = clf.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, width, height) in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (225, 225, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)


    def mouse_click(self, event, x, y, flags, param):
            """Handles mouse click events."""
            if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
                param["coordinates"].append((x,y)) #Store the coordinates in the dictionary
                cv2.circle(param["frame"], (x, y), 5, (0, 0, 255), -1)

    def show_image_with_clicks(self, img):
        """Displays an image and handles mouse clicks."""

        #Create a window and set the mouse callback function
        cv2.namedWindow("Image with Clicks") #Important to have a named window before setting the callback
        coordinates = [] #List to store coordinates
        data = {"frame": img, "coordinates": coordinates} #Dictionary to pass data to callback
        cv2.setMouseCallback("Image with Clicks", self.mouse_click, data) #Pass the dictionary as parameter
        features = ['outside of right eye', 'inside of right eye', 'inside of left eye',
                    'outside of left eye', 'left edge of nose', 'right edge of nose',
                    'top of nose', 'bottom of nose', 'left edge of mouth', 'right edge of mouth']
        while True:

            cv2.imshow("Image with Clicks", img) #Display the image (updated with clicks)
            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' to quit
                break
        cv2.destroyAllWindows()
        return data["coordinates"] #Return the coordinates when the loop is broken


def getLandmarks(photo):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        image = photo
        image = cv2.resize(image, (400, 400))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # MediaPipe needs RGB
        results = face_detection.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV display

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                landmarks = []
                #Landmarks (if you want them):
                for i in range(0,6): #5 landmarks by default in MediaPipe
                    landmark = detection.location_data.relative_keypoints[i]
                    x = int(landmark.x * iw)
                    y = int(landmark.y * ih)
                    cv2.circle(image, (x,y), 5, (0,0,255), -1)
                    landmarks.append((x, y))
        cv2.destroyAllWindows()
        return image, landmarks


def getRatios(landmarks, userCoordinates):
    score = 0
    ratios = []
    rightEye = landmarks[0]
    leftEye = landmarks[1]
    nose = landmarks[2]
    mouth = landmarks[3]
    rightSide = landmarks[4]
    leftSide = landmarks[5]
    width = leftSide[0] - rightSide[0]
    print(width)
    ratios.append(400/width) # face length to width -- ideally 1.5
    rightEye = userCoordinates[0:2]
    leftEye = userCoordinates[2:4]
    noseWidth = userCoordinates[4:6]
    noseLength = userCoordinates[6:8]
    mouthWidth = userCoordinates[8:10]
    eyeWidth = (rightEye[1][0]-rightEye[0][0] + leftEye[1][0] - leftEye[0][0])/2
    eyeGap = leftEye[0][0] - rightEye[1][0]
    ratios.append(eyeGap/eyeWidth) # distance between eyes vs width -- ideally 1.618
    ratios.append((noseLength[1][1] - noseLength[0][1])/(noseWidth[1][0] - noseWidth[0][0])) # nose length to width -- ideally 1.618
    ratios.append((mouthWidth[1][0] - mouthWidth[0][0])/(noseWidth[1][0] - noseWidth[0][0])) # mouth vs nose width -- golden ratio
    ratios.append(width/(mouthWidth[1][0] - mouthWidth[0][0])) # face vs mouth width -- golden ratio
    ratios.append((mouth[1] - noseLength[0][1])/(400-mouth[1])) # top of nose to mouth vs mouth to chin -- golden ratio
    for i in range(len(ratios)):
        if i == 0:
            expected = 1.5
        else:
            expected = 1.618
        score += 100 / 6 * (1 - percentDiff(ratios[i], expected))
    return score

def percentDiff(real, expected):
    return abs(real-expected)/expected

# Create a window and pass it to the Application object
App(tk.Tk(), "Tkinter and OpenCV")