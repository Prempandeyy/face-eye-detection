from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load Haar Cascades for multiple features
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades_mcs_nose.xml') 
mouth_cascade = cv2.CascadeClassifier('haarcascades_mcs_mouth.xml')  
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
hand_cascade = cv2.CascadeClassifier('haarcascades/hand.xml') 

def generater_func():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect features
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
            hands = hand_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw rectangles around the features and label them
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # Detect eyes within the face
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(frame, 'Eye', (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Detect nose within the face
                nose = nose_cascade.detectMultiScale(roi_gray, 1.1, 4)
                for (nx, ny, nw, nh) in nose:
                    cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 255, 0), 2)
                    cv2.putText(frame, 'Nose', (x + nx, y + ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Detect mouth within the face
                mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 20)
                for (mx, my, mw, mh) in mouth:
                    # Adjust the position to avoid detecting above the nose as a mouth
                    if y + my > y + h / 2:
                        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                        cv2.putText(frame, 'Mouth', (x + mx, y + my + mh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Detect body
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(frame, 'Body', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # Detect hands
            for (x, y, w, h) in hands:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, 'Hand', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('stream.html')

@app.route('/video')
def video():
    return Response(generater_func(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
