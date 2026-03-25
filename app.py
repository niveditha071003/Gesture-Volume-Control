from flask import Flask, render_template, Response, jsonify
import cv2, mediapipe as mp, math, numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

running = False
current_volume = 0
finger_distance = 0   

# Audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol, _ = volume_interface.GetVolumeRange()

# Hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands()
draw = mp.solutions.drawing_utils

# Function to generate video frames
def generate_frames():
    global running, current_volume, finger_distance

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR to RGB for Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw hand landmarks and calculate volume
        if running and result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                draw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

                lm = hand.landmark
                x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
                x2, y2 = int(lm[8].x * w), int(lm[8].y * h)
                # draw circles on fingers
                cv2.circle(frame, (x1, y1), 8, (0, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 8, (0, 255, 255), -1)

                # draw line between thumb and index
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # distance
                dist = math.hypot(x2 - x1, y2 - y1)
                finger_distance = int(dist)

                # volume
                current_volume = int(np.interp(dist, [20, 200], [0, 100]))
                vol = np.interp(current_volume, [0, 100], [minVol, maxVol])
                volume_interface.SetMasterVolumeLevel(vol, None)

        _, buffer = cv2.imencode('.jpg', frame) 
        frame = buffer.tobytes()

        # Yield the frame in a format suitable for streaming
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    #

 
 # Flask routes
@app.route('/')     
def index():
    return render_template("index.html")


@app.route('/start')
def start():
    global running
    running = True
    return jsonify({"status": "started"})


@app.route('/stop')
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/volume')
def volume():
    return jsonify({
        "volume": current_volume,
        "distance": finger_distance
    })


if __name__ == "__main__":
    app.run(debug=True)