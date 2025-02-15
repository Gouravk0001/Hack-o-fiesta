from flask import Flask, render_template, jsonify
import threading
import cv2
import cvzone
import numpy as np
import pickle
import time

app = Flask(__name__, template_folder='site')

width, height = 115, 44
parking_lot_data = {}
parking_lot_addresses = {
    "CarPark1.mp4": "123 Main St, Cityville",
    "CarPark2.mp4": "456 Elm St, Townsville",
}

def process_video(video_path, car_park_pos_file):
    with open(car_park_pos_file, 'rb') as f:
        poslist = pickle.load(f)

    vid = cv2.VideoCapture(video_path)

    def checkParkSpace(imgPro):
        freespaces = 0
        for pos in poslist:
            x, y = pos
            cropimg = imgPro[y:y + height, x:x + width]
            count = cv2.countNonZero(cropimg)
            if count < 850:
                freespaces += 1
        return freespaces

    while True:
        success, img = vid.read()
        if not success:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, img = vid.read()

        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgblur = cv2.GaussianBlur(imggray, (3, 3), 1)
        imgthreshold = cv2.adaptiveThreshold(imgblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        imgmedian = cv2.medianBlur(imgthreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgdialate = cv2.dilate(imgmedian, kernel, iterations=1)

        freespaces = checkParkSpace(imgdialate)
        parking_lot_data[video_path] = freespaces

        cv2.waitKey(1)

@app.route('/')
def index():
    if not parking_lot_data:
        return "No video data available", 404

    max_lot = max(parking_lot_data, key=parking_lot_data.get)
    max_spaces = parking_lot_data[max_lot]
    lot_address = parking_lot_addresses.get(max_lot, "Address not available")
    
    return render_template('index.html', lot_name=max_lot, free_spaces=max_spaces, address=lot_address)

@app.route('/get_parking_data')
def get_parking_data():
    if not parking_lot_data:
        return jsonify({"error": "No video data available"}), 404

    max_lot = max(parking_lot_data, key=parking_lot_data.get)
    max_spaces = parking_lot_data[max_lot]
    lot_address = parking_lot_addresses.get(max_lot, "Address not available")
    
    return jsonify({"lot_name": max_lot, "free_spaces": max_spaces, "address": lot_address})

def start_video_processing():
    video_paths = ["CarPark1.mp4"]
    car_park_pos_mapping = {
        "CarPark1.mp4": "CarPark1.pkl",
    }

    threads = []
    for video_path in video_paths:
        car_park_pos_file = car_park_pos_mapping.get(video_path)
        if car_park_pos_file:
            thread = threading.Thread(target=process_video, args=(video_path, car_park_pos_file))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    threading.Thread(target=start_video_processing, daemon=True).start()
    app.run(debug=True)
