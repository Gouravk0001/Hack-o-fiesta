import threading
import cv2
import cvzone
import numpy as np
import pickle
import time
import math
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, template_folder="site")

parking_lots = {
    "CarPark1.mp4": {"posfile": "CarPark1.pkl", "address": "Vallet parking phoenix palassio mall", "lat": 26.808985359151155, "lng": 81.01240011349245},
    "CarPark2.mp4": {"posfile": "CarPark2.pkl", "address": "Lullu Parking", "lat": 26.785440205615682, "lng": 80.99146827973452},
}

width, height = 115, 44
parking_lot_data = {}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  
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
            continue

        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgblur = cv2.GaussianBlur(imggray, (3, 3), 1)
        imgthreshold = cv2.adaptiveThreshold(imgblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        imgmedian = cv2.medianBlur(imgthreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgdialate = cv2.dilate(imgmedian, kernel, iterations=1)

        freespaces = checkParkSpace(imgdialate)
        parking_lot_data[video_path] = {"freespaces": freespaces, "lat": parking_lots[video_path]["lat"], "lng": parking_lots[video_path]["lng"]}
        cv2.imshow(f"Processed", img)
        cv2.waitKey(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_closest_parking', methods=['GET'])
def get_closest_parking():
    lat = float(request.args.get('lat'))
    lng = float(request.args.get('lng'))

    if not parking_lot_data:
        return jsonify({"error": "No video data available"}), 404

    closest_lot = None
    min_distance = float('inf')

    for lot, data in parking_lot_data.items():
        dist = haversine(lat, lng, data["lat"], data["lng"])
        if data["freespaces"] > 0 and dist < min_distance:
            min_distance = dist
            closest_lot = lot

    if closest_lot:
        lot_name = parking_lots[closest_lot]["address"]
        free_spaces = parking_lot_data[closest_lot]["freespaces"]
        return jsonify({"lot_name": lot_name, "free_spaces": free_spaces, "address": lot_name})
    else:
        return jsonify({"error": "No parking lot with free spaces found"}), 404

def start_video_processing():
    video_paths = ["CarPark1.mp4", "CarPark2.mp4"]
    car_park_pos_mapping = {
        "CarPark1.mp4": "CarPark1.pkl",
        "CarPark2.mp4": "CarPark2.pkl",
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
