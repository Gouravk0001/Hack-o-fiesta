import cv2
import pickle
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

width, height = 115, 44

def get_pickle_file(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return base_name + '.pkl'

def load_positions(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except:
        return []

def mouseClick(events, x, y, flags, param):
    global poslist, pickle_file
    if events == cv2.EVENT_LBUTTONDOWN:
        poslist.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(poslist):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                poslist.pop(i)
                break
    with open(pickle_file, 'wb') as f:
        pickle.dump(poslist, f)

def select_image():
    Tk().withdraw()
    image_path = askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    return image_path

image_path = select_image()
if not image_path:
    print("No image selected. Exiting.")
    exit()

pickle_file = get_pickle_file(image_path)
poslist = load_positions(pickle_file)

while True:
    img = cv2.imread(image_path)

    for pos in poslist:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (0, 255, 0), 2)

    cv2.imshow('image', img)
    cv2.setMouseCallback("image", mouseClick)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
