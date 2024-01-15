import cv2
import cvlib as cv
import numpy as np
from cvlib.object_detection import draw_bbox
import threading
import urllib.request
import firebase_admin
from firebase_admin import credentials, db

url = 'http://192.168.43.56/cam-hi.jpg'
im = None
exit_flag = False

cred = credentials.Certificate('sampahmas-3a4f0-firebase-adminsdk-jgrow-b06954ec1f.json')  
firebase_admin.initialize_app(cred, {'databaseURL': 'https://sampahmas-3a4f0-default-rtdb.firebaseio.com/'})  
ref = db.reference('/object_count')  

def send_to_firebase(label):
    data = ref.get()
    if data:
        if label == 'bottle':
            ref.update({'jumlah': data.get('jumlah', 0) + 1})
            ref.update({'bottle': "true" })
        else:
            ref.update({'non_bottle': data.get('non_bottle', 0)})
            ref.update({'bottle': "false" })
    else:
        if label == 'bottle':
            ref.update({'bottle': 1, 'non_bottle': 0})
        else:
            ref.update({'bottle': 0, 'non_bottle': 1})

def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while not exit_flag:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        cv2.imshow('live transmission', im)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while not exit_flag:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        bbox, label, conf = cv.detect_common_objects(im)
        im = draw_bbox(im, bbox, label, conf)

        cv2.imshow('detection', im)

        # Check if 'bottle' is detected
        if 'bottle' in label:
            send_to_firebase('bottle')
        else:
            send_to_firebase('non_bottle')

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Started")
    
    # Start two threads
    thread1 = threading.Thread(target=run1)
    thread2 = threading.Thread(target=run2)
    
    thread1.start()
    thread2.start()
    
    # Wait for both threads to finish or for 'q' key to be pressed
    thread1.join()
    thread2.join()
