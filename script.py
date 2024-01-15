import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import threading

url = 'http://192.168.88.103/cam-lo.jpg'
im = None
exit_flag = False

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
