# tracker.py
import numpy as np
import cv2

class ObjectHasher:
    def __init__(self, threshold=20, size=8, max_track_frame=10, radius_tracker=5):
        self.threshold = 20
        self.size = 8
        self.max_track_frame = 10
        self.radius_tracker = 5

    def getCenter(self, xmin, ymin, xmax, ymax):
        x_center = int((xmin + xmax)/2)
        y_center = int((ymin+ymax)/2)
        return (x_center, y_center)


    def getObjectId(self, image_np, xmin, ymin, xmax, ymax, hamming_dict={}):
        croppedImage = self.getCropped(image_np,int(xmin*0.8), int(ymin*0.8), int(xmax*0.8), int(ymax*0.8))
        croppedImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)

        resizedImage = self.resize(croppedImage, self.size)

        hash = self.getHash(resizedImage)
        center = self.getCenter(xmin*0.8, ymin*0.8, xmax*0.8, ymax*0.8)

        # hamming_dict = self.createHammingDict(hash, center, hamming_dict)
        hamming_dict = self.getObjectCounter(hash, hamming_dict)
        return hash, hamming_dict


    def getCropped(self, image_np, xmin, ymin, xmax, ymax):
        return image_np[ymin:ymax, xmin:xmax]

    def resize(self, cropped_image, size=8):
        resized = cv2.resize(cropped_image, (size+1, size))
        return resized

    def getHash(self, resized_image):
        diff = resized_image[:, 1:] > resized_image[:, :-1]
        # convert the difference image to a hash
        dhash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        return int(np.array(dhash, dtype="float64"))

    def hamming(self, hashA, hashB):
        # compute and return the Hamming distance between the integers
        return bin(int(hashA) ^ int(hashB)).count("1")

    def createHammingDict(self, dhash, center, hamming_dict):
        centers = []
        matched = False
        matched_hash = dhash
        # matched_classid = classid

        if hamming_dict.__len__() > 0:
            if hamming_dict.get(dhash):
                matched = True

            else:
                for key in hamming_dict.keys():

                    hd = self.hamming(dhash, key)

                    if(hd < self.threshold):
                        centers = hamming_dict.get(key)
                        if len(centers) > self.max_track_frame:
                            centers.pop(0)
                        centers.append(center)
                        del hamming_dict[key]
                        hamming_dict[dhash] = centers
                        matched = True
                        break

        if not matched:
            centers.append(center)
            hamming_dict[dhash] = centers

        return  hamming_dict

    def getObjectCounter(self, dhash, hamming_dict):
        matched = False
        matched_hash = dhash
        lowest_hamming_dist = self.threshold
        object_counter = 0

        if len(hamming_dict) > 0:
            if dhash in hamming_dict:
                lowest_hamming_dist = 0
                matched_hash = dhash
                object_counter = hamming_dict.get(dhash)
                matched = True

            else:
                for key in hamming_dict.keys():
                    hd = self.hamming(dhash, key)
                    if(hd < self.threshold):
                        if hd < lowest_hamming_dist:
                            lowest_hamming_dist = hd
                            matched = True
                            matched_hash = key
                            object_counter = hamming_dict.get(key)
        if not matched:
            object_counter = len(hamming_dict)
        if matched_hash in hamming_dict:
            del hamming_dict[matched_hash]

        hamming_dict[dhash] = object_counter
        return  hamming_dict


    def drawTrackingPoints(self, image_np, centers, color=(0,0,255)):
        image_np = cv2.line(image_np, centers[0], centers[len(centers) - 1], color)
        return image_np


