import os
import shutil
import cv2


class FileManager:
    def __init__(self):
        pass

    # Read each line from the label_path and move images from image_path to dest_path
    def img_to_class_dir(self, img_path=None, label_path=None, dest_path=None):
        labels = open(label_path, "r")
        for line in labels:
            fields = line.split("\t")
            print("imageid:",fields[0])
            print("disease code: " + fields[1].split(" ")[0])
            image_filename = img_path+""+fields[0]+".ppm"
            dest_filename = dest_path+"/"+fields[1].split(" ")[0]+"/"+fields[0]+".jpg"
            if not os.path.exists(dest_path+"/"+fields[1].split(" ")[0]):
                os.makedirs(dest_path+"/"+fields[1].split(" ")[0])
            if os.path.exists(image_filename):
                shutil.move(image_filename, dest_filename)
                print("Moved file")
    def move_file(filepath=None, dest_path=None):
        print("file move")



#FileManager.img_to_class_dir(img_path="resources/all-images/", label_path="resources/retina_labels.txt", dest_path="images/retina")
for path, subdirs, files in os.walk("images/retina/"):
    for name in files:
        print(os.path.join(path, name))
        images = cv2.imread(os.path.join(path, name))
        cv2.imwrite("images/retina/pbgs/"+name+".png", images)
        cv2.imshow("im", images)
        #cv2.waitKey(0)