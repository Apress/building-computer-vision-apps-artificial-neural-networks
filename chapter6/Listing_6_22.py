import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path, img_path, label_path):
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    class_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        xml_list = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            imagename = str(root.find('filename').text)
            print("image", imagename)
            index = int(imagename.rfind("_"))
            print("index: ", index)
            classname = imagename[0:index]

            class_index = 0
            if (class_list.count(classname) > 0):
                class_index = class_list.index(classname)

            else:
                class_list.append(classname)
                class_index = class_list.index(classname)

            print("width: ", root.find("size").find("width").text)
            print("height: ", root.find("size").find("height").text)
            print("minx: ", member[4][0].text)
            print("ymin:", member[4][1].text)
            print("maxx: ", member[4][2].text)
            print("maxy: ", member[4][3].text)
            w = float(root.find("size").find("width").text)
            h = float(root.find("size").find("height").text)
            dw = 1.0 / w
            dh = 1.0 / h
            x = (float(member[4][0].text) + float(member[4][2].text)) / 2.0 - 1
            y = (float(member[4][1].text) + float(member[4][3].text)) / 2.0 - 1
            w = float(member[4][2].text) - float(member[4][0].text)
            h = float(member[4][3].text) - float(member[4][1].text)
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh

            value = (class_index,
                     x,
                     y,
                     y,
                     h
                     )
            print("The line value is: ", value)
            print("csv file name: ", os.path.join(label_path, imagename.rsplit('.', 1)[0] + '.txt'))
            xml_list.append(value)
            df = pd.DataFrame(xml_list)
            df.to_csv(os.path.join(label_path, imagename.rsplit('.', 1)[0] + '.txt'), index=None, header=False, sep=' ')

    class_df = pd.DataFrame(class_list)
    return class_df


def create_training_and_test(image_dir, label_dir):
    file_list = []
    for img in glob.glob(image_dir + "/*"):
        print(os.path.abspath(img))

        imagefile = os.path.basename(img)

        textfile = imagefile.rsplit('.', 1)[0] + '.txt'

        if not os.path.isfile(label_dir + "/" + textfile):
            print("delete image file ", img)
            os.remove(img)
            continue
        file_list.append(os.path.abspath(img))

    file_df = pd.DataFrame(file_list)
    train = file_df.sample(frac=0.7, random_state=10)
    test = file_df.drop(train.index)
    train.to_csv("petdata/train.txt", index=None, header=False)
    test.to_csv("petdata/test.txt", index=None, header=False)


def main():
    img_dir = "petdata/images"
    label_dir = "petdata/labels"

    xml_path = os.path.join(os.getcwd(), 'petdata/annotations/xmls')
    img_path = os.path.join(os.getcwd(), img_dir)
    label_path = os.path.join(os.getcwd(), label_dir)

    class_df = xml_to_csv(xml_path, img_path, label_path)
    class_df.to_csv('petdata/class.data', index=None, header=False)
    create_training_and_test(img_dir, label_path)
    print('Successfully converted xml to csv.')


main()