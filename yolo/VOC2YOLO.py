'''
    * This file is mdde to convert VOC type dataset into YOLO format.
    * Usage:
        - Modify main() function / import to a notebook and cal it
        - Output:
            <CURRENT_DIR>/
            |-- VOC<year>/
            |   |-- Annotation          # .xml annotations for all images
            |   |-- ImageSets           # .txt files specify which set each image belongs
            |   |   |-- Main
            |   |   |   |-- test.txt
            |   |   |   |-- train.txt
            |   |   |   |-- val.txt
            |   |-- JPEGImages          # all the jpg files
        - Output:
            <CURRENT_DIR>/
            |-- <tgt_folder_name>/
            |   |-- images              # all the jpg files
            |   |   |-- test
            |   |   |-- train
            |   |   |-- val
            |   |-- labels              # .txt labels files for each image
            |   |   |-- test
            |   |   |-- train
            |   |   |-- val
        - Reminder: It *MOVES* not copy the images, i.e. the original VOC folder will become empty afterwords.
'''

import os
import shutil
from xml.etree import ElementTree
# from os import listdir, getcwd

   
def main():
    converter = VOC2YOLO(
        year=2028, 
        sets=['val', 'test'],
        classes=['no_helmet', 'helmet'],
        tgt_folder_name="YOLO"              # your desire folder name for all outputs
    )
    converter.converts()

def convertBBox(size, xmlbox):
    
    xmin = float(xmlbox.find('xmin').text)
    xmax = float(xmlbox.find('xmax').text)
    ymin = float(xmlbox.find('ymin').text)
    ymax = float(xmlbox.find('ymax').text)
    
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (xmin + xmax)/2.0 - 1
    y = (ymin + ymax)/2.0 - 1
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convertToLabel(cls_id, bb):
    return str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'

class VOC2YOLO():
    # SETS = ['val', 'test']
    # CLASSES = ['no_helmet', 'helmet']
    
    def __init__(self, year: int, sets: list, classes: list, tgt_folder_name='YOLO'):
        self.year = year
        self.img_sets = sets
        self.classes = classes
        self.tgt_folder_name = tgt_folder_name
        
    def updatePaths(self, image_set):
        lable_path = f'./{self.tgt_folder_name}/labels/{image_set}'
        img_path = f'./{self.tgt_folder_name}/images/{image_set}'
        
        if not os.path.exists(lable_path):
            os.makedirs(lable_path)
            
        if not os.path.exists(img_path):
            os.makedirs(img_path)
            
        self.lable_path = lable_path
        self.img_path = img_path
        
    def getImageIDs(self, image_set):
        with open(f'./VOC{self.year}/ImageSets/Main/{image_set}.txt') as txt_file:
            
            return txt_file.read().strip().split()

    
    def getAnnotations(self, image_id):
        with open(f'./VOC{self.year}/Annotations/{image_id}.xml') as in_file :
            tree = ElementTree.parse(in_file)
        return tree
    
    def convertAnnotations(self, image_id):
        tree = self.getAnnotations(image_id)
        
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        out_annotations = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult)==1:
                continue
            
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            bb = convertBBox((w,h), xmlbox)
            out_annotations.append(convertToLabel(cls_id, bb))
    
        return out_annotations
        ...

    def generateLables(self, image_id):
        out_annotations = self.convertAnnotations(image_id)

        with open(f'{self.lable_path}/{image_id}.txt', 'w') as out_file:
            for content in out_annotations:
                out_file.write(content)
        
        return
    
    def partitionImg(self, img_id):
        og_file = f'./VOC{self.year}/JPEGImages/{img_id}.jpg'
        
        shutil.move(og_file, self.img_path)
    
    def converts(self):
        for img_set in self.img_sets:
            self.updatePaths(img_set)
            for img_id in self.getImageIDs(img_set):
                print(f'./VOC{self.year}/JPEGImages/{img_id}.jpg')
                self.partitionImg(img_id)
                self.generateLables(img_id)
        print("Finished processing")
    
if __name__ == '__main__':
    main()