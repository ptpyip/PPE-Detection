"""This file perform inference on img / video to evaluate our models"""

import cv2
import json
import argparse
from PIL import Image  


import torch
from torch.utils.data import DataLoader 
from torch.optim import lr_scheduler 

import torchvision as tv
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2 as faster_rcnn
from torchvision.models.detection.ssd import ssd300_vgg16 as ssd
from torchvision.transforms import transforms as T

from detection import utils        # from references/detection, required download
from detection.engine import train_one_epoch, evaluate

from dataloader import myVOCDetection, getTransforms
from test import getTransforms

CLASSES = ['Background', 'no_helmet', 'helmet']       
 
def detectImg(img_path, model):
    model.eval()
    
    transforms = getTransforms()
    
    img = cv2.imread(img_path)

    img_in = Image.fromarray(img).convert("RGB") 
    img_tensor = transforms(img_in).unsqueeze(0).to(torch.device('cpu'))
    out = model(img_tensor) 
    drawBBoxs(img, out[0])
                          
def detectVideo(cap, model, writer, args):
    model.eval()
    
    transforms = getTransforms()
    
    device = torch.device('cpu')
    if torch.cuda.is_available(): 
        device = torch.device('cuda') 
    model.to(device)
    
    ret = True
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if not ret: break
        
        img_in = Image.fromarray(frame).convert("RGB") 
        img_tesnsor = transforms(img_in).unsqueeze(0).to(device)
        out = model(img_tesnsor) 
        drawBBoxs(frame, out[0], args)  
        writer.write(frame)
        print("next frame")
        
def drawBBox(img, box):
    tv.utils.draw_bounding_boxes(img, box)
    return img
    
def drawBBoxs(img, out, args):
    boxes = out['boxes'].to(torch.device('cpu'))
    labels = out['labels'].to(torch.device('cpu'))
    scores =  out['scores'].to(torch.device('cpu'))

    for box, label, score in zip(boxes, labels, scores):        # iterate through detected obj
        if float(score) < args.conf_thres: 
            # only draw when confidence level high enough
            continue        
        
        xmin, ymin, xmax, ymax = box
        print(f"{CLASSES[label]} found!")
        cv2.rectangle(img, 
                      (int(xmin), int(ymin)), 
                      (int(xmax), int(ymax)), 
                      (0, 255, 0), 2
        )
        label = int(label)
        cv2.putText(img, f"{CLASSES[label]}_{score}", 
                    (int(xmin), int(ymax)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255*(label-1)), 1)
        
    return 
       
# handle user input 
def parse_args():
    parser = argparse.ArgumentParser(description='Train the networks with random input shape.')
    parser.add_argument('--model', type=str, default='faste_rcnn',
                        #ssd  mobilenet1.0 mobilenet0.25
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--dataset-dir', type=str, default='../content/VOC2028',
                        help='Dirctary of training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='1',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--model-path', type=str, default='',
                        help='Saving parameter path')
    parser.add_argument('--checkpt-path', type=str, default='',
                        help='Saving parameter path')
    parser.add_argument('--save-path', type=str, default='/content/results.json',
                        help='Saving parameter path')
    parser.add_argument('--source-path', type=str, default='',
                        help='Saving parameter path')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    args = parser.parse_args()
    return args


### 2. Finetuning from a pretrained model
def loadModel(model_name, model_path):
    model = None
    if model_path == '': return
    
    if model_name == 'ssd':
        model = ssd(weights="DEFAULT")
    if model_name == 'faste_rcnn':
        model = faster_rcnn(weights="DEFAULT")
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
      
    model.load_state_dict(state_dict)

    for param in model.parameters():
          param.requires_grad = False
          
    return model
             
    
def main():
    args = parse_args()
    if args.dataset != 'voc': 
        print("Only support VOC dataset!")
    
    torch.manual_seed(args.seed)
    model = loadModel(args, num_classes=2)     # exclude background
    
    model.eval()
    
    cap = cv2.VideoCapture(args.source_path)    
    img = cv2.imread()
    w = cv2.CAP_PROP_FRAME_WIDTH
    h = cv2.CAP_PROP_FRAME_HEIGHT
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))
    
    detectVideo(cap, model, writer, args)
    
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    return
    
    
if __name__ == "__main__":
    main()
    