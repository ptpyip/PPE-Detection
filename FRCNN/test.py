import json
import argparse
import torch
from torch.utils.data import DataLoader 
from torch.optim import lr_scheduler 

import torchvision as tv
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn as faster_rcnn
from torchvision.models.detection.ssd import ssd300_vgg16 as ssd

from detection import transforms as T
from detection import utils        # from references/detection, required download
from detection.engine import train_one_epoch, evaluate
from dataloader import myVOCDetection

CLASSES = ['Background', 'person', 'hat']        # Background is default for Pytorch, custom class must start with 1

def detect(model, test_dataloader, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    device = torch.device('cpu')
    if torch.cuda.is_available(): 
        device = torch.device('cuda') 
        
    ouput_list = []
    for images, targets in metric_logger.log_every(test_dataloader, 100, "Test:"):
        images = list(img.to(device) for img in images)
        outputs = model(images)
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
        store2List(res, ouput_list)
    stroe2JSON(ouput_list, args)
    
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
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    args = parser.parse_args()
    return args

# data augmentation
def getTransforms():
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    
    return T.Compose(transforms)

def getVOCDataLoader(args, dataset_transform=True):
    root, year = args.dataset_dir.split('/VOC')
    batch_size = args.batch_size
    num_workers  = args.num_workers
    
    # load datasets
    if dataset_transform:
        test_dataset = myVOCDetection(root, year, 
                                   image_set='test', classes=CLASSES,
                                   transforms=getTransforms())
    else:
        test_dataset = myVOCDetection(root, year,
                                   image_set='test', classes=CLASSES,)
    
    # setup data loader
    test_data_loader = DataLoader(test_dataset, 
                                batch_size, shuffle=True, 
                                num_workers=num_workers,
                                collate_fn=utils.collate_fn)
    
    return test_data_loader


### 2. Finetuning from a pretrained model
def loadModel(args, num_classes):
    model = None
    if args.model_path == '': return
    
    if args.model == 'ssd':
        model = ssd(weights="DEFAULT")
    if args.model == 'faste_rcnn':
        model = faster_rcnn(weights="DEFAULT")
    
    if args.model_path == '':
        checkpoint = torch.load(args.checkpt_path)        # force load back to gpu
        
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = torch.load(args.model_path)
        
    model.load_state_dict(state_dict)
        
    return model

def store2List(res, ouput_list):
    '''
        COCO Result Format
        [{
            "image_id": int, 
            "category_id": int, 
            "bbox": [x,y,width,height], 
            "score": float,
        }]
    '''
    # ouput_list = []
    for image_id, output in res.items():
        boxes, labels, scores = output['boxes'], output['labels'], output['scores'], 
        for box, label, score in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = box.tolist()
            ouput_list.append({
                "image_id": image_id, 
                "category_id": int(label)-1, 
                "bbox": [xmin, ymin, xmax-xmin, ymax-ymin], 
                "score": float(score),
            })
    return 

def stroe2JSON(ouput_list, args=None):
    if args == None: save_path = './results.json'
    else: save_path = args.save_path
    
    with open(save_path, 'w') as f:
        json.dump(ouput_list, f)

def main():
    args = parse_args()
    if args.dataset != 'voc': 
        print("Only support VOC dataset!")
        
    torch.manual_seed(args.seed)
    test_dataloader = getVOCDataLoader(args)
    model = loadModel(args, num_classes=2)     # exclude background
    
    model.eval()
    detect(model, test_dataloader, args)
    return

def test_work():
    
    res = {1069: {'boxes': torch.tensor(
                [[534.8536, 214.1557, 583.4321, 268.3827],
                [ 97.9795, 219.4712, 142.1924, 271.7810],
                [ 34.8681, 204.6419,  74.3532, 246.3727],
                [ 37.3356, 209.6120,  88.2365, 261.1911],
                [166.4344, 197.9505, 192.8831, 229.7748],
                [388.4300, 202.4401, 426.0899, 244.9302],
                [109.3546, 229.4491, 151.8606, 280.4128],
                [105.9546, 208.5557, 141.2658, 247.8899],
                [165.7556, 193.8674, 189.2045, 218.6615],
                [469.8923, 205.8327, 504.2503, 241.1631],
                [252.9027, 205.7362, 284.9241, 242.4979],
                [304.9023, 200.1141, 326.9103, 224.6072],
                [330.9700, 203.4847, 355.6691, 229.4630],
                [336.3507, 197.7083, 359.4549, 222.4954],
                [246.4833, 213.4467, 288.6286, 258.9347],
                [193.1730, 234.0852, 242.0958, 286.1906],
                [452.2488, 192.8806, 472.1194, 214.4563],
                [446.4348, 202.5840, 476.2869, 234.1583],
                [440.4492, 198.5108, 465.9226, 227.3947],
                [218.8901, 200.6793, 248.0048, 233.3392],
                [303.6786, 205.6786, 329.6560, 233.2693],
                [383.7263, 190.0203, 402.7723, 210.6623],
                [257.9940, 222.5753, 293.6563, 266.3939],
                [388.7335, 191.4140, 409.3017, 214.8000],
                [ 55.9736, 207.2611,  91.7826, 248.7385],
                [551.4503, 211.4175, 588.5673, 253.6874],
                [346.4560, 198.8718, 366.8894, 219.6568],
                [378.1184, 184.8212, 394.4105, 202.6353],
                [307.1334, 186.9592, 325.5958, 210.0361],
                [347.3299, 191.8309, 366.0440, 213.3251]]), 
    'labels': torch.tensor(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1]), 
    'scores': torch.tensor(
                [0.8990, 0.7910, 0.7215, 0.6885, 0.5548, 0.5032, 0.4796, 0.4677, 0.3956,
                0.3910, 0.3768, 0.3408, 0.3066, 0.3060, 0.3012, 0.2971, 0.2820, 0.2693,
                0.2641, 0.2102, 0.2049, 0.1969, 0.1796, 0.1693, 0.1028, 0.0984, 0.0769,
                0.0703, 0.0660, 0.0614])
    }}
    ouput_list = []
    store2List(res, ouput_list)
    stroe2JSON(ouput_list)
    
if __name__ == "__main__":
    main()
    