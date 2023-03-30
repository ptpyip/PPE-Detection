'''
    Follow the torchvision object detection finetuning tutorial
    (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
'''
import torch
from torch.utils.data import DataLoader 
from torch.optim import lr_scheduler 

import torchvision as tv
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2 as faster_rcnn
from torchvision.models.detection.ssd import ssd300_vgg16 as ssd

# detection module from Torchversion/references/detection (download required)
from detection import transforms as T
from detection import utils           
from detection.engine import train_one_epoch, evaluate

# self-developed module
from dataloader import myVOCDetection, getVOCDataLoader
from utils import parse_args

### 0. Preperation 
''''
Download TorchVision repo to use some files from references/detection
!git clone https://github.com/pytorch/vision.git
or 
upload detection
'''

CLASSES = ['Background', 'person', 'hat']        # Background is default for Pytorch, custom class must start with 1

# handle user input 
# CLI setup based on GitHub repo
# (https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/blob/master/train_yolo.py)

# data augmentation
def getTransforms(is_train: bool = False):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    
    # Data augmentation 
    if is_train:
        #Nomalization is done implicitly 
        transforms.append(T.RandomAffine( 
            degrees = 10, translate = (0.1, 0.1), 
            scale = (0.9, 1.9) ,shear = 10
        ))
        transforms.append(T.RandomHorizontalFlip(p=0.5))               # applied to 
    return T.Compose(transforms)

### 2. Finetuning from a pretrained model
def getModel(args, num_classes):
    model = None
    if args.model == 'ssd':
        model = ssd(weights="DEFAULT")
    if args.model == 'faste_rcnn':
        model = faster_rcnn(weights="DEFAULT")
    return model

def train(model, data_loaders, args):
    train_data_loader, val_data_loader = data_loaders
    
    device = torch.device('cpu')
    if torch.cuda.is_available(): 
        device = torch.device('cuda') 
        
    model.to(device) 
    optimizer = getOptimizer(model, args)
    scheduler = getScheduler(optimizer, args)
    
    start_epoch = 0
    if len(args.resume) != 0:                   # add resume functionality
        args.save_path = f"{args.resume.rsplit('_', maxsplit=1)[0]}.pt"
        checkpoint = torch.load(args.resume)        # force load back to gpu
        
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
     
    for epoch in range(start_epoch, args.epochs):    
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_data_loader, 
                        device, epoch, print_freq=args.log_interval)
        # print("done 1 ep")
        # break
        scheduler.step()
        evaluate(model, val_data_loader, device=device)
        
        if epoch%args.save_interval == 0:
            save_path = f"{args.save_path.rstrip('.pt')}_{epoch+1}.pt"
            checkpoint = {
                'epoch'             : epoch,
                'model_state_dict'  : model.state_dict(),
                'optimizer_dict'    : optimizer.state_dict(),
                'scheduler_dict'    : scheduler.state_dict(),
            }
            torch.save(checkpoint, save_path)
            
    torch.save(model.state_dict(), args.save_path)
    return
    
def testTrain():
    from torchvision import transforms

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    train_dataset = myVOCDetection('/content/', '2028', 
                                   image_set='train', classes=CLASSES,
                                   transforms=getTransforms(is_train=True))
        
    data_loader = DataLoader(train_dataset, 
                                batch_size=2, shuffle=True, 
                                num_workers=0)
    model = getModel(num_classes=2)
    
    images, targets = next(iter(data_loader))
    images = [image for image in images]
    targets = [{k: v for k, v in t.items()} for t in targets]
    print(targets[0].keys())
    output = model(images, targets) 
    model.eval()    
    
    return

def getOptimizer(model, args):
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.wd
    
    # for possible fintuning optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, learning_rate, momentum, 
                           weight_decay=weight_decay)
    
def getScheduler(optimizer, args):
    if args.lr_mode == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=args.lr_decay)
    return scheduler

def main():
    args = parse_args()
    if args.dataset != 'voc': 
        print("Only support VOC dataset!")
        
    torch.manual_seed(args.seed)
    data_loaders = getVOCDataLoader(args)
    model = getModel(args, num_classes=2)     # exclude background
    
    train(model, data_loaders, args)
    
    return

if __name__ == "__main__":
    main() 