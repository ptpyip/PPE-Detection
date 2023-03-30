import os
import collections
from PIL import Image  
from typing import Any, Callable, Dict, List, Optional, Tuple  
from xml.etree.ElementTree import Element as ET_Element
from xml.etree.ElementTree import parse as ET_parse

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader 
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F

from detection import transforms as T
from detection import utils   

CLASSES = ['Background', 'person', 'hat']        # Background is default for Pytorch, custom class must start with 1

def getVOCDataLoader(args, getTransforms, dataset_transform=True):
    root, year = args.dataset_dir.split('/VOC')
    batch_size = args.batch_size
    num_workers  = args.num_workers
    
    # load datasets
    if dataset_transform:
        train_dataset = myVOCDetection(root, year, 
                                    image_set='train', classes=CLASSES,
                                    transforms=getTransforms(is_train=True))
        val_dataset = myVOCDetection(root, year, 
                                   image_set='val', classes=CLASSES,
                                   transforms=getTransforms())
    else: 
        train_dataset = myVOCDetection(root, year, 
                                   image_set='train', classes=CLASSES,)
        val_dataset = myVOCDetection(root, year,
                                   image_set='val', classes=CLASSES,)
    
    # setup data loader
    train_data_loader = DataLoader(train_dataset, 
                                batch_size, shuffle=True, 
                                num_workers=num_workers,
                                collate_fn=utils.collate_fn)
    val_data_loader = DataLoader(val_dataset,
                                batch_size, shuffle=True, 
                                num_workers=num_workers,
                                collate_fn=utils.collate_fn)
    
    return train_data_loader, val_data_loader

def cleanDataSet(label_path, img_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()       
    
    name_list = []
    for line in lines:
        name = str(line).strip('\n')
        name_list.append(f"{name}.jpg")
        
    file_names = [file_name for file_name in os.listdir(img_path)]
    delete_names = []
    for name in name_list:
        if(not name in file_names):
            print(name)
            delete_names.append(name)
                
    # delete matching content
    with open(label_path, 'w') as f:
        for line in lines:
            name = line.strip('\n')
            name = f"{name}.jpg"
            if name not in delete_names:
                f.write(line)
            else:
                print(f"Deletting {name}")
                
    return 

class myVOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"
    
    def __init__(
        self,
        root: str,
        year: str,
        image_set: str = "train",
        classes: List = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        
        super().__init__(root, transforms, transform, target_transform)
        self.classes = classes
        
        self.filename = f"VOC{year}"
        voc_root = os.path.join(self.root, self.filename)

        if not os.path.isdir(voc_root):
            raise RuntimeError(voc_root)

        self.__getVOCData(voc_root, image_set)
        
        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)
       
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")     # - image: a PIL Image (H, W)
        target = self.__get_targets(index)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __getVOCData(self, voc_root, image_set):
        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]
        
        return

    def __parse_xml(self, node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.__parse_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    @property
    def annotations(self) -> List[str]:
        return self.targets
        
    def __get_targets(self, index):
        ''' 
            - target: a dict containing the following fields
                - boxes (FloatTensor[N, 4]): 
                    the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, 
                    ranging from 0 to W and 0 to H

                - labels (Int64Tensor[N]): the label for each bounding box. 
                    0 represents always the background class.

                - image_id (Int64Tensor[1]): an image identifier. 
                    It should be unique between all the images in the dataset, 
                    and is used during evaluation

                - area (Tensor[N]): The area of the bounding box. 
                    This is used during evaluation with the COCO metric, 
                    to separate the metric scores between small, medium and large boxes.

                - iscrowd (UInt8Tensor[N]): 
                    instances with iscrowd=True will be ignored during evaluation.
        '''


        target = self.__parse_xml(
            ET_parse(self.annotations[index]).getroot()
        )
        
        labels, boxes, areas= [], [], []
        for object in target['annotation']['object']:
            name = object['name']
            if name not in self.classes: continue
            
            xmax = int(object['bndbox']['xmax'])
            xmin = int(object['bndbox']['xmin'])
            ymax = int(object['bndbox']['ymax'])
            ymin = int(object['bndbox']['ymin'])
            
            labels.append(self.classes.index(name))
            boxes.append([xmin, ymin, xmax, ymax])
            areas.append((xmax - xmin) * (ymax - ymin))
            
        target = {
            'image_id'  : torch.tensor([index]),
            'labels'    : torch.tensor(labels, dtype=torch.int64),
            'boxes'     : torch.tensor(boxes, dtype=torch.float32) ,
            'area'      : torch.tensor(areas, dtype=torch.float32),
            'iscrowd'   : torch.zeros((len(labels),), dtype=torch.uint8)    # suppose all instances are not crowd
        }
        
        return target

    

