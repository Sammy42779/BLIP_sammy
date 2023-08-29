import os
import json
import random

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import torch

from data.utils import pre_caption

class coco_karpathy_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
    
class coco_karpathy_caption_eval_adv(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30, prompt=''):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words    
        self.prompt = prompt  
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)    
        
        image_name = ann['image'].split('/')[-1][:-4]      
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        if type(ann['caption']) == list:
            # pre_caption(random.choice(ann['caption']), self.max_words)
            caption = self.prompt + pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            # pre_caption(ann['caption'], self.max_words)
            caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        
        # caption是从5个paired caption list里面随机选一个;对于SGA来说,应该是保留这5个caption
        
        return image, int(img_id), caption, image_name
    
    
class coco_karpathy_caption_eval_adv_sga(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30, prompt=''): 
        
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        self.ann = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt  

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            self.image.append(ann['image'])
            for j, caption in enumerate(ann['caption']):
                self.text.append(self.prompt + pre_caption(caption, self.max_words))  # 加上prompt
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, self.image[index])
        image_name = ann['image'].split('/')[-1][:-4]      
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text_ids =  self.img2txt[index]  # index=0; text_ids=[0, 1, 2, 3, 4]
        texts = [self.text[i] for i in self.img2txt[index]]  
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        return image, texts, int(img_id), text_ids, image_name

    def collate_fn(self, batch):  # 当我们使用DataLoader来批量获取数据时, collate_fn用于定义如何处理不同的数据样本以将其组合成一个批次
        imgs, txt_groups, img_ids, text_ids_groups, image_names = list(zip(*batch))        
        imgs = torch.stack(imgs, 0)
        return imgs, txt_groups, list(img_ids), text_ids_groups, image_names
    
class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)   
    
    
class coco_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index