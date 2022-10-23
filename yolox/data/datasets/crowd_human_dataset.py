import os
from loguru import logger

import cv2
import numpy as np

from .datasets_wrapper import Dataset

class CrowdHumanDataset(Dataset):
    """
    Crowd Human dataset class.
    Link: http://www.crowdhuman.org/
    
    hbox: a head bounding-box
    vbox: human visible-region bounding-box
    fbox: human full-body bounding-box
    """

    def __init__(
        self,
        name,
        mode,
        data_dir,
        class_list,
        train_label_file,
        valid_label_file,
        img_size=(416, 416),
        preproc=None
    ):
    
        super().__init__(img_size)
        print(' ---', mode, 'CrowdHumanDataset ---')
        
        # Images
        self.data_path = []
        if mode == 'train':
            data_dir_train1 = os.path.join(data_dir, 'CrowdHuman_train01', 'Images')
            for fname in os.listdir(data_dir_train1):
            	self.data_path.append(os.path.join(data_dir, 'CrowdHuman_train01', 'Images', fname))
            data_dir_train2 = os.path.join(data_dir, 'CrowdHuman_train02', 'Images')
            for fname in os.listdir(data_dir_train2):
            	self.data_path.append(os.path.join(data_dir, 'CrowdHuman_train02', 'Images', fname))
            data_dir_train3 = os.path.join(data_dir, 'CrowdHuman_train03', 'Images')
            for fname in os.listdir(data_dir_train3):
            	self.data_path.append(os.path.join(data_dir, 'CrowdHuman_train03', 'Images', fname))
        elif mode == 'valid':
            data_dir_valid = os.path.join(data_dir, 'CrowdHuman_val', 'Images')
            for fname in os.listdir(data_dir_valid):
            	self.data_path.append(os.path.join(data_dir, 'CrowdHuman_val', 'Images', fname))
        print('Length of images:', len(self.data_path))
        self.data_path = self.data_path
        
        train_label_path = os.path.join(data_dir, train_label_file)
        valid_label_path = os.path.join(data_dir, valid_label_file)

        # Labels        
        if mode == 'train':
            print('Train label path:', train_label_path)
            with open(train_label_path, 'r+') as f:
               datalist = f.readlines()    

        if mode == 'valid':            
            print('Valid label path:', valid_label_path)
            with open(valid_label_path, 'r+') as f:
                datalist = f.readlines()        
        
        self.inputfile = self.__parsing_label(datalist)
        print('Length of labels:', len(self.inputfile.keys()))
       
        self.name = name
        self.class_ids = class_list 
        self.img_size = img_size
        self.preproc = preproc
        print('Image size:', self.img_size)

    def __len__(self):
        return len(self.data_path)
 
    def __parsing_label(self, datalist):
        
        inputfile = {}   
        for i in np.arange(len(datalist)):
            adata = dict(eval(datalist[i].strip()))
            file_name = adata['ID']
            inputfile[file_name] = []
            gtboxes = adata['gtboxes']
            for gtbox in gtboxes:
                if gtbox['tag']=='person':
                    data = {
                    'name': 'person',
                    'a head': gtbox['hbox'],
                    'human visible-region': gtbox['vbox']
                    }
                    inputfile[file_name].append(data)
		    
        return inputfile

    def load_res(self, index, img_info, r):
        
        img_path = self.data_path[index]
        img_name = img_path.split('/')[-1][:-4]
        annos = self.inputfile[img_name]
               
        width = img_info[1]
        height = img_info[0]
        
        objs = []
        for anno in annos:
            head_box = anno['a head']
            person_box = anno['human visible-region']
                
            # person    
            x1 = np.max((0, person_box[0]))
            y1 = np.max((0, person_box[1]))
            x2 = np.min((width, x1 + np.max((0, person_box[2]))))
            y2 = np.min((height, y1 + np.max((0, person_box[3]))))
            if x2 >= x1 and y2 >= y1:
                objs.append([x1, y1, x2, y2, 'person'])
            
            # head 
            x1 = np.max((0, head_box[0]))
            y1 = np.max((0, head_box[1]))
            x2 = np.min((width, x1 + np.max((0, head_box[2]))))
            y2 = np.min((height, y1 + np.max((0, head_box[3]))))
            if x2 >= x1 and y2 >= y1:
                objs.append([x1, y1, x2, y2, 'head'])

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj[4])
            res[ix, 0:4] = obj[0:4]
            res[ix, 4] = cls
            
        res[:, :4] *= r
        
        # x1, y1, x2, y2, class        
        return res

    def load_resized_img(self, index):
        img, img_info = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        return resized_img, img_info, r

    def load_image(self, index):
        img_path = self.data_path[index]

        img = cv2.imread(img_path)
        assert img is not None, f"file named {img} not found" 
        
        img_info = (img.shape[0], img.shape[1]) # (h,w)

        return img, img_info

    def pull_item(self, index):
        
        if str(type(index)) == "<class 'torch.Tensor'>":
        	index = index.cpu().numpy()
        img, img_info, r = self.load_resized_img(index)
        res = self.load_res(index, img_info, r)
        
        return img, res.copy(), img_info, np.array([index])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
    
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        
        return img, target, img_info, img_id
        
    def collate_fn(self, batch):
        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs, labels, img_info, img_id = list(zip(*batch))
        
        # Images
        imgs = np.stack(([img for img in imgs]), axis=0) # e.g. (4, 3, 640, 64)
        
        # Labels
        labels = np.stack(([list(img_id[i]) + list(l) for i, label in enumerate(labels) for a, l in enumerate(list(label))]), axis=0) # e.g. (81, 6)
        
        return imgs, labels, img_info
