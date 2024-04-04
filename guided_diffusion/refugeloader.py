import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class REFUGEDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):


        # df = pd.read_csv(os.path.join(data_path, "ISBI2016_ISIC_Part3B_" + mode + "_GroundTruth.csv"), encoding="gbk")
        # self.name_list = df.iloc[:,0].tolist()
        # self.name_list = [os.path.join("ISBI2016_ISIC_Part3B_" +mode+"_Data",file_name+".jpg") for file_name in self.name_list]
        # self.label_list = df.iloc[:,0].tolist()
        # self.label_list = [os.path.join("ISBI2016_ISIC_Part3B_" +mode+"_Data",file_name+"_Segmentation.png") for file_name in self.label_list]
        self.name_list = os.listdir(os.path.join(data_path, mode, "images"))
        # self.label_list = os.listdir(os.path.join(data_path, mode, "mask"))

        self.data_path = data_path
        # print(self.label_list)
        print(self.name_list)
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""

        # print("hi idx: ", index)
        # print(self.name_list)
        # print(self.label_list)

        name = self.name_list[index]
        # print("hi name: ", name)
        img_path = os.path.join(os.path.join(self.data_path, self.mode, "images"), name)

        mask_name = name.split(".")[0]+".bmp"
        msk_path = os.path.join(os.path.join(self.data_path, self.mode, "mask"), mask_name)
        print("names: ", name, img_path, mask_name, msk_path)


        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        print(img.size, mask.size)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)


        return (img, mask, name)

# import os
# import sys
# import pickle
# import cv2
# from skimage import io
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms.functional as F
# import torchvision.transforms as transforms
# import pandas as pd
# from skimage.transform import rotate

# class ISICDataset(Dataset):
#     def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):


#         df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
#         self.name_list = df.iloc[:,1].tolist()
#         self.label_list = df.iloc[:,2].tolist()
#         self.data_path = data_path
#         self.mode = mode

#         self.transform = transform

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, index):
#         """Get the images"""
#         name = self.name_list[index]
#         img_path = os.path.join(self.data_path, name)
        
#         mask_name = self.label_list[index]
#         msk_path = os.path.join(self.data_path, mask_name)

#         img = Image.open(img_path).convert('RGB')
#         mask = Image.open(msk_path).convert('L')

#         # if self.mode == 'Training':
#         #     label = 0 if self.label_list[index] == 'benign' else 1
#         # else:
#         #     label = int(self.label_list[index])

#         if self.transform:
#             state = torch.get_rng_state()
#             img = self.transform(img)
#             torch.set_rng_state(state)
#             mask = self.transform(mask)


#         return (img, mask, name)


# import os
# import sys
# import pickle
# import cv2
# from skimage import io
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms.functional as F
# import torchvision.transforms as transforms
# import pandas as pd
# from skimage.transform import rotate

# class ISICDataset(Dataset):
#     def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):


#         df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
#         self.name_list = df.iloc[:,1].tolist()
#         self.label_list = df.iloc[:,2].tolist()
#         self.data_path = data_path
#         self.mode = mode

#         self.transform = transform

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, index):
#         """Get the images"""
#         name = self.name_list[index]
#         img_path = os.path.join(self.data_path, name)
        
#         mask_name = self.label_list[index]
#         msk_path = os.path.join(self.data_path, mask_name)

#         img = Image.open(img_path).convert('RGB')
#         mask = Image.open(msk_path).convert('L')

#         # if self.mode == 'Training':
#         #     label = 0 if self.label_list[index] == 'benign' else 1
#         # else:
#         #     label = int(self.label_list[index])

#         if self.transform:
#             state = torch.get_rng_state()
#             img = self.transform(img)
#             torch.set_rng_state(state)
#             mask = self.transform(mask)


#         return (img, mask, name)