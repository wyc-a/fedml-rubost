import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

def image_loader(images,transforms,mode):
    image=Image.open(images).convert(mode)
    image = transforms(image)

    return image

def read_images(image_path,transforms,mode='RGB'):
    images=[]
    num=0
    for filename in os.listdir(image_path):
        # print(filename)
        images.append(image_loader(image_path+'/'+filename,transforms,mode))
        num+=1

    images=torch.stack(images)  #图片组为4维张量 [N,C,H,W]
    #print(images.shape)
    return images

'''
images=read_images('C:\\Users\\WYC\\PycharmProjects\\data\\DRIVE\\training\\mask',1)
print(images.type)
print(images)
'''

class Resize:
    def __init__(self, shape):
        self.shape = [shape, shape] if isinstance(shape, int) else shape

    def __call__(self, img, mask):
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        return img[0], mask[0].byte()

class RandomResize:
    def __init__(self, w_rank,h_rank):
        self.w_rank = w_rank
        self.h_rank = h_rank

    def __call__(self, img, mask):
        random_w = random.randint(self.w_rank[0],self.w_rank[1])
        random_h = random.randint(self.h_rank[0],self.h_rank[1])
        self.shape = [random_w,random_h]
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        return img[0], mask[0].long()

class RandomCrop:
    def __init__(self, shape):
        self.shape = [shape, shape] if isinstance(shape, int) else shape
        self.fill = 0
        self.padding_mode = 'constant'

    def _get_range(self, shape, crop_shape):
        if shape == crop_shape:
            start = 0
        else:
            start = random.randint(0, shape - crop_shape)
        end = start + crop_shape
        return start, end

    def __call__(self, img, mask):
        _, h, w = img.shape
        sh, eh = self._get_range(h, self.shape[0])
        sw, ew = self._get_range(w, self.shape[1])
        return img[:, sh:eh, sw:ew], mask[:, sh:eh, sw:ew]

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(1)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        return normalize(img, self.mean, self.std, False), mask

class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img,cnt,[1,2])
        return img

    def __call__(self, img, mask):
        cnt = random.randint(0,self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask



class TrainDataset(Dataset):
    def __init__(self,image_path='/home/wangycheng/data/DRIVE/training/images'
                 ,mask_path='/home/wangycheng/data/DRIVE/training/mask'
                 ,lable_path="/home/wangycheng/data/DRIVE/training/1st_manual"):
        ''' self.transforms = Compose([
            # RandomResize([56,72],[56,72]),
            RandomCrop((48, 48)),
            RandomFlip_LR(prob=0.5),
            RandomFlip_UD(prob=0.5),
            RandomRotate()
        ])
     '''
        self.transforms=transforms.Compose([transforms.Resize((640,640)),
                                            transforms.ToTensor()])
        self.transforms_=transforms.Compose([transforms.Resize((640,640)),
                                             transforms.ToTensor()])
        self.images=read_images(image_path,self.transforms,)
        self.lables=read_images(lable_path,self.transforms_,'L')
        self.lables=self.lables.long()
        self.lables=torch.squeeze(self.lables)
        print(self.lables.shape)
        self.mask=read_images(mask_path,self.transforms,'L')



    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img=self.images[idx]
        lable=self.lables[idx]
        #mask=self.mask[idx]
        train_data={}
        train_data['x']=img
        train_data['y']=lable
        return train_data


class TestDataset(Dataset):
    def __init__(self,image_path='/home/wangycheng/data/DRIVE/test/images'
                 ,mask_path='/home/wangycheng/data/DRIVE/test/mask',
                 lable_path="/home/wangycheng/data/DRIVE/test/1st_manual"):
        self.transforms = transforms.Compose([transforms.Resize((640,640)),
                                              transforms.ToTensor()])
        self.transforms_ = transforms.Compose([transforms.Resize((640,640)),
                                               transforms.ToTensor()])
        self.images = read_images(image_path, self.transforms)
        self.lables = read_images(lable_path, self.transforms_,'L')
        self.lables = self.lables.long()
        self.lables = torch.squeeze(self.lables)
        self.mask = read_images(mask_path, self.transforms,'L')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        lable = self.lables[idx]
        #mask = self.mask[idx]
        train_data = {}
        train_data['x'] = img
        train_data['y'] = lable
        return train_data

'''
MyDataset=TrainDataset()
print(MyDataset.images[0:2])
'''

def batch_combine(batch_size,dataset,client_idx,batch_num):
    data=[]
    base=client_idx*batch_size*batch_num  #从何处开始取数据
    for i in range(0,batch_num):
        data_x =[]
        data_y =[]
        for j in range(base + i * batch_size, base + (i + 1) * batch_size):
            train_data=dataset.__getitem__(j)
            data_x.append(train_data['x'])
            data_y.append((train_data['y']))

        data_x=torch.stack(data_x)
        data_y=torch.stack(data_y)
        data.append((data_x,data_y))

    return data


def load_fed_data(client_num,train_dataset,test_dataset,batch_size):
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    class_num=[384,384]
    train_batch_num=train_dataset.__len__()//(client_num*batch_size)
    test_batch_num=test_dataset.__len__()//(client_num*batch_size)

    for client_idx in range(client_num):
        train_data=batch_combine(batch_size, train_dataset,client_idx,train_batch_num)
        test_data=batch_combine(batch_size,test_dataset,client_idx,test_batch_num)
        train_data_num+=len(train_data)
        test_data_num+=len(test_data)
        train_data_local_num_dict[client_idx]=len(train_data)

        train_data_local_dict[client_idx]=train_data
        test_data_local_dict[client_idx]=test_data
        train_data_global+=train_data
        test_data_global+=test_data

        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict,  test_data_local_dict, class_num]

    return  dataset

'''
trainset=TrainDataset(image_path='C:/Users/WYC/PycharmProjects/data/DRIVE/training/images',
                      lable_path='C:/Users/WYC/PycharmProjects/data/DRIVE/training/1st_manual',
                      mask_path='C:/Users/WYC/PycharmProjects/data/DRIVE/training/mask')

testset=TestDataset(image_path='C:/Users/WYC/PycharmProjects/data/DRIVE/test/images',
                      lable_path='C:/Users/WYC/PycharmProjects/data/DRIVE/test/1st_manual',
                      mask_path='C:/Users/WYC/PycharmProjects/data/DRIVE/test/mask')


result=load_fed_data(10,trainset,testset,2)
#print(result[5][0][0])

'''







