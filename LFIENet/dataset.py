import os.path
import h5py
import math
import os
import torch
import torch.utils.data as data
import cv2
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]



def get_imlistj(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.JPG')]



def Im2Patch4(data, size, stride):
    endc = data.shape[2]
    enda1 = data.shape[0]
    enda2 = data.shape[1]
    endw = data.shape[3]
    endh = data.shape[4]
    num = math.ceil((endw - size)/stride + 1)
    Y = torch.zeros([num*num, enda1, enda2,endc,size, size])
    for ix in range(num):
        for iy in range(num):
            ind = iy + (num * ix)
            iby = min(endw-size, iy * stride)# 128-16
            ibx = min(endh-size, ix * stride)
            Y[ind, :, :, : ,:,:]=data[:,:,:,ibx:ibx+size,iby:iby+size]
    return Y



def get_h5py_lf55(path, save_path, size, stride, img_size):
    file_h5f = h5py.File(save_path, 'w')
    list_i = os.listdir(path)
    list_i.sort()
    #print(list_i)
    idx = 0
    for file_num in range(len(list_i)):
        img_list = get_imlist(os.path.join(path, list_i[file_num]))
        img_list.sort()
        LF = torch.zeros(1, 5, 5, img_size, img_size)
        for i in range(5):
            for j in range(5):
                img = ToTensor()(Image.open(img_list[i*5+j]))
                LF[:, i, j, :, :] = img
        data = LF.float()
        patches = Im2Patch4(data, size, stride)
        patches = patches.numpy()
        for patch_num in range(len(patches)):
            patch = file_h5f.create_dataset(str(idx), data=patches[patch_num, :, :, :, :, :])
            idx += 1
    return file_h5f

def prepare_data(data_path, size, stride, img_size):
    print('process training data')
    lf_path = os.path.join(data_path, 'lf')
    dslr_path = os.path.join(data_path, 'dslr')
    lfh_path = os.path.join(data_path, 'he_lf')
    dslrh_path = os.path.join(data_path, 'he_dslr')

    save_lf_path = os.path.join(data_path, 'lf.h5')
    save_dslr_path = os.path.join(data_path, 'dslr.h5')
    save_lfh_path = os.path.join(data_path, 'lfh.h5')
    save_dslrh_path = os.path.join(data_path, 'dslrh.h5')

    lf_h5f = get_h5py_lf55(lf_path, save_lf_path, size, stride, img_size)
    dslr_h5f = get_h5py_lf55(dslr_path, save_dslr_path, size, stride, img_size)
    lfh_h5f = get_h5py_lf55(lfh_path, save_lfh_path, size, stride, img_size)
    dslrh_h5f = get_h5py_lf55(dslrh_path, save_dslrh_path, size, stride, img_size)

    lf_h5f.close()
    dslr_h5f.close()
    lfh_h5f.close()
    dslrh_h5f.close()



class TrainData(data.Dataset):
    def __init__(self, train_data_dir):
        super().__init__()
        self.path = train_data_dir
        self.patch_size = 128
        self.stride=96
        self.img_size = 512
        self.num = math.ceil((self.img_size - self.patch_size) / self.stride + 1)
        print(self.num)

    def __getitem__(self, index):
        lf_path = os.path.join(self.path, 'lf.h5')
        dslr_path = os.path.join(self.path, 'dslr.h5')

        lfh_path = os.path.join(self.path,'lfh.h5')
        dslrh_path = os.path.join(self.path, 'dslrh.h5')

        lf_h5f = h5py.File(lf_path, 'r')
        dslr_h5f = h5py.File(dslr_path, 'r')

        lfh_h5f = h5py.File(lfh_path, 'r')
        dslrh_h5f = h5py.File(dslrh_path, 'r')

        lf = lf_h5f[str(index)][:]
        dslr = dslr_h5f[str(index)][:]
        lfh = lfh_h5f[str(index)][:]
        dslrh = dslrh_h5f[str(index)][:]

        lf_h5f.close()
        dslr_h5f.close()
        lfh_h5f.close()
        dslrh_h5f.close()

        return lf, dslr, lfh, dslrh

    def __len__(self):
        list_i = os.listdir(os.path.join(self.path, 'dslr'))
        return int(len(list_i) * self.num * self.num)






