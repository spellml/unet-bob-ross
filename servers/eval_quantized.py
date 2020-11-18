import os
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torch.optim as optim
import torch.quantization

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import time


class BobRossSegmentedImagesDataset(Dataset):
    def __init__(self, dataroot):        
        super().__init__()
        self.dataroot = dataroot
        self.imgs = list((self.dataroot / 'train' / 'images').rglob('*.png'))
        self.segs = list((self.dataroot / 'train' / 'labels').rglob('*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((164, 164)),
            transforms.Pad(46, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                            mean=(0.459387, 0.46603974, 0.4336706),
                            std=(0.06098535, 0.05802868, 0.08737113)
            )
        ])
        self.color_key = {
            3 : 0,
            5: 1,
            10: 2,
            14: 3,
            17: 4,
            18: 5,
            22: 6,
            27: 7,
            61: 8
        }        
        assert len(self.imgs) == len(self.segs)
        # TODO: remean images to N(0, 1)?
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        def translate(x):
            return self.color_key[x]
        translate = np.vectorize(translate)
        
        img = Image.open(self.imgs[i])
        img = self.transform(img)
        
        seg = Image.open(self.segs[i])
        seg = seg.resize((256, 256), Image.NEAREST)
        
        # Labels are in the ADE20K ontology and are not consequetive,
        # we have to apply a remap operation over the labels in a just-in-time
        # manner. This slows things down, but it's fine, this is just a demo
        # anyway.
        seg = translate(np.array(seg)).astype('int64')
        
        # One-hot encode the segmentation mask.
        # def ohe_mat(segmap):
        #     return np.array(
        #         list(
        #             np.array(segmap) == i for i in range(9)
        #         )
        #     ).astype(int).reshape(9, 256, 256)
        # seg = ohe_mat(seg)
        
        # Additionally, the original UNet implementation outputs a segmentation map
        # for a subset of the overall image, not the image as a whole! With this input
        # size the segmentation map targeted is a (164, 164) center crop.
        seg = seg[46:210, 46:210]
        
        return img, seg

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_1 = torch.quantization.QuantStub()
        self.conv_1_1 = nn.Conv2d(3, 64, 3)
        torch.nn.init.kaiming_normal_(self.conv_1_1.weight)
        self.relu_1_2 = nn.ReLU()
        self.norm_1_3 = nn.BatchNorm2d(64)
        self.conv_1_4 = nn.Conv2d(64, 64, 3)
        torch.nn.init.kaiming_normal_(self.conv_1_4.weight)
        self.relu_1_5 = nn.ReLU()
        self.norm_1_6 = nn.BatchNorm2d(64)
        self.pool_1_7 = nn.MaxPool2d(2)
        
        self.conv_2_1 = nn.Conv2d(64, 128, 3)
        torch.nn.init.kaiming_normal_(self.conv_2_1.weight)        
        self.relu_2_2 = nn.ReLU()
        self.norm_2_3 = nn.BatchNorm2d(128)
        self.conv_2_4 = nn.Conv2d(128, 128, 3)
        torch.nn.init.kaiming_normal_(self.conv_2_4.weight)        
        self.relu_2_5 = nn.ReLU()
        self.norm_2_6 = nn.BatchNorm2d(128)
        self.pool_2_7 = nn.MaxPool2d(2)
        
        self.conv_3_1 = nn.Conv2d(128, 256, 3)
        torch.nn.init.kaiming_normal_(self.conv_3_1.weight)
        self.relu_3_2 = nn.ReLU()
        self.norm_3_3 = nn.BatchNorm2d(256)
        self.conv_3_4 = nn.Conv2d(256, 256, 3)
        torch.nn.init.kaiming_normal_(self.conv_3_4.weight)
        self.relu_3_5 = nn.ReLU()
        self.norm_3_6 = nn.BatchNorm2d(256)
        self.pool_3_7 = nn.MaxPool2d(2)
        
        self.conv_4_1 = nn.Conv2d(256, 512, 3)
        torch.nn.init.kaiming_normal_(self.conv_4_1.weight)
        self.relu_4_2 = nn.ReLU()
        self.norm_4_3 = nn.BatchNorm2d(512)
        self.conv_4_4 = nn.Conv2d(512, 512, 3)
        torch.nn.init.kaiming_normal_(self.conv_4_4.weight)
        self.relu_4_5 = nn.ReLU()
        self.norm_4_6 = nn.BatchNorm2d(512)
        self.dq_1 = torch.quantization.DeQuantStub()
        
        # deconv is the '2D transposed convolution operator'
        self.deconv_5_1 = nn.ConvTranspose2d(512, 256, (2, 2), 2)
        # 61x61 -> 48x48 crop
        self.c_crop_5_2 = lambda x: x[:, :, 6:54, 6:54]
        self.concat_5_3 = lambda x, y: torch.cat((x, y), dim=1)
        self.q_2 = torch.quantization.QuantStub()
        self.conv_5_4 = nn.Conv2d(512, 256, 3)
        torch.nn.init.kaiming_normal_(self.conv_5_4.weight)        
        self.relu_5_5 = nn.ReLU()
        self.norm_5_6 = nn.BatchNorm2d(256)
        self.conv_5_7 = nn.Conv2d(256, 256, 3)
        torch.nn.init.kaiming_normal_(self.conv_5_7.weight)
        self.relu_5_8 = nn.ReLU()
        self.norm_5_9 = nn.BatchNorm2d(256)
        self.dq_2 = torch.quantization.DeQuantStub()
        
        self.deconv_6_1 = nn.ConvTranspose2d(256, 128, (2, 2), 2)
        # 121x121 -> 88x88 crop
        self.c_crop_6_2 = lambda x: x[:, :, 17:105, 17:105]
        self.concat_6_3 = lambda x, y: torch.cat((x, y), dim=1)
        self.q_3 = torch.quantization.QuantStub()
        self.conv_6_4 = nn.Conv2d(256, 128, 3)
        torch.nn.init.kaiming_normal_(self.conv_6_4.weight)
        self.relu_6_5 = nn.ReLU()
        self.norm_6_6 = nn.BatchNorm2d(128)
        self.conv_6_7 = nn.Conv2d(128, 128, 3)
        torch.nn.init.kaiming_normal_(self.conv_6_7.weight)
        self.relu_6_8 = nn.ReLU()
        self.norm_6_9 = nn.BatchNorm2d(128)
        self.dq_3 = torch.quantization.DeQuantStub()
        
        self.deconv_7_1 = nn.ConvTranspose2d(128, 64, (2, 2), 2)
        # 252x252 -> 168x168 crop
        self.c_crop_7_2 = lambda x: x[:, :, 44:212, 44:212]
        self.concat_7_3 = lambda x, y: torch.cat((x, y), dim=1)
        self.q_4 = torch.quantization.QuantStub()
        self.conv_7_4 = nn.Conv2d(128, 64, 3)
        torch.nn.init.kaiming_normal_(self.conv_7_4.weight)
        self.relu_7_5 = nn.ReLU()
        self.norm_7_6 = nn.BatchNorm2d(64)
        self.conv_7_7 = nn.Conv2d(64, 64, 3)
        torch.nn.init.kaiming_normal_(self.conv_7_7.weight)        
        self.relu_7_8 = nn.ReLU()
        self.norm_7_9 = nn.BatchNorm2d(64)
        
        # 1x1 conv ~= fc; n_classes = 9
        self.conv_8_1 = nn.Conv2d(64, 9, 1)
        self.dq_4 = torch.quantization.DeQuantStub()
        
        # residual connections need to be dequantized seperately
        self.dq_resid_1 = torch.quantization.DeQuantStub()
        self.dq_resid_2 = torch.quantization.DeQuantStub()
        self.dq_resid_3 = torch.quantization.DeQuantStub()
        

    def forward(self, x):
        x = self.q_1(x)        
        x = self.conv_1_1(x)
        x = self.relu_1_2(x)
        x = self.norm_1_3(x)
        x = self.conv_1_4(x)
        x = self.relu_1_5(x)
        x_resid_1_quantized = self.norm_1_6(x)
        x = self.pool_1_7(x_resid_1_quantized)
        x_resid_1 = self.dq_resid_1(x_resid_1_quantized)
        
        x = self.conv_2_1(x)
        x = self.relu_2_2(x)
        x = self.norm_2_3(x)
        x = self.conv_2_4(x)
        x = self.relu_2_5(x)
        x_resid_2_quantized = self.norm_2_6(x)
        x = self.pool_2_7(x_resid_2_quantized)
        x_resid_2 = self.dq_resid_2(x_resid_2_quantized)
        
        x = self.conv_3_1(x)
        x = self.relu_3_2(x)
        x = self.norm_3_3(x)
        x = self.conv_3_4(x)
        x = self.relu_3_5(x)
        x_resid_3_quantized = self.norm_3_6(x)
        x = self.pool_3_7(x_resid_3_quantized)
        x_resid_3 = self.dq_resid_3(x_resid_3_quantized)
        
        x = self.conv_4_1(x)
        x = self.relu_4_2(x)
        x = self.norm_4_3(x)        
        x = self.conv_4_4(x)
        x = self.relu_4_5(x)
        x = self.norm_4_6(x)
        x = self.dq_1(x)
        
        x = self.deconv_5_1(x)
        x = self.concat_5_3(self.c_crop_5_2(x_resid_3), x)
        x = self.q_2(x)
        x = self.conv_5_4(x)
        x = self.relu_5_5(x)
        x = self.norm_5_6(x)
        x = self.conv_5_7(x)
        x = self.relu_5_8(x)
        x = self.norm_5_9(x)
        x = self.dq_2(x)
        
        x = self.deconv_6_1(x)
        x = self.concat_6_3(self.c_crop_6_2(x_resid_2), x)
        x = self.q_3(x)
        x = self.conv_6_4(x)
        x = self.relu_6_5(x)
        x = self.norm_6_6(x)
        x = self.conv_6_7(x)
        x = self.relu_6_8(x)
        x = self.norm_6_9(x)
        x = self.dq_3(x)
        
        x = self.deconv_7_1(x)
        x = self.concat_7_3(self.c_crop_7_2(x_resid_1), x)
        x = self.q_4(x)
        x = self.conv_7_4(x)
        x = self.relu_7_5(x)
        x = self.norm_7_6(x)
        x = self.conv_7_7(x)
        x = self.relu_7_8(x)
        x = self.norm_7_9(x)
        
        x = self.conv_8_1(x)
        x = self.dq_4(x)        
        return x

def main():
    print("Starting up...")
    dataroot = Path('/mnt/segmented-bob-ross-images/')
    dataset = BobRossSegmentedImagesDataset(dataroot)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=8)

    print("Loading the model...")
    model = UNet()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    checkpoints_dir = '/spell/checkpoints'
    model.load_state_dict(
        torch.load(f"{checkpoints_dir}/model_50.pth", map_location=torch.device('cpu'))
    )
    model.eval()
    
    # NEW
    # NOTE(aleksey): we could potentially speed this up even more by switching from
    # conv->relu->batchnorm order to conv->batchnorm->relu order. PyTorch curently supports
    # conv->batchnorm->relu fusion *only*.
    #
    # Which placement of the relu layer is optimal is a subject of academic debate. The order
    # that the model *currently* uses seems to be the more popular option. I am not swapping the
    # order of the operations out of laziness -- but you can probably speed things up a little
    # bit more by going ahead and making that more invasive change.
    model = torch.quantization.fuse_modules(
        model,
        [
            ['conv_1_1', 'relu_1_2'],
            ['conv_1_4', 'relu_1_5'],
            ['conv_2_1', 'relu_2_2'],
            ['conv_2_4', 'relu_2_5'],
            ['conv_3_1', 'relu_3_2'],
            ['conv_3_4', 'relu_3_5'],
            ['conv_4_1', 'relu_4_2'],
            ['conv_4_4', 'relu_4_5'],
        ]
    )
    model = torch.quantization.prepare(model)
    print(f"Quantizing the model...")
    start_time = time.time()
    
    for i, (batch, segmap) in enumerate(dataloader):
        # batch = batch.cuda()
        # segmap = segmap.cuda()
        model(batch)

    model = torch.quantization.convert(model)
    print(f"Quantization done in {str(time.time() - start_time)} seconds.")

    print(f"Evaluating the model...")
    start_time = time.time()
    for i, (batch, segmap) in enumerate(dataloader):
        # batch = batch.cuda()
        # segmap = segmap.cuda()
        model(batch)
    
    print(f"Evaluation done in {str(time.time() - start_time)} seconds.")


if __name__ == "__main__":
    main()
