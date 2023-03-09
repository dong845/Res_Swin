import os
import random
import torch
import torch.optim as optim
import numpy as np
import albumentations as A
from glob import glob
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader, Dataset
from measure import compute_PSNR, compute_SSIM

from models.transunet import TransUNet
from models.unet import Unet_34, Unet_50
from models.res_swin_dbm import Res_Swin_DBM
from models.res_swin_sg import Res_Swin_SG
from models.res_swin_db import Res_Swin_DB
from models.compare_models import Unet34_Swin, Unet34_MCNN
from models.red_cnn import RED_CNN

img_size = (512, 512)
model_dict = {
    "red_cnn": RED_CNN(),
    "unet34": Unet_34(),
    "unet50": Unet_50(),
    "unet34_swin": Unet34_Swin(),
    "unet34_mcnn": Unet34_MCNN(),
    "res_swin_sg": Res_Swin_SG(),
    "res_swin_db": Res_Swin_DB(), 
    "transunet": TransUNet(img_dim=512,in_channels=1,out_channels=128,head_num=4,mlp_dim=512,block_num=8,patch_dim=16,class_num=1),
    "res_swin_dbm": Res_Swin_DBM()
}

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     # torch.backends.cudnn.deterministic = True
setup_seed(0)

def statistics(args):
    path = args.data_path
    target_path = sorted(glob(os.path.join(path, '*target*.npy')))
    mx = float("inf")
    mn = float("-inf")
    for f in target_path:
        img = np.load(f)
        mx = max(mx, np.max(img))
        mn = min(mn, np.min(img))
    return mx, mn

class CT_Dataset(Dataset):
    def __init__(self, mode, saved_path, test_patient, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"

        input_path = sorted(glob(os.path.join(saved_path, '*input*.npy')))
        target_path = sorted(glob(os.path.join(saved_path, '*target*.npy')))
        self.transform = transform

        if mode == "train":
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            self.input_ = input_
            self.target_ = target_
        elif mode == "test":
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            self.input_ = input_
            self.target_ = target_

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        input_img, target_img = np.float32(np.load(input_img)), np.float32(np.load(target_img))
        augmentations = self.transform(image=input_img, mask=target_img)
        image = augmentations["image"]
        label = augmentations["mask"]
        return image, label

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    ToTensorV2()
])

test_transform = A.Compose([
    ToTensorV2()
])

best_psnr = 0
best_ssim = 0
def test(args, model, test_dataset, mx, mn):
    global best_psnr
    global best_ssim
    psnrs = []
    ssims = []
    imgs = []
    names = []
    pkl_save = args.save_path+"/train_files_gl/"+args.model
    img_save = args.save_path+"/final_save_cts_gl/"+args.model
    record_save = args.save_path+"/records_gl/"+args.model
    if args.device=="cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    model.eval()
    with torch.no_grad():
        for i, (img, label) in enumerate(test_dataset):
            img = img.unsqueeze(0).float()
            pred = model(img.to(device))
            pred=pred.cpu()
            pred[pred<mn] = mn
            pred[pred>mx] = mx
            pred_new = pred.numpy().squeeze(0)
            pred_new = pred_new.reshape(512,512)
            label_new = label.cpu().numpy()
            label_new = label_new.reshape(512,512)
            img_name = test_dataset.target_[i]
            image_name = img_name.split('/')[-1]
            out_path = os.path.join(img_save, image_name)
            names.append(out_path)
            imgs.append(pred_new)
            psnrs.append(compute_PSNR(label_new, pred_new, data_range=mx-mn))
            ssims.append(compute_SSIM(label, pred, data_range=mx-mn)) 
    print("PSNR:", np.mean(np.array(psnrs)))
    print("SSIM:", np.mean(np.array(ssims)))
    
    with open(record_save+"/psnr.txt",'a') as f:
        temp_psnr = str(np.mean(np.array(psnrs)))
        f.write(temp_psnr+'\n')
    with open(record_save+"/ssim.txt",'a') as f1:
        temp_ssim = str(np.mean(np.array(ssims)))
        f1.write(temp_ssim+'\n')
    pt = np.mean(np.array(psnrs))
    st = np.mean(np.array(ssims))
    if pt>best_psnr and st>best_ssim:
        best_psnr = pt
        best_ssim = st
        for i in range(len(names)):
            np.save(names[i], imgs[i])
        path_file = os.path.join(pkl_save,"weight.pkl")
        torch.save(model.state_dict(), path_file)
    print("best PSNR:", best_psnr)
    print("best SSIM:", best_ssim)

def train(args):
    train_dataset = CT_Dataset("train",saved_path=args.data_path, test_patient="test",transform=train_transform)
    test_dataset = CT_Dataset("test",saved_path=args.data_path, test_patient="test",transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    record_save = args.save_path+"/records_gl/"+args.model
    
    mx, mn = statistics(args)
    if args.device=="cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    batch_size = args.batch_size
    model = model_dict[args.model].to(device)
    
    model.train()
    nepoch=args.nepoch
    warmup_epochs = args.warmup_epochs
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,betas=(0.9,0.999),eps=1e-8, weight_decay=args.weight_decay).to(device)

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,nepoch-warmup_epochs,eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer,
            multiplier=1,total_epoch=warmup_epochs,
            after_scheduler=scheduler_cosine)
    
    for epoch in range(args.epochs+1):
        losses = 0
        for i, (image, target) in enumerate(train_loader):
            image = image.to(device)
            target = target.unsqueeze(1).to(device)
            pred = model(image)
            loss = F.mse_loss(pred, target)
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print("epoch:", epoch, "loss:", float(losses / len(train_dataset)))
        
        if epoch % args.interval == 0:
            with open(record_save+'/loss.txt','a') as f:
                temp_loss = str(float(losses / len(train_dataset)))
                f.write(temp_loss+'\n')
            test(args, model, test_dataset, mx, mn)


