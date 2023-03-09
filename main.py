import argparse
from train import train
from visualize import draw

def argparser():
    parser = argparse.ArgumentParser("Pytorch version of Res-Swin models")
    parser.add_argument('--epochs', help='the total episodes', type=int, default=11000)  # data 1 can be about 500
    parser.add_argument('--interval', help='time interval for testing', type=int, default=50)
    parser.add_argument('--nepoch', help='the episodes of scheduler', type=int, default=500)  # data 1 & data 2: 500, data 3: 2000
    parser.add_argument('--warmup_epochs', help='warmup epochs', type=int, default=20)
    parser.add_argument('--lr', help='the learning rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=5e-5)
    parser.add_argument('--data_path', help='data path', type=str, default="/home/s3084361/data/denoise/dataset/global_artifact/npy_imgs_gl")
    parser.add_argument('--save_path', help='save path', type=str, default="/home/s3084361/data/swin_denoise")
    parser.add_argument('--img_path', help='path saving images', type=str, default="/Users/lyudonghang/Downloads/Training_Image_Data/")
    parser.add_argument('--batch_size', help='the batch size', type=int, default=8)
    parser.add_argument('--model', help='model type', type=str, default="res_swin", choices=["red_cnn", "unet34", "unet50","unet34-swin","unet34-mcnn","res_swin_sg", "res_swin_db","transunet","res_swin_dbm"])
    parser.add_argument('--data_type', help='data type', type=str, default="data2", choices=["data1", "data2", "data3"])
    parser.add_argument('--mode', help='choosing mode', type=str, default="train", choices=["train", "visualize"])
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "visualize":
        draw(args, "", "")

if __name__ == '__main__':
    main()
