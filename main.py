import argparse
from train import train

def argparser():
    parser = argparse.ArgumentParser("Pytorch of Res_Swin")
    parser.add_argument('--epochs', help='the total episodes', type=int, default=11000)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--stop_epoch', help='the episodes of saving in test process', type=int, default=11000)
    parser.add_argument('--nepoch', help='the episodes of scheduler', type=int, default=2000)  # data 1: 1000 data 2: 500 data 3: 2000
    parser.add_argument('--warmup_epochs', help='warmup epochs', type=int, default=20)
    parser.add_argument('--lr', help='the learning rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=5e-5)
    parser.add_argument('--data_path', help='data path', type=str, default="/home/s3084361/data/denoise/dataset/global_artifact/npy_imgs_g")
    parser.add_argument('--save_path', help='save path', type=str, default="/home/s3084361/data/swin_denoise")
    parser.add_argument('--batch_size', help='the batch size', type=int, default=8)
    parser.add_argument('--model', help='model type', type=str, default="res_swin")
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
