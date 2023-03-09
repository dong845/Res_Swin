import matplotlib.pyplot as plt
import numpy as np
import os

# clipped area
def img_show(path):
    gap = 60
    img = np.load(path)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    img_small = img[256 - gap:256 + gap, 256 - gap:256 + gap]
    plt.imshow(img_small, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# clip the getting image within value range of label one
def img_show_res(path, label_path):
    gap = 60
    label_img = np.load(label_path)
    mx = np.max(label_img)
    mn = np.min(label_img)
    img = np.load(path)
    img[img<mn] = mn
    img[img>mx] = mx
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    img_small = img[256 - gap:256 + gap, 256 - gap:256 + gap]
    plt.imshow(img_small, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def draw(args, input_file, target_file):
    if args.dataset == "data1":
        overall_path = args.img_path+"/data1/res_save_ct"
        label_path = args.img_path+"/data1/np_img_L506"
        input_path = os.path.join(label_path, input_file)
        target_path = os.path.join(label_path, target_file)
        res_path = os.path.join(overall_path, target_file)
    elif args.dataset == "data2":
        overall_path = args.img_path+"/data2/"+args.model
        label_path = args.img_path+"/data2/npy_imgs"
        input_path = os.path.join(label_path, input_file)
        target_path = os.path.join(label_path, target_file)
        res_path = os.path.join(overall_path, target_file)
    else:
        overall_path = args.img_path+"/data3/final_save_cts_gl/"+args.model
        label_path = args.img_path+"/data3/npy_imgs_gl_target"
        input_path = args.img_path+"/data3/npy_imgs_gl_input"
        input_path = os.path.join(input_path, input_file)
        target_path = os.path.join(label_path, target_file)
        res_path = os.path.join(overall_path, target_file)
    img_show(input_path)
    img_show(target_path)
    img_show_res(res_path, target_path)
    
