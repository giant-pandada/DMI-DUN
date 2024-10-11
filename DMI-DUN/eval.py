import os
import time
import torch
import argparse
import torchvision
import numpy as np
import scipy.io as scio
from skimage.metrics import structural_similarity as SSIM
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision
from time import time

import utils
import models


parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.10, type=float)
parser.add_argument("--device", default="0")
opt = parser.parse_args()
opt.device = "cuda:" + opt.device


def evaluate():
    print("Start evaluate...")
    config = utils.GetConfig(ratio=opt.rate, device=opt.device)
    net = models.MsDUF(config).to(config.device).eval()

    if os.path.exists(config.model):
        if torch.cuda.is_available():
            trained_model = torch.load(config.model, map_location=config.device)
        else:
            trained_model = torch.load(config.model, map_location="cpu")

        net.load_state_dict(trained_model)
        print("Trained model loaded.")
    else:
        raise FileNotFoundError("Missing trained models.")

    res(config, net, save_img=True)


def res(config, net, save_img):
    
    tensor2image = torchvision.transforms.ToPILImage()
    save_img = save_img
    batch_size = 1

    net = net.eval()

    file_no = [

        100,

    ]

    folder_name = [

        "General100",

    ]
    transform_image = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Grayscale(num_output_channels=1)
    ])

    for idx, item in enumerate(folder_name):
        p_total = 0
        s_total = 0
        mse_total = 0
        time_total = 0
        path = "{}/".format(config.test_path) + item
        print(path)
        print("*", ("  test dataset: " + path + ", device: " + str(config.device) + "  ").center(120, "="), "*")
        files = os.listdir(path)
        count_all = 0
        with torch.no_grad():
            for file in files:
                count_all = count_all + 1
                print(path + "/" + file)
                image = cv2.imread(path + "/" + file)
                '''x = np.array(image, dtype=np.float32) / 255
                x = np.array(x, dtype=np.float32)
                x = transform_image(x)
                x = x[0]
                '''
                image1 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                x = np.array(image1, dtype=np.float32) / 255
                x = transform_image(x)
                x = x[0]
                #print(x.shape)

                #x = torch.from_numpy(np.array(x)).to(config.device)

                x = x.float().to(config.device)
                ori_x = x

                h = x.size()[0]
                h_lack = 0
                w = x.size()[1]
                w_lack = 0

                if h % config.phi_size != 0:
                    h_lack = config.phi_size - h % config.phi_size
                    temp_h = torch.zeros(h_lack, w).to(config.device)
                    h = h + h_lack
                    x = torch.cat((x, temp_h), 0)

                if w % config.phi_size != 0:
                    w_lack = config.phi_size - w % config.phi_size
                    temp_w = torch.zeros(h, w_lack).to(config.device)
                    w = w + w_lack
                    x = torch.cat((x, temp_w), 1)

                x = torch.unsqueeze(x, 0)
                x = torch.unsqueeze(x, 0)

                idx_h = range(0, h, config.block_size)
                idx_w = range(0, w, config.block_size)
                num_patches = h * w // (config.block_size ** 2)

                temp = torch.zeros(num_patches, batch_size, config.channel, config.block_size, config.block_size)


                ori = x.to(config.device)#[:, :, a:a + config.block_size, b:b + config.block_size].to(config.device)
                start = time()
                #print(ori_y)
                output = net(ori)
                end = time()
                time_total = time_total + end - start
                recon_x = output[:, :, 0:h - h_lack, 0:w - w_lack]

                recon_x = torch.squeeze(recon_x).to("cpu")
                ori_x = torch.squeeze(ori_x).to("cpu")

                mse = np.mean(np.square(recon_x.numpy() - ori_x.numpy()))
                mse_total += mse
                p = 10 * np.log10(1 / mse)
                p_total = p_total + p

                ssim = SSIM(recon_x.numpy(), ori_x.numpy(), data_range=1)
                s_total = s_total + ssim


                print("\r=> process {:2} done! Run time for {} is {:5.4f}, PSNR: {:5.2f}, SSIM: {:5.4f}"
                      .format(count_all,file,(end-start), p, ssim))

                if save_img:
                    img_path = "./results/image/{}/".format(int(config.ratio * 100))
                    if not os.path.isdir("./results/image/"):
                        os.mkdir("./results/image/")
                    if not os.path.isdir(img_path):
                        os.mkdir(img_path)
                        print("\rMkdir {}".format(img_path))
                    recon_x = tensor2image(recon_x)
                    recon_x.save(img_path + "({})_{}_{}.png".format(count_all, p, ssim))

            print("=> All the {:2} images done!, your AVG PSNR: {:5.2f}, AVG SSIM: {:5.4f}, AVG TIME: {:5.4f}"
                  .format(file_no[idx], p_total / file_no[idx], s_total / file_no[idx], time_total / file_no[idx]))


if __name__ == "__main__":
    evaluate()
