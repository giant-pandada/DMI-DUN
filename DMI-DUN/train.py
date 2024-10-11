import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data
#import scipy.io as scio
import torch.optim as optim
#import torchsummary.torchsummary
import torch.optim.lr_scheduler as LS

from skimage.metrics import structural_similarity as SSIM
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import cv2
#from torchstat import stat



import models
import utils

parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.1, type=float)
parser.add_argument("--bs", default=16, type=int)
parser.add_argument("--device", default="0")
parser.add_argument("--time", default=0, type=int)
opt = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val_p(config, net):
    """net = net.eval()
    # with torch.no_grad():
    transform_image = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1)])
    # image = Image.open("/home/shen/BSD500/val/108005.jpg").convert('L')
    image = Image.open("C:/home/shen/BSD500/val/108005.jpg")
    x = np.array(image, dtype=np.float32) / 255
    x = transform_image(x)
    x = x[0]

    p_total = 0
    over_all_time = time.time()

    x = x.float().to(config.device)
    ori_x = x
    batch_size = 1
    h = x.size()[0]
    h_lack = 0
    w = x.size()[1]
    w_lack = 0

    if h % config.block_size != 0:
        h_lack = config.block_size - h % config.block_size
        temp_h = torch.zeros(h_lack, w).to(config.device)
        h = h + h_lack
        x = torch.cat((x, temp_h), 0)

    if w % config.block_size != 0:
        w_lack = config.block_size - w % config.block_size
        temp_w = torch.zeros(h, w_lack).to(config.device)
        w = w + w_lack
        x = torch.cat((x, temp_w), 1)

    x = torch.unsqueeze(x, 0)
    x = torch.unsqueeze(x, 0)

    idx_h = range(0, h, config.block_size)
    idx_w = range(0, w, config.block_size)
    num_patches = h * w // (config.block_size ** 2)

    temp = torch.zeros(num_patches, batch_size, config.channel, config.block_size, config.block_size)
    count = 0
    for a in idx_h:
        for b in idx_w:
            output = net(x[:, :, a:a + config.block_size, b:b + config.block_size])
            temp[count, :, :, :, :, ] = output
            count = count + 1

    y = torch.zeros(batch_size, config.channel, h, w)
    count = 0
    for a in idx_h:
        for b in idx_w:
            y[:, :, a:a + config.block_size, b:b + config.block_size] = temp[count, :, :, :, :]
            count = count + 1

    recon_x = y[:, :, 0:h - h_lack, 0:w - w_lack]

    recon_x = torch.squeeze(recon_x).to("cpu")
    ori_x = ori_x.to("cpu")

    mse = np.mean(np.square(recon_x.detach().numpy() - ori_x.detach().numpy()))
    p = 10 * np.log10(1 / mse)
    ssim = SSIM(recon_x.detach().numpy(), ori_x.detach().numpy(), data_range=1)
    print('PSNR', p)
    print('SSIM', ssim)

    plt.figure(1)  # 图名
    plt.imshow(recon_x.detach().numpy(), cmap='gray')  # cmap即colormap，颜色映射
    # plt.gca().invert_yaxis()
    plt.figure(2)  # 图名
    plt.imshow(ori_x.detach().numpy(), cmap='gray')  # cmap即colormap，颜色映射
    plt.show()"""
    batch_size = 1
    torch.cuda.empty_cache()
    net = net.eval()
    file_no = [

        14,

    ]

    folder_name = [

        "Set14",

    ]
    transform_image = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1),
        ])

    for idx, item in enumerate(folder_name):
        p_total = 0
        path = "{}/".format(config.test_path) + item
        print(path)
        print("*", ("  test dataset: " + path + ", device: " + str(config.device) + "  ").center(120, "="), "*")
        files = os.listdir(path)
        count_all = 0
        with torch.no_grad():
            for file in files:
                count_all = count_all + 1
                print(path + "/" + file)
                #image = Image.open(path + "/" + file)
                image = cv2.imread(path + "/" + file)
                #image = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
                x = np.array(image, dtype=np.float32) / 255
                x = np.array(x, dtype=np.float32)
                x = transform_image(x)
                x = x[0]
                # x = torch.from_numpy(np.array(x)).to(config.device)

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

                ori = x.to(config.device) #[:, :, a:a + config.block_size, b:b + config.block_size].to(config.device)
                output = net(ori)

                recon_x = output[:, :, 0:h - h_lack, 0:w - w_lack]

                recon_x = torch.squeeze(recon_x).to("cpu")
                ori_x = ori_x.to("cpu")

                mse = np.mean(np.square(recon_x.numpy() - ori_x.numpy()))
                #ssim = SSIM(recon_x.detach().numpy(), ori_x.detach().numpy(), data_range=1)
                p = 10 * np.log10(1 / mse)
                p_total = p_total + p #+ ssim

            return p_total / file_no[idx]


def main():
    device = "cuda:" + opt.device
    config = utils.GetConfig(ratio=opt.rate, device=device)
    config.check()
    set_seed(22)
    print("Data loading...")
    torch.cuda.empty_cache()
    dataset_train = utils.train_loader(batch_size=opt.bs)
    net = models.MsDUF(config).to(config.device)


    optimizer = optim.AdamW(net.parameters(), lr=20e-5)

    if os.path.exists(config.model):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(config.model, map_location=config.device))
            info = torch.load(config.info, map_location=config.device)
            optimizer.load_state_dict(torch.load(config.optimizer, map_location=config.device))
        else:
            net.load_state_dict(torch.load(config.model, map_location="cpu"))
            info = torch.load(config.info, map_location="cpu")
            optimizer.load_state_dict(torch.load(config.optimizer, map_location="cpu"))

        start_epoch = info["epoch"]
        best = info["res"]
        print("Loaded trained model of epoch {:2}, res: {:8.4f}.".format(start_epoch, best))
    else:
        start_epoch = 1
        best = 0
        print("No saved model, start epoch = 1.")
    net.train()
    #scheduler = LS.MultiStepLR(optimizer, milestones=[25, 35, 45], gamma=0.1)
    scheduler = LS.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, threshold=0.0001,
                                     threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    over_all_time = time.time()
    """scheduler1 = LS.StepLR(optimizer, 1, gamma=10)
    batch_loss1=[]
    LR=[]
    best_loss = 1e9"""
    for epoch in range(start_epoch, 80):
        print("Lr: op {}.".format(optimizer.param_groups[0]['lr']))

        epoch_loss = 0
        dic = {"rate": config.ratio, "epoch": epoch,
               "device": config.device}
        for idx, xi in enumerate(tqdm(dataset_train, desc="Now training: ", postfix=dic)):
            xi = xi.to(config.device)
            optimizer.zero_grad()
            xo = net(xi)
            batch_loss = torch.mean(torch.pow(xo - xi, 2)).to(config.device)
            #batch_loss1.append(batch_loss)
            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            #p = val_p(config, net)
            #print(p)
            """LR.append(optimizer.param_groups[0]['lr'])
            scheduler1.step()
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            if batch_loss < best_loss:
                best_loss = batch_loss
            if batch_loss > 4 * best_loss or optimizer.param_groups[0]['lr'] > 1.:
                break"""

            if idx % 10 == 0:
                tqdm.write("\r[{:5}/{:5}], Loss: [{:8.6f}]"
                           .format(config.batch_size * (idx + 1),
                                   dataset_train.__len__() * config.batch_size,
                                   batch_loss.item()))
        """LR = torch.tensor(LR).to("cpu")
        batch_loss1 = torch.tensor(batch_loss1).to("cpu")
        LR=np.array(LR)
        batch_loss1 = np.array(batch_loss1)
        plt.figure()
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.plot(np.log(LR)//np.log(10), batch_loss1)
        plt.show()"""

        avg_loss = epoch_loss / dataset_train.__len__()
        print("\n=> Epoch of {:2}, Epoch Loss: [{:8.6f}]"
              .format(epoch, avg_loss))

        if epoch == 1:
            if not os.path.isfile(config.log):
                output_file = open(config.log, 'w')
                output_file.write("=" * 120 + "\n")
                output_file.close()
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)
            output_file.write("\nAbove is {} test. Note：{}.\n"
                              .format("???", None) + "=" * 120 + "\n")
            output_file.write(old)
            output_file.close()

        print("\rNow val..")
        p = val_p(config, net)
        print("{:5.3f}".format(p))
        if p > best:
            info = {"epoch": epoch, "res": p}
            torch.save(net.state_dict(), config.model)
            torch.save(optimizer.state_dict(), config.optimizer)
            torch.save(info, config.info)
            print("*", "  Check point of epoch {:2} saved  ".format(epoch).center(120, "="), "*")
            best = p
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)

            output_file.write("Epoch {:2.0f}, Loss of train {:8.10f}, Res {:2.3f}\n".format(epoch, avg_loss, best))
            output_file.write(old)
            output_file.close()

        scheduler.step(p)
        print("Over all time: {:.3f}s".format(time.time() - over_all_time))
    print("Train end.")


def gpu_info():
    memory = int(os.popen('nvidia-smi | grep %').read()
                 .split('C')[int(opt.device) + 1].split('|')[1].split('/')[0].split('MiB')[0].strip())
    return memory


if __name__ == "__main__":
    main()
