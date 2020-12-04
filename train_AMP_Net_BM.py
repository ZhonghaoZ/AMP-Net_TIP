from dataset import dataset_full
import os
import numpy as np
import glob
from utils import *
from scipy import io
import torch
from torch.nn import Module
from torch import nn
from torch.autograd import Variable
"""
A + deblocking
AMP-Net-K-BM

"""
class Denoiser(Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 1, 3, padding=1,bias=False))

    def forward(self, inputs):
        inputs = torch.unsqueeze(torch.reshape(torch.transpose(inputs,0,1),[-1,33,33]),dim=1)
        output = self.D(inputs)
        # output=inputs-output
        output = torch.transpose(torch.reshape(torch.squeeze(output),[-1,33*33]),0,1)
        return output

class Deblocker(Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 1, 3, padding=1,bias=False))

    def forward(self, inputs):
        inputs = torch.unsqueeze(inputs,dim=1)
        output = self.D(inputs)
        output = torch.squeeze(output,dim=1)
        return output

class AMP_net_Deblock(Module):
    def __init__(self,layer_num, A):
        super().__init__()
        self.layer_num = layer_num
        self.denoisers = []
        self.deblocks = []
        self.steps = []
        self.register_parameter("A",nn.Parameter(torch.from_numpy(A).float(),requires_grad=True))
        self.register_parameter("Q", nn.Parameter(torch.from_numpy(np.transpose(A)).float(), requires_grad=True))
        for n in range(layer_num):
            self.denoisers.append(Denoiser())
            self.deblocks.append(Deblocker())
            self.register_parameter("step_" + str(n + 1), nn.Parameter(torch.tensor(1.0),requires_grad=True))
            self.steps.append(eval("self.step_" + str(n + 1)))
        for n,denoiser in enumerate(self.denoisers):
            self.add_module("denoiser_"+str(n+1),denoiser)
        for n,deblock in enumerate(self.deblocks):
            self.add_module("deblock_"+str(n+1),deblock)

    def forward(self, inputs, output_layers):
        H = int(inputs.shape[2]/33)
        L = int(inputs.shape[3]/33)
        S = inputs.shape[0]

        y = self.sampling(inputs)
        X = torch.matmul(self.Q,y)
        for n in range(output_layers):
            step = self.steps[n]
            denoiser = self.denoisers[n]
            deblocker = self.deblocks[n]

            z = self.block1(X, y,step)
            noise = denoiser(X)
            X = z - torch.matmul(
                (step * torch.matmul(torch.transpose(self.A,0,1), self.A)) - torch.eye(33 * 33).float().cuda(), noise)

            X = self.together(X,S,H,L)

            X = X - deblocker(X)
            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=1), dim=0)
            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=2), dim=0)
            X = torch.transpose(torch.reshape(X, [-1, 33 * 33]), 0, 1)

        X = self.together(X, S, H, L)
        return torch.unsqueeze(X, dim=1)


    def sampling(self,inputs):
        inputs = torch.squeeze(inputs,dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=1), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=2), dim=0)
        inputs = torch.transpose(torch.reshape(inputs, [-1, 33*33]),0,1)
        outputs = torch.matmul(self.A, inputs)
        return outputs

    def block1(self,X,y,step):
        # X = torch.squeeze(X)
        # X = torch.transpose(torch.reshape(X, [-1, 33 * 33]),0,1)
        outputs = torch.matmul(torch.transpose(self.A,0,1),y-torch.matmul(self.A,X))
        outputs = step * outputs + X
        # outputs = torch.unsqueeze(torch.reshape(torch.transpose(outputs,0,1),[-1,33,33]),dim=1)
        return outputs

    def together(self,inputs,S,H,L):
        inputs = torch.reshape(torch.transpose(inputs,0,1),[-1,33,33])
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H*S, dim=0), dim=2)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
        return inputs


def compute_loss(outputs, target):
    loss = []
    for output in outputs:
        loss.append(torch.mean((output - target) ** 2))
    return loss


def get_final_loss(loss_all):
    output = 0
    for loss in loss_all:
        output += loss
    return output


def get_loss(outputs, noise_all, Xs, H, target, sigma=0.01):
    loss1 = torch.mean((outputs[-1] - target) ** 2)
    loss2 = torch.mean(torch.abs(outputs[-1] - target))
    num = 0
    for n in range(len(noise_all)):
        num += 1
        X = Xs[n]
        noise = noise_all[n]
        loss2 += torch.mean((noise - torch.matmul(H, target - X)) ** 2)

    return loss1, loss2

def train(model, opt, train_loader, epoch, batch_size, CS_ratio,PhaseNum):
    model.train()
    n = 0
    for data in train_loader:
        n = n + 1
        opt.zero_grad()
        data = torch.unsqueeze(data,dim=1)
        data = Variable(data.float().cuda())
        outputs= model(data,PhaseNum)

        loss = torch.mean((outputs-data)**2)
        loss.backward()
        opt.step()
        if n % 25 == 0:
            output = "CS_ratio: %d [%02d/%02d] loss: %.4f " % (
            CS_ratio, epoch, batch_size * n, loss.data.item())
            print(output)


def get_val_result(model,PhaseNum, is_cuda=True):
    model.eval()
    with torch.no_grad():
        test_set_path = "dataset/bsds500/val"
        test_set_path = glob.glob(test_set_path + '/*.tif')
        ImgNum = len(test_set_path)
        PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
        model.eval()
        for img_no in range(ImgNum):
            imgName = test_set_path[img_no]

            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
            Icol = img2col_py(Ipad, 33) / 255.0
            Ipad /= 255.0
            if is_cuda:
                inputs = Variable(torch.from_numpy(Ipad.astype('float32')).cuda())
            else:
                inputs = Variable(torch.from_numpy(Ipad.astype('float32')))
            inputs = torch.unsqueeze(torch.unsqueeze(inputs,dim=0),dim=0)
            outputs = model(inputs, PhaseNumber)
            outputs = torch.squeeze(outputs)
            if is_cuda:
                outputs = outputs.cpu().data.numpy()
            else:
                outputs = outputs.data.numpy()

            images_recovered = outputs[0:row,0:col]
            # images_recovered = col2im_CS_py(output, row, col, row_new, col_new)
            rec_PSNR = psnr(images_recovered * 255, Iorg)
            PSNR_All[0, img_no] = rec_PSNR

    out = np.mean(PSNR_All)
    return out


def load_sampling_matrix(CS_ratio):
    path = "dataset/sampling_matrix"
    data = io.loadmat(os.path.join(path, str(CS_ratio) + '.mat'))['sampling_matrix']
    return data


def get_Q(data_set,A):
    A = torch.from_numpy(A)
    n = 0
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=len(data_set),
                                shuffle=True, num_workers=2)
    for data, target in data_loader:
        data = torch.transpose(torch.reshape(data, [-1, 33 * 33]), 0, 1)
        target = torch.transpose(torch.reshape(target, [-1, 33 * 33]), 0, 1)
        y = torch.matmul(A.float(),data.float())
        x = target.float()
        if n==0:
            ys = y
            Xs = x
            n = 1
        else:
            ys = torch.cat([ys,y],dim=1)
            Xs = torch.cat([Xs,x],dim=1)
    Q = torch.matmul(torch.matmul(Xs,torch.transpose(ys,0,1)),
                     torch.inverse(torch.matmul(ys, torch.transpose(ys, 0, 1))))
    return Q.numpy()


if __name__ == "__main__":
    is_cuda = True
    CS_ratio = 25  # 4, 10, 25, 30, 40, 50
    CS_ratios = [30,10]
    # n_output = 1089
    PhaseNumbers = [9]
    # PhaseNumber = 9
    # nrtrain = 88912
    learning_rate = 0.0001
    EpochNum = 100
    batch_size = 32
    results_saving_path = "results"

    net_name = "AMP_Net_K_BM"

    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    results_saving_path = os.path.join(results_saving_path, net_name)
    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    print('Load Data...')  # jiazaishuju

    train_dataset = dataset_full(train=True, transform=None,
                                 target_transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    for CS_ratio in CS_ratios:
        for PhaseNumber in PhaseNumbers:
            A = load_sampling_matrix(CS_ratio)
            model = AMP_net_Deblock(PhaseNumber,A)
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model.cuda()
            sub_path = os.path.join(results_saving_path, str(CS_ratio))

            if not os.path.exists(sub_path):
                os.mkdir(sub_path)
            sub_path = os.path.join(sub_path, str(PhaseNumber))

            if not os.path.exists(sub_path):
                os.mkdir(sub_path)
            best_psnr = 0
            for epoch in range(1, EpochNum + 1):
                train(model, opt, train_loader, epoch, batch_size, CS_ratio,PhaseNumber)
                one_psnr = get_val_result(model, PhaseNumber)
                print_str = "CS_ratio: %d Phase: %d epoch: %d  psnr: %.4f" % (CS_ratio, PhaseNumber, epoch, one_psnr)
                print(print_str)

                output_file = open(sub_path + "/log_PSNR.txt", 'a')
                output_file.write("PSNR: %.4f\n" % (one_psnr))
                output_file.close()

                if one_psnr > best_psnr:
                    best_psnr = one_psnr
                    output_file = open(sub_path + "/log_PSNR_best.txt", 'a')
                    output_file.write("PSNR: %.4f\n" % (best_psnr))
                    output_file.close()
                    torch.save(model.state_dict(), sub_path + "/best_model.pkl")
