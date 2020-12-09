from dataset import dataset_full,dataset
import os
import numpy as np
import glob
from utils import *
from scipy import io
import torch
from torch.nn import Module
from torch import nn
from torch.autograd import Variable
from skimage.io import imsave

"""
No mask training, no deblocking

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
        self.steps = []
        self.register_parameter("A",nn.Parameter(torch.from_numpy(A).float(),requires_grad=True))
        self.register_parameter("Q", nn.Parameter(torch.from_numpy(np.transpose(A)).float(), requires_grad=True))
        for n in range(layer_num):
            self.denoisers.append(Denoiser())
            self.register_parameter("step_" + str(n + 1), nn.Parameter(torch.tensor(1.0),requires_grad=True))
            self.steps.append(eval("self.step_" + str(n + 1)))
        for n,denoiser in enumerate(self.denoisers):
            self.add_module("denoiser_"+str(n+1),denoiser)

    def forward(self, inputs, output_layers):
        H = int(inputs.shape[2]/33)
        L = int(inputs.shape[3]/33)
        S = inputs.shape[0]

        y = self.sampling(inputs)
        X = torch.matmul(self.Q,y)
        for n in range(output_layers):
            step = self.steps[n]
            denoiser = self.denoisers[n]

            z = self.block1(X, y,step)
            noise = denoiser(X)
            X = z - torch.matmul(
                (step * torch.matmul(torch.transpose(self.A,0,1), self.A)) - torch.eye(33 * 33).float().cuda(), noise)

            X = self.together(X,S,H,L)
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
        outputs = torch.matmul(torch.transpose(self.A,0,1),y-torch.matmul(self.A,X))
        outputs = step * outputs + X

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
    for data,_ in train_loader:
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


def get_val_result(model,PhaseNum, save_path, is_cuda=True):
    model.eval()
    with torch.no_grad():
        test_set_path = "./dataset/bsds500/test"
        # test_set_path = "./dataset/Set11"
        test_set_path = glob.glob(test_set_path + '/*.tif')
        ImgNum = len(test_set_path)
        PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
        SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
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
            outputs = model(inputs, PhaseNum)
            outputs = torch.squeeze(outputs)
            if is_cuda:
                outputs = outputs.cpu().data.numpy()
            else:
                outputs = outputs.data.numpy()

            images_recovered = outputs[0:row, 0:col] * 255

            aaa = images_recovered.astype(int)
            bbb = aaa < 0
            aaa[bbb] = 0
            bbb = aaa > 255
            aaa[bbb] = 255

            rec_PSNR = psnr(aaa, Iorg) 
            PSNR_All[0, img_no] = rec_PSNR
            rec_SSIM = compute_ssim(aaa, Iorg)
            SSIM_All[0, img_no] = rec_SSIM
			imgname_for_save = (imgName.split('/')[-1]).split('.')[0]
			imsave(os.path.join(save_path,imgname_for_save+'_'+str(rec_PSNR)+'_'+str(rec_SSIM)+'.jpg'),aaa)

        return np.mean(PSNR_All), np.mean(SSIM_All)


def load_sampling_matrix(CS_ratio):
    path = "../dataset/sampling_matrix"
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
    model_name = "AMP_Net_K_M"

    CS_ratios = [30, 10, 4]
    phase = 6
	save_path = "./results/generated_images"
    for CS_ratio in CS_ratios:
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		sub_save_path = os.path.join(results_saving_path, str(CS_ratio))
		if not os.path.exists(sub_save_path):
			os.mkdir(sub_save_path)
		sub_save_path = os.path.join(results_saving_path, str(phase))
		if not os.path.exists(sub_save_path):
			os.mkdir(sub_save_path)
			
        path = os.path.join("results", model_name, str(CS_ratio), str(phase), "best_model.pkl")

        A = load_sampling_matrix(CS_ratio)

        model = AMP_net_Deblock(phase,A)
        model.cuda()
        model.load_state_dict(torch.load(path))
        print("Start")
        one_psnr, one_ssim = get_val_result(model, phase, is_cuda=True)  # test AMP_net

        print(one_psnr, "dB", one_ssim)
