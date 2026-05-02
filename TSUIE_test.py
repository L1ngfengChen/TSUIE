import torch
import time
import argparse
import cv2
import os
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from models.FENet.network import NetWork
from models.FENet.FusionEnhance import FENet
from data.data import get_eval_set
from models.Diffusion.diffusion import GaussianDiffusion
from models.Diffusion.Unet import Unet


parser = argparse.ArgumentParser(description='Fusion^2 Testing')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--data_test_dataset', type=str, default='U90')
parser.add_argument('--T', type=int, default=2000)
parser.add_argument('--linear_start', type=float, default=1e-6)
parser.add_argument('--linear_end', type=float, default=1e-2)
parser.add_argument('--seed', type=int, default=123, help='random seed to use Default=123')
parser.add_argument('--diffusion_model', default='checkpoint/Diffusion.pth')
parser.add_argument('--fusion_model', default='checkpoint/FENet.pth')
parser.add_argument('--output', type=str, default='output')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8)

opt = parser.parse_args()
device = torch.device(opt.device)
cudnn.benchmark = True

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

print('==> Loading datasets')
test_set = get_eval_set(opt.data_test_dataset)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)

print('===> Building models')
Unet = Unet().to(device)
Unet.load_state_dict(torch.load(opt.diffusion_model, map_location=lambda storage, loc: storage))
Sampler = GaussianDiffusion(Unet, opt).to(device)

Net = FENet(depths=[2, 2, 4, 2],num_heads=[2,4,8,4])
Net.load_state_dict(torch.load(opt.fusion_model, map_location=lambda storage, loc: storage))
FusionNet = NetWork(Net, device)
print('===>model is loaded')


def save_img(img, name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    save_dir = opt.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_list = name.split('.', 1)
    save_fn = save_dir + '/' + name_list[0] + '.' + name_list[1]
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def TestModel():
    FusionNet.eval()
    Unet.eval()
    torch.set_grad_enabled(False)
    pbar = tqdm(total=len(test_loader), desc='Testing', unit='img')
    for batch in test_loader:
        with torch.no_grad():
            input, tiny_input, name = Variable(batch[0]), Variable(batch[1]), batch[2]
        if cuda:
            input = input.cuda(device)
            tiny_input = tiny_input.cuda(device)

        with torch.no_grad():
            enhance = Sampler.ddim_sample(tiny_input)
            enhance = F.interpolate(enhance, size=(256,256), mode='bicubic', align_corners=False)
            output = FusionNet(input, enhance, Training = False)

        output = torch.clamp(output, -1, 1)
        output = output*0.5+0.5

        save_img(output.cpu().data, name[0])
        pbar.set_postfix({'name': name[0]})
        pbar.update(1)
    pbar.close()



TestModel()