import os
import cv2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import torch.optim as optim
from net.network import LCFSC
from data.datasets import get_loader
from utils import *
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torch.distributed as dist
import torch.nn as nn
import scipy.io as scio
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='LCFSC')
parser.add_argument('--training', default='True',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='UDIS-train',
                    choices=['UDIS-train', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='UDIS-test',
                    choices=['UDIS-test', 'CLIC21'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MS-SSIM',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='LCFSC',
                    choices=['', 'LCFSC_W/O'],
                    help='LCFSC model')
parser.add_argument('--channel-type', type=str, default='MIMO',
                    choices=['awgn', 'rayleigh', 'MIMO'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=32*3,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                    help='random or fixed snr')
parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='Use NVIDIA GPU acceleration')
parser.add_argument('--output-dir', type=str, default='output',
                    help='Output directory to save the images')
# parser.add_argument('--local_rank', default=0, type=int, help='rank of distributed processes')

args = parser.parse_args()

class config():
    local_rank = int(os.environ["LOCAL_RANK"])
    seed = 1024
    pass_channel = True
    CUDA = True
    multi_gpu = True
    norm = False
    # logger
    print_step = 10
    plot_step = 1000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 5000

    args.training = True
    args.trainset = 'UDIS-train'
    args.testset = 'UDIS-test'

    print(torch.cuda.is_available())
    # device = torch.device('cuda')
    
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    # 分布式初始化
    args.dist_url = 'env://'  # 设置url
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    dist.barrier()  # 等待所有进程都初始化完毕，即所有GPU都要运行到这一步以后在继续
        
    if args.trainset == 'UDIS-train':
        save_model_freq = 20
        image_dims = (3, 128, 128)
        train_data_dir = 'TWC/datasets/UDIS-D-TWC/train'
        test_data_dir = 'TWC/datasets/UDIS-D-TWC/test'

        batch_size = 16
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=4, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 6, 2, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 2, 6, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )

CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained


def train_one_epoch(args, H_MIMO, training_epoch):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'UDIS-train':
        for batch_idx, (input ,_) in enumerate(train_loader):
            start_time = time.time()
            input = input.to(config.device)
            global_step += 1
            recon_image, res_image, CBR, SNR, mse, loss_G, loss_F, loss_KL = net(input, H_MIMO, 0, training_epoch, 1)

            if training_epoch >= 3000:
                loss = loss_G + 1 * (loss_F + loss_KL) 
            else:
                loss = loss_G 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - loss_G
                msssims.update(msssim)

            else:
                psnrs.update(100)
                msssims.update(100)
            
            if (global_step % config.print_step) == 0:
                if config.local_rank == 0:
                    process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                    log = (' | '.join([
                        f'Epoch {epoch}',
                        f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    ]))

                    logger.info(log)
                    for i in metrics:
                        i.clear()

    for i in metrics:
        i.clear()

def test(H_MIMO):
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    
    with torch.no_grad():
        if args.testset == 'UDIS-test':
            for batch_idx, (input, label) in enumerate(test_loader):
                start_time = time.time()
                input = input.to(config.device)
                recon_image, res_image, CBR, SNR, mse, loss_G, loss_F, loss_KL = net(input, H_MIMO, 0, global_step, 0)
                            
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnrs.update(100)
                    msssims.update(100)

                log = (' | '.join([
                    f'Time {elapsed.val:.3f}',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f}',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                ]))
                if config.local_rank == 0:
                    logger.info(log)


        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg

        for t in metrics:
            t.clear()

    if config.local_rank == 0:
        print("SNR: {}" .format(results_snr.tolist()))
        print("LOSS: {}" .format(results_snr.tolist()))
        print("CBR: {}".format(results_cbr.tolist()))
        print("PSNR: {}" .format(results_psnr.tolist()))
        print("MS-SSIM: {}".format(results_msssim.tolist()))
        print("Finish Test!")

    return results_psnr[-1]

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed_torch()
    if config.local_rank == 0:
        logger = logger_configuration(config, save_log=True)
        logger.info(config.__dict__)

    f = 'H_MIMO.mat'
    H = scio.loadmat(f)
    H = np.complex64(H['Hnew'])
    H = torch.from_numpy(H)
    #H1 = sig(H.real)
    #H2 = sig(H.imag)
    #H = H1 + 1j*H2
    index = torch.randperm(H.shape[2])
    H = H[:,:, index]
    H_train = H[:, :, 0:1000]
    H_test = H[:, :, 100:H.shape[2]]
    print(H.shape,H_test.shape)

    net = WITT(args, config)

    model_path = "./TWC/model_sv/CSI_UDIS.model"
    # load_weights(model_path)

    
    if config.multi_gpu:
        net = net.to(config.device)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=True)
    else:
        net = net.to(config.device)

    global_step = 0
    steps_epoch = 0
    psnrs = []
    p = 0
    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            # optimizer = torch.optim.Adam(net.parameters(), lr=0.00005, weight_decay=0.)
            if epoch <=600:
                optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.)
            elif epoch <=1200:
                optimizer = torch.optim.Adam(net.parameters(), lr=0.00006, weight_decay=0.)
            else:
                optimizer = torch.optim.Adam(net.parameters(), lr=0.00002, weight_decay=0.)

            train_loader, test_loader = get_loader(args, config, epoch)
            train_one_epoch(args, H_train, epoch)
            if (epoch + 1) % config.save_model_freq == 0:
                torch.save(net.module.state_dict(), model_path)
                psnr = test(H_test)
 
            torch.cuda.empty_cache()  # 释放显存
    else:
        psnr = test(H_test)
        psnrs = [psnrs, psnr]