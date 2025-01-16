import faulthandler

faulthandler.enable()

import torch
import utility
import loss

torch.backends.cudnn.enabled = False
import argparse
from mydata import FlouresceneVCD, Flouresceneproj
from torch.utils.data import dataloader
import model

import os
from decimal import Decimal
import torch.nn.utils as utils
# import imageio
from utility import savecolorim
import numpy as np
from mydata import normalize, PercentileNormalizer
# from tifffile import imsave

rp = os.path.dirname(__file__)


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--test_only', action='store_true', default=test_only, help='set this option to test the model')
    parser.add_argument('--task', type=int, default=task)
    parser.add_argument('--resume', type=int, default=0, help='-2:best;-1:latest; 0:pretrain; >0: resume')
    parser.add_argument('--pre_train', type=str, default=pre_train, help='pre-trained model directory')
    parser.add_argument('--save', type=str, default=savename, help='_itefile name to save')
    
    # Data specifications
    parser.add_argument('--test_every', type=int, default=test_every)
    parser.add_argument('--print_every', type=int, default=100, help='')
    parser.add_argument('--data_test', type=str, default=testset, help='demo image directory')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=batch, help='input batch size for training')
    parser.add_argument('--patch_size', type=int, default=patch, help='input batch size for training')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--n_colors', type=int, default=1, help='')
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)
    # Loss specifications
    parser.add_argument('--loss', type=str, default='1*L1', help='loss function configuration')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay', type=str, default='100', help='learning rate decay type')
    
    parser.add_argument('--cpu', action='store_true', default=False, help='')
    parser.add_argument('--load', type=str, default='', help='file name to load')
    
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    parser.add_argument('--n_resblocks', type=int, default=8, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=32, help='number of feature maps')
    parser.add_argument('--save_models', action='store_true', default=True, help='save all intermediate models')
    
    parser.add_argument('--scale', type=str, default='1', help='super resolution scale')
    parser.add_argument('--chop', action='store_true', default=True, help='enable memory-efficient forward')
    parser.add_argument('--self_ensemble', action='store_true', help='use self-ensemble method for test')
    
    # Model specifications
    parser.add_argument('--act', type=str, default='relu', help='activation function')
    parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')
    parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'), help='FP precision for test (single | half)')
    
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    # Optimization specifications
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAM',
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')
    
    args = parser.parse_args()
    
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    
    return args


def get_data_loader(t, testonly=False):
    if testonly:
        loader_train = None
    
    if not testonly:
        loader_train = dataloader.DataLoader(
            Flouresceneproj(patch_size=args.patch_size, name=testset, istrain=True,
                            condition=condition, rootdatapath=prodatapath),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)
    loader_test = [dataloader.DataLoader(
        Flouresceneproj(patch_size=args.patch_size, name=testset, istrain=False,
                        condition=condition, rootdatapath=prodatapath),
        batch_size=1,
        shuffle=False,
        pin_memory=not args.cpu,
        num_workers=0)]

    # elif t == 5:
    #    if not testonly:
    #        loader_train = dataloader.DataLoader(
    #            FlouresceneVCD(patch_size=args.patch_size, istrain=True, subtestset=subtestset,
    #                           test_only=False, rootdatapath=voldatapath),
    #            batch_size=args.batch_size,
    #            shuffle=False,
    #            pin_memory=not args.cpu,
    #            num_workers=0)
    #    loader_test = [dataloader.DataLoader(
    #        FlouresceneVCD(patch_size=args.patch_size, istrain=False, subtestset=subtestset,
    #                       test_only=testonly, rootdatapath=voldatapath),
    #        batch_size=1,
    #        shuffle=False,
    #        pin_memory=not args.cpu,
    #        num_workers=0)]
       

    return loader_train, loader_test


def train():
    _model = model.Model(args, checkpoint, unimodel, epochall=-1, dataper=1.0, rp=rp)
    _loss = loss.Loss(args, checkpoint)
    loader_train, loader_test = get_data_loader(t=task)
    
    t = Trainer(args, loader_train, loader_test, args.data_test, _model, _loss, checkpoint)
    while t.terminate():
        t.trainUni(tsk=task)
    
    checkpoint.done()


def test():
    _model = model.Model(args, checkpoint, unimodel, rp=rp)
    
    loader_train, loader_test = get_data_loader(t=task, testonly=True)
    t = Trainer(args, loader_train, loader_test, args.data_test, _model, None, checkpoint)
    
    t.testproj(condition=condition)


class Trainer():
    def __init__(self, args, loader_train, loader_test, datasetname, my_model, my_loss, ckp):
        self.args = args
        gpu = torch.cuda.is_available()
        self.device = torch.device('cpu' if (not gpu) else 'cuda')
        self.scale = args.scale
        self.datasetname = datasetname
        self.bestpsnr = 0
        self.bestep = 0
        self.ckp = ckp
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.normalizer = PercentileNormalizer(2, 99.8)
        self.normalizerhr = PercentileNormalizer(2, 99.8)
        self.sepoch = args.resume
        self.epoch = 0
        self.ite = 0
        
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        
        self.error_last = 1e8
        self.dir = os.path.join(rp, 'experiment', self.args.save)
        
        print('Trainer self.dir = ', self.dir)
        
        os.makedirs(self.dir, exist_ok=True)
        if not self.args.test_only:
            self.testsave = self.dir + '/Valid-{}/'.format(self.datasetname)
            os.makedirs(self.testsave, exist_ok=True)
            
            self.filepsnr = open(self.testsave + "TrainPsnr.txt", 'w')
            self.filess = open(self.testsave + "TrainSSIM.txt", 'w')
            self.fileloss = open(self.testsave + "Trainloss.txt", 'w')
    
    def trainUni(self, tsk=1):
        self.loss.step()
        if self.sepoch > 0:
            epoch = self.sepoch
            self.sepoch = 0
        else:
            epoch = self.epoch
        self.epoch = epoch
        
        self.pretrain_epoch_num = 30
        self.filepsnr.write('epoch ' + str(self.epoch) + '\n')
        self.filess.write('epoch ' + str(self.epoch) + '\n')
        self.fileloss.write('epoch ' + str(self.epoch) + '\n')
        
        lr = self.optimizer.get_lr()
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        timer_data, timer_model = utility.timer(), utility.timer()
        if tsk == 1:
            self.model.scale = 2
        
        self.model.train()
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            if (tsk == 1) or (tsk == 2) or (tsk == 3):
                sr = self.model(lr, tsk)
                loss = self.loss(sr, hr)
            elif tsk == 4:
                sr_stg1, sr = self.model(lr, tsk)
                
                if epoch <= self.pretrain_epoch_num:
                    loss = 0.001 * self.loss(sr_stg1, hr) + self.loss(sr, hr)
                else:
                    loss = self.loss(sr, hr)
            elif tsk == 5:
                sr_stg1, sr = self.model(lr, tsk)
                
                if epoch <= self.pretrain_epoch_num:
                    loss = self.loss(sr_stg1, hr)
                else:
                    loss = 0.1 * self.loss(sr_stg1, hr) + self.loss(sr, hr)
            
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(), self.args.gclip)
            self.optimizer.step()
            self.fileloss.write(str(loss.cpu().detach().numpy()) + ',')
            self.fileloss.flush()
            
            timer_model.hold()
            if batch % self.args.print_every == 0:
                sr2dim = np.float32(normalize(np.squeeze(sr[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                hr2dim = np.float32(normalize(np.squeeze(hr[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                print('training patch- PSNR/SSIM = %f/%f' % (psm, ssmm))
            
            if batch % self.args.test_every == 0:
                print('Batch%d/Epoch%d, Loss = ' % (batch, epoch), loss)
                print('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format((batch + 1) * self.args.batch_size,
                                                           len(self.loader_train.dataset),
                                                           self.loss.display_loss(batch),
                                                           timer_model.release(),
                                                           timer_data.release()))
                
                self.loss.end_log(len(self.loader_train))
                self.error_last = self.loss.log[-1, -1]
                self.optimizer.schedule()
                if tsk == 1:
                    psnr, ssim = self.testSR(epoch)
                elif tsk == 2:
                    psnr, ssim = self.test3Ddenoise(epoch)
                elif tsk == 3:
                    psnr, ssim = self.testiso(epoch)
                elif tsk == 4:
                    psnr, ssim = self.testproj(epoch)
                elif tsk == 5:
                    psnr, ssim = self.test2to3(epoch)
                
                self.filepsnr.write(str(psnr) + ',')
                self.filess.write(str(ssim) + ',')
                self.filepsnr.flush()
                self.filess.flush()
                print('Evaluation End -- Batch%d/Epoch%d' % (batch, epoch))
                self.model.train()
                self.loss.step()
                self.loss.start_log()
            timer_data.tic()
        
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
    

    # # -------------------------- Projection --------------------------
    def testproj(self, epoch=0, condition=0):
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/c%d/' % (self.args.resume, condition)
            os.makedirs(self.testsave, exist_ok=True)
        print('save to', self.testsave)
        
        datamin, datamax = self.args.datamin, self.args.datamax
        
        torch.set_grad_enabled(False)
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnrall, ssimall = [], []
        psnralls1, ssimalls1 = [], []
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            if not self.args.test_only and num >= 1:
                break
            num += 1
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            
            # 1.3D norm 2 998
            lrt, hrt = self.prepare(lrt, hrt)
            
            a_stg1, a = self.model(lrt, 4)  # [1, 1, h, w]
            
            sr_stg1 = np.float32(np.squeeze(a_stg1.cpu().detach().numpy()))
            sr = np.float32(np.squeeze(a.cpu().detach().numpy()))
            
            # 3D norm 2 998 tiff save
            srtf = sr
            if self.args.test_only:
                axes_restored = 'YX'
                utility.save_tiff_imagej_compatible(self.testsave + name + '.tif', srtf, axes_restored)
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))  # [1, 1, h, w]
            
            ##  PSNR/SSIM
            hr2dim = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255  # [0, 1]
            sr2dim = np.float32(normalize(np.float32(srtf), datamin, datamax, clip=True)) * 255  # norm_srtf
            sr2dim_stg1 = np.float32(normalize(np.float32(sr_stg1), datamin, datamax, clip=True)) * 255  # norm_srtf
            psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
            psm_stg1, ssmm_stg1 = utility.compute_psnr_and_ssim(sr2dim_stg1, hr2dim)
            print('2D img Norm-%s - PSNR/SSIM = %f/%f / Output of StageI = %f/%f' % (
                name, psm, ssmm, psm_stg1, ssmm_stg1))
            
            psnralls1.append(psm_stg1)
            ssimalls1.append(ssmm_stg1)
            psnrall.append(psm_stg1)
            ssimall.append(ssmm_stg1)
        
        psnrallm = np.mean(np.array(psnrall))
        ssimallm = np.mean(np.array(ssimall))
        psnrallms1 = np.mean(np.array(psnralls1))
        ssimallms1 = np.mean(np.array(ssimalls1))
        
        if self.args.test_only:
            file = open(self.testsave + "Psnrssim.txt_c%d.txt" % condition, 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(psnrall) + '\n SSIM \n' + str(ssimall))
            file.close()
        else:
            if psnrallm > self.bestpsnr:
                self.bestpsnr = psnrallm
                self.bestep = epoch
            self.model.save(self.dir, epoch, is_best=(self.bestep == epoch))
        
        print('+++++++++ condition %d StageI/II ++++++++++++' % condition, psnrallm, ssimallm, psnrallms1, ssimallms1)
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrallm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
        return psnrallm, ssimallm
    
    # # -------------------------- 2D to 3D --------------------------
    def test2to3(self, epoch=0, subtestset='to_predict'):
        if self.args.test_only:
            self.testsave = self.dir + 'results/model_%d/%s/' % (self.args.resume, subtestset)
            os.makedirs(self.testsave, exist_ok=True)
            print('make save path', self.testsave)
        
        datamin, datamax = self.args.datamin, self.args.datamax
        
        torch.set_grad_enabled(False)
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnralls1 = []
        psnrall = []
        ssimalls1 = []
        ssimall = []
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            if not self.args.test_only and num >= 2:
                break
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            if name == '':
                name = 'im%d' % idx_data
            print('image %s ' % (name + '.tif'))
            lrt, hrt = self.prepare(lrt, hrt)  # [1, 121, h//11, w//11]
            
            as1, a = self.model(lrt, 5)
            
            sr = np.float32(a.cpu().detach().numpy())
            srs1 = np.float32(as1.cpu().detach().numpy())
            
            imsave(self.testsave + name + 'norm.tif', np.squeeze(sr))
            print('Save TIF image \' %s \' ' % (self.testsave + name + '.tif'))
            
            sr = (np.clip(np.squeeze(sr), -1, 1) + 1) / 2
            srs1 = (np.clip(np.squeeze(srs1), -1, 1) + 1) / 2
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))  # [61, h, w]
            hr = (np.clip(hr, -1, 1) + 1) / 2
            lr = np.float32(np.squeeze(lrt.cpu().detach().numpy()))  # [121, h//11, w//11]
            lr = (np.clip(lr, -1, 1) + 1) / 2
            
            if self.args.test_only:
                c, h, w = hr.shape
                if h == sr.shape[1]:
                    savecolorim(self.testsave + 'OriIm' + name + '-HR.png', hr[0])
                    wf2d = np.zeros([h, w])
                    d = 0
                    for i in range(11):
                        for j in range(11):
                            wf2d[i: h: 11, j: w: 11] = lr[d, :, :]
                            d += 1
                    savecolorim(self.testsave + 'OriIm' + name + '-LR.png', wf2d)
                    
                    for i in range(0, len(hr), 10):
                        savecolorim(self.testsave + 'OriIm' + name + '-Result%d.png' % i, sr[i])
                        num += 1
                        hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255  # [0, 1]
                        sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                        sr2dims1 = np.float32(normalize(srs1[i], datamin, datamax, clip=True)) * 255
                        psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                        psms1, ssmms1 = utility.compute_psnr_and_ssim(sr2dims1, hr2dim)
                        psnrall.append(psm)
                        ssimall.append(ssmm)
                        psnralls1.append(psms1)
                        ssimalls1.append(ssmms1)
                        print('I%d, 2D img Norm-%s - PSNR/SSIM/MSE = %f/%f StageI  %f/%f' % (
                            i, name, psm, ssmm, psms1, ssmms1))
                else:
                    h, w = h * 11, w * 11
                    print('hr.shape = ', (h, w, 61))
                    wf2d = np.zeros([h, w])
                    d = 0
                    for i in range(11):
                        for j in range(11):
                            wf2d[i: h: 11, j: w: 11] = hr[d, :, :]
                            d += 1
                    savecolorim(self.testsave + 'OriIm' + name + '-LR.png', wf2d)
                    for i in range(0, len(sr), 10):
                        savecolorim(self.testsave + 'OriIm' + name + '.png', sr[i])
                        lr2dim = np.float32(normalize(wf2d, datamin, datamax, clip=True)) * 255  # [0, 1]
                        sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                        sr2dims1 = np.float32(normalize(srs1[i], datamin, datamax, clip=True)) * 255
                        num += 1
                        psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, lr2dim)
                        psms1, ssmms1 = utility.compute_psnr_and_ssim(sr2dims1, lr2dim)
                        psnrall.append(psm)
                        ssimall.append(ssmm)
                        psnralls1.append(psms1)
                        ssimalls1.append(ssmms1)
                        print('I%d, 2D img Norm-%s - PSNR/SSIM/MSE = %f/%f StageI  %f/%f' % (
                            i, name, psm, ssmm, psms1, ssmms1))
            else:
                savecolorim(self.testsave + 'OriIm' + name + '-Result.png', sr[0])
                savecolorim(self.testsave + 'OriIm' + name + '-HR.png', hr[0])
                
                for i in range(0, len(hr), 10):
                    num += 1
                    hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255  # [0, 1]
                    sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                    sr2dims1 = np.float32(normalize(srs1[i], datamin, datamax, clip=True)) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                    psms1, ssmms1 = utility.compute_psnr_and_ssim(sr2dims1, hr2dim)
                    psms1 = np.max([0, np.min([100, psms1])])
                    psnralls1.append(psms1)
                    ssimalls1.append(ssmms1)
                    psnrall.append(psms1)
                    ssimall.append(ssmms1)
                    
                    print('Stage I 2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, psms1, ssmms1))
                    print('Stage II 2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, psm, ssmm))
        psnrmeans1 = np.mean(psnralls1)
        ssmeans1 = np.mean(ssimalls1)
        psnrmeans2 = np.mean(psnrall)
        ssmeans2 = np.mean(ssimall)
        if self.args.test_only:
            psnrmean = psnrmeans2
            ssmean = ssmeans2
        else:
            if epoch <= self.pretrain_epoch_num:
                psnrmean = psnrmeans1
                ssmean = ssmeans1
            else:
                psnrmean = psnrmeans2
                ssmean = ssmeans2
        
        if psnrmean > self.bestpsnr:
            self.bestpsnr = psnrmean
            self.bestep = epoch
        if not self.args.test_only:
            self.model.save(self.dir, epoch, is_best=(self.bestep == epoch))
        print('+++++++++ StageI/II ++++++++++++', psnrmeans1, ssmeans1, psnrmeans2, ssmeans2)
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrmean, self.bestpsnr, self.bestep)
        
        torch.set_grad_enabled(True)
        return psnrmean, ssmean
    
    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        
        return [_prepare(a) for a in args]
    
    def terminate(self):
        if self.args.test_only:
            return False
        else:
            self.epoch = self.epoch + 1
            if self.epoch > self.args.epochs:
                self.filepsnr.close()
                self.filess.close()
                self.fileloss.close()
            return self.epoch <= self.args.epochs


if __name__ == '__main__':
    task = 1
    test_only = False  # True  #
    pre_train = './experiment/Uni-SwinIR/model_best.pt'
    
    test_every = 1000
    srdatapath = '/home/user2/dataset/microscope/CSB/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/'
    denoisedatapath = '/home/user2/dataset/microscope/CSB/DataSet/'
    isodatapath = '/home/user2/dataset/microscope/CSB/DataSet/Isotropic/'
    prodatapath = '/home/user2/dataset/microscope/CSB/DataSet/'
    voldatapath = '/home/user2/dataset/microscope/VCD/vcdnet/'
    
    if task == 1:  # SR
        testset = 'ER'  # 'CCPs'  # 'Microtubules'  # 'F-actin'  #
        batch = 1
        patch = 128
    elif task == 2:  # denoise
        condition = 1
        patch = 64
        batch = 32
        testset = 'Denoising_Planaria'  # testset = 'Denoising_Tribolium'
    elif task == 3:  # isotropic
        testset = 'Isotropic_Liver'
        batch = 32
        patch = 128
    elif task == 4:  # projection
        condition = 2
        batch = 4
        patch = 64
        testset = 'Projection_Flywing'
    elif task == 5:  # 2D to 3D
        batch = 4
        patch = 64
        testset = 'VCD'
        subtestset = 'to_predict'
    
    savename = 'Uni-SwinIR%s/' % testset
    
    args = options()
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    assert checkpoint.ok
    
    unimodel = model.UniModel(args, tsk=task)
    if not test_only:
        train()
    else:
        test()
