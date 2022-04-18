import torch
import os
from torch.utils.data import DataLoader
import cv2

from tools.logger import Logger
from tools.utils import mkdir


class Solver(object):
    def __init__(self, opt, trn_dset, tst_dset, model):
        self.trn_loader = DataLoader(trn_dset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
        self.tst_loader = DataLoader(tst_dset, batch_size=1, shuffle=False, num_workers=opt.workers, drop_last=True)
        self.model = model
        self.opt = opt
        self.iter_n = 0
        self.start_epoch = 0
        
        self.best_psnr = 0
        self.best_ssim = 0

        if self.opt.train:
            self.logger = Logger(os.path.join(opt.save_path, 'log'))
            mkdir(self.logger.path())
        if not self.opt.train:
            self.save_dir = os.path.join(opt.save_path, opt.save_tst_out)
            mkdir(self.save_dir)
            self.log_dir = open(os.path.join(self.save_dir, 'log%s.txt'%(self.opt.save_fix)), 'a')

    def restore_break_info(self, break_info):
        self.start_epoch = break_info['epoch'] + 1
        self.iter_n = break_info['iter_n']
        self.best_psnr = break_info['best_psnr']
        self.best_ssim = break_info['best_ssim']
        for _ in range(self.start_epoch):
            self.model.adjust_lr()
        print('Restore the break information.')

    def train_a_epoch(self, epoch):
        print('--- EPOCH %d ---'%(epoch))
        # self.model.adjust_lr()
        
        for batch, data in enumerate(self.trn_loader):
            data['epoch'] = epoch
            losses = self.model.train_a_step(data)
            self.model.adjust_lr()
            self.iter_n += 1
            print('%s cuda%d epoch %d(%d)/batch %d/iter %d'%(self.opt.save_path.split('/')[-1], self.opt.cuda, epoch, self.opt.epochs, batch, self.iter_n), end=': ')
            for key in losses:
                print('%s=%.4f'%(key, losses[key]), end=' ')
                self.logger.add_scalar(losses[key], self.iter_n, key)
            print('; best psnr=%.4f ssim=%.4f'%(self.best_psnr, self.best_ssim))
            if (self.iter_n % self.opt.test_interval == 0):
                self.test(self.iter_n, epoch)
            if ('save_iter' in self.opt) and (self.iter_n % self.opt.save_iter == 0):
                self.save_model('iter_%d.pth'%(self.iter_n))
        
        if epoch > 0 and epoch % self.opt.test_per_epoch == 0:
            self.test(self.iter_n, epoch)
        if epoch > 0 and epoch % self.opt.save_interval == 0:
            self.save_model('%d.pth'%(epoch))

    def save_model(self, name, break_info=None):
        if break_info is None:
            break_info = self.break_info
        self.model.save_model(os.path.join(self.opt.save_path, name), break_info)
        print('Save model: %s'%(name))

    def test(self, n=0, epoch=0):
        save = bool(self.opt.save_img)
        self.model.eval_()
        print('save:', save)
        print('--------------- TEST ---------------')
        print('n %d'%(n))
        if not self.opt.train:
            self.log_dir.write('--------------- TEST ---------------\n')
            self.log_dir.write('%s\n'%(self.opt.resume))
        with torch.no_grad():
            psnrs = 0
            ssims = 0
            LPIPS_singles = 0
            cnt = 0
            for batch, data in (enumerate(self.tst_loader)):
                pred, metric = self.model.test_a_step(data)
                psnr, ssim, LPIPS_single = metric
                psnrs += psnr
                ssims += ssim
                LPIPS_singles += LPIPS_single
                cnt += 1

                img_name = data['hr_path'][0].split('/')[-1]
                img_name = img_name[:-4] + self.opt.save_fix + '.png'
                                
                print(batch, img_name, psnr, ssim, LPIPS_single)
                
                if not self.opt.train:
                    hr_file = data['hr_path'][0].split('/')[-1]
                    self.log_dir.write('%d %s ycbcr psnr=%.4f, ssim=%.4f, LPIPS_single=%.4f\n'%(batch, hr_file, psnr, ssim, LPIPS_single))
                    if save:
                        img = pred[0].permute(1, 2, 0)
                        img = torch.clamp(img, 0, 1)
                        img = (img.numpy() * 255.)
                        cv2.imwrite(os.path.join(self.save_dir, img_name), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            avg_psnr = psnrs / cnt
            avg_ssim = ssims / cnt
            avg_LPIPS_single = LPIPS_singles / cnt
            print('CNT:', cnt)

            break_info = {
                'epoch': epoch,
                'iter_n': self.iter_n,
                'best_psnr': max(self.best_psnr, avg_psnr),
                'best_ssim': max(self.best_ssim, avg_ssim),
            }

            print('TEST: n %d: ycbcr psnr=%.4f, ssim=%.4f LPIPS_single=%.4f'%(n, avg_psnr, avg_ssim, avg_LPIPS_single))
            if not self.opt.train:
                self.log_dir.write('MEAN YCBCR PSNR: %.4f\n'%(avg_psnr))
                self.log_dir.write('MEAN SSIM: %.4f\n'%(avg_ssim))
                self.log_dir.write('MEAN LPIPS_single: %.4f\n\n'%(avg_LPIPS_single))
                self.log_dir.close()
            else:
                print('add scalar')
                self.logger.add_scalar(torch.tensor(avg_psnr), n, 'psnr')
                self.logger.add_scalar(torch.tensor(avg_ssim), n, 'ssim')
                self.logger.add_scalar(torch.tensor(avg_LPIPS_single), n, 'lpips')
                val = avg_psnr
                if (val > self.best_psnr):   
                    self.best_psnr = val
                    self.model.save_model(os.path.join(self.opt.save_path, 'best.pth'), break_info)    
                    print('Save to best model: ycbcr psnr %.4f -> %.4f'%(self.best_psnr, val))
                if (avg_ssim > self.best_ssim):
                    self.best_ssim = avg_ssim
                self.model.save_model(os.path.join(self.opt.save_path, 'current.pth'), break_info)
                self.log_dir = open(os.path.join(self.opt.save_path, 'train_log.txt'), 'a')
                self.log_dir.write('%d: ycbcr_psnr=%.4f (best=%.4f); ssim=%.4f Lpips=%.4f\n'%(n, avg_psnr, self.best_psnr, avg_ssim, avg_LPIPS_single)) 
                self.log_dir.close()

            self.break_info = break_info
            return avg_psnr

    def tensor_to_image(self, pred):
        img = pred[0].permute(1, 2, 0)
        img = torch.clamp(img, 0, 1)
        img = (img.cpu().numpy() * 255.)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def train(self):
        for epoch in range(self.start_epoch, self.opt.epochs):
            self.train_a_epoch(epoch)
        self.logger.close()
        self.model.save_model(os.path.join(self.opt.save_path, 'final.pth'), self.break_info)
