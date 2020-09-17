import os
import sys
import time
import pickle
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

# pix2pix modules
pix2pix_dir = './pix2pixHD'
sys.path.append(pix2pix_dir)
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html

import train_opt as train_opt
import test_opt as test_opt


def train_pix2pix(opt=train_opt):
    ''' Train pix2pix model '''

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    print(f'# training images = {len(data_loader)}')

    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    model = create_model(opt)
    model = model.cuda()

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                      Variable(data['image']), Variable(data['feat']), infer=save_fake)

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            ############### Backward Pass ####################
            # update generator weights
            model.optimizer_G.zero_grad()
            loss_G.backward()
            model.optimizer_G.step()

            # update discriminator weights
            model.optimizer_D.zero_grad()
            loss_D.backward()
            model.optimizer_D.step()

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print(f'saving the latest model (epoch {epoch}, total_steps {total_steps})')
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}, iters {total_steps}')
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()

    torch.cuda.empty_cache()

def transfer(opt=test_opt):
    ''' Transfer source video to target video '''
    
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    print(f'# testing images = {len(data_loader)}')
    
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}'))
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.which_epoch}')

    model = create_model(opt)
    model = model.cuda()

    for data in tqdm(dataset):
        minibatch = 1 
        generated = model.inference(data['label'], data['inst'])
            
        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                               ('synthesized_image', util.tensor2im(generated.data[0]))])
        img_path = data['path']
        visualizer.save_images(webpage, visuals, img_path)
    
    webpage.save()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    train_pix2pix()
    transfer()
