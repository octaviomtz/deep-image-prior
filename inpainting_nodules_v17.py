'''
Performs inpaiting on lung nodules
-Based on the notebook: "inpainting nodules of all patients v17 (from v15) - 2D conv on 3D vol.ipynb"
-Uses 2D convolutions
-This was modified (~line 189)
    image_last = images_generated_all[0] * block_lungs
    image_last = images_generated_all[-1] * block_lungs
-We use num_channels_skip = [128]*5, 
'''
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.optim
import imageio
from copy import copy
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from tqdm import tqdm_notebook
from skimage import measure, morphology
from itertools import groupby, count
import matplotlib.patches as patches
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from torch.autograd import Variable

from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from utils.inpainting_utils import *

from inpainting_nodules_functions import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('skip_idx', type=int, help='skip indices already processed')
parser.add_argument('cuda_gpu', type=int, help='gpu to use')
args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
PLOT = True
imsize = -1
dim_div_by = 64

path_data = f'/data/OMM/Datasets/LIDC_other_formats/LIDC_preprocessed_3D v4 - inpaint before preprocess/'
path_data_old = '/data/OMM/Datasets/LIDC_other_formats/LIDC_preprocessed_3D v2/'
path_img_dest = '/data/OMM/project results/Feb 5 19 - Deep image prior/dip results all 17/'

torch.cuda.set_device(args.cuda_gpu)
torch.cuda.empty_cache()

dtype = torch.cuda.FloatTensor
from torch.autograd import Variable

# NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet
pad = 'zero' # 'zero' OMM it was reflection
OPT_OVER = 'net'
OPTIMIZER = 'adam'
INPUT = 'noise'
input_depth = 96*2 # 
#LR = 0.000001 
num_iter = 20001 # 10001
param_noise = False
show_every = 500
#figsize = 5
reg_noise_std = 0.00

def closure():
    global i
    images_all = []
    
    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50
    
    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
    out = net(net_input)
    #print(np.shape(net_input), np.shape(out), np.shape(mask_var), np.shape(img_var))
    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()
        
    print ('Iteration %05d    Loss %.12f' % (i, total_loss.item()), '\r', end='')
    #if  PLOT and i % show_every == 0:
    if  PLOT:
        out_np = torch_to_np(out)
        image_to_save = out_np
        #if np.shape(out_np)[0] == 1:
            #image_to_save = out_np[0]
        #plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1) # DEL original fun
        #plot_for_gif(image_to_save, num_iter, i) # DEL save images to make gif
        images_all.append(image_to_save)
        
    i += 1    
#     if  PLOT and i % show_every == 0: image_generated = image_to_save
#     else: image_generated = []
    
    return total_loss, images_all

ids = os.listdir(path_data)
ids = np.sort(ids)
for idx_name, name in enumerate(ids):
    torch.cuda.empty_cache()
    if idx_name <= args.skip_idx:continue
    print(f'{name}, ({idx_name})')
    vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small = read_slices3D_v2(path_data, name)
    slice_middle = np.shape(vol_small)[0] // 2
    xmed_1, ymed_1, xmed_2, ymed_2 = erode_and_split_mask(mask_lungs_small,slice_middle)
    coord_min_side1, coord_max_side1, coord_min_side2, coord_max_side2 = nodule_right_or_left_lung(mask_maxvol_small, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2)
    try:
        c_zmin2, c_zmax2, c_xmin2, c_xmax2, c_ymin2, c_ymax2 = box_coords_contain_masks_right_size_search(coord_max_side2, coord_min_side2, 2, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small)
        c_zmin1, c_zmax1, c_xmin1, c_xmax1, c_ymin1, c_ymax1 = box_coords_contain_masks_right_size_search(coord_max_side1, coord_min_side1, 1,  slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small)
    except ValueError:
        # Example :LIDC-IDRI-0218, (211)
        # box_coords_contain_masks_right_size_search (line 407)
        # xmin = int(xmed_X-(dist2//2)+jj);
        # ValueError: cannot convert float NaN to integer
        print('skipping scan, ValueError ')
        continue
    # Block1 and Block2: lungs, ndl mask, lungs mask, ndl&lungs mask
    block1, block1_mask, block1_mask_maxvol_and_lungs, block1_mask_lungs = get_four_blocks(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, c_zmin1, c_zmax1, c_xmin1, c_xmax1, c_ymin1, c_ymax1)
    block2, block2_mask, block2_mask_maxvol_and_lungs, block2_mask_lungs = get_four_blocks(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, c_zmin2, c_zmax2, c_xmin2, c_xmax2, c_ymin2, c_ymax2)
    # Normalization is applied using the min and max of all images
    block1 = (block1 - (-1018.0))/(1171.0-(-1018.0)) 
    block1 = np.clip(block1,0,1)
    block2 = (block2 - (-1018.0))/(1171.0-(-1018.0)) 
    block2 = np.clip(block2,0,1)
    # Apply lungs' mask
    block1 = block1*block1_mask_lungs
    block2 = block2*block2_mask_lungs
    # Get those blocks where there is a nodule in
    blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names, slice1, slice2 =  get_block_if_ndl(block1, block2, block1_mask, block2_mask, block1_mask_maxvol_and_lungs, block2_mask_maxvol_and_lungs, block1_mask_lungs, block2_mask_lungs)
    # delete variables 
    del block1, block2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small
    for (block, block_mask, block_maxvol_and_lungs, block_lungs, block_name) in zip(blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names): 
        torch.cuda.empty_cache()
        print(block_name)
        # Here we dont add batch channels
        img_np = block
        img_mask_np = block_maxvol_and_lungs

        # LR FOUND
        LR = np.load(f'{path_img_dest}learning rates/{name}_{block_name}.npy')
        
        # INPAINTING
        restart_i = 0
        restart = True

        while restart == True:
            start = time.time()
            print(f'training initialization {restart_i} with LR = {LR:.12f}')
            restart_i += 1

            #lungs_slice, mask_slice, nodule, outside_lungs = read_slices(new_name)
            #img_np, img_mask_np, outside_lungs = make_images_right_size(lungs_slice, mask_slice, nodule, outside_lungs)

            # Loss
            mse = torch.nn.MSELoss().type(dtype)
            img_var = np_to_torch(img_np).type(dtype)
            mask_var = np_to_torch(img_mask_np).type(dtype)

            net = skip(input_depth, img_np.shape[0], 
                    num_channels_down = [128]*5,
                    num_channels_up   = [128]*5, 
                    num_channels_skip = [128]*5, 
                    upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3, 
                    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
            net = net.type(dtype)        
            net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

            #path_trained_model = f'{path_img_dest}models/v6_Unet_init_sample_{idx}.pt'
            #torch.save(net.state_dict(), path_trained_model)

            #mse_error = []
            i = 0
            net_input_saved = net_input.detach().clone()
            noise = net_input.detach().clone()
            p = get_params(OPT_OVER, net, net_input)
            mse_error, images_generated_all, best_iter, restart = optimize_v17(OPTIMIZER, p, closure, LR, num_iter, show_every, path_img_dest, restart, annealing=True, lr_finder_flag=False)
            mse_error = np.squeeze(mse_error)
            mse_error = [i.detach().cpu().numpy() for i in mse_error]
        #     mse_error_all.append(mse_error)
        #     mse_error_last = mse_error[-1].detach().cpu().numpy()

            if restart_i % 10 == 0: # reduce lr if the network is not learning with the initializations
                LR /= 1.2
            if restart_i == 30: # if the network cannot be trained continue (might not act on for loop!!)
                continue
        print('')
        #print(np.shape(images_generated_all))
        print('')
        image_last = images_generated_all[-1] * block_lungs
        image_orig = img_np[0] * block_lungs
        best_iter = f'{best_iter:4d}'

        stop = time.time()
        np.save(f'{path_img_dest}arrays/last/{name}_{block_name}.npy',image_last)
        np.save(f'{path_img_dest}arrays/orig/{name}_{block_name}.npy',img_np)
        np.savez_compressed(f'{path_img_dest}arrays/masks/{name}_{block_name}',block_maxvol_and_lungs)
        np.savez_compressed(f'{path_img_dest}arrays/masks nodules/{name}_{block_name}',block_mask)
        np.savez_compressed(f'{path_img_dest}arrays/masks lungs/{name}_{block_name}',block_lungs)
        np.save(f'{path_img_dest}mse error curves inpainting/{name}_{block_name}.npy',mse_error)
        np.save(f'{path_img_dest}inpainting times/{name}_{block_name}.npy',stop-start)
        torch.save({'epoch': len(mse_error), 'model_state_dict': net.state_dict(),
            'LR': LR,'loss': mse_error, 'net_input_saved': net_input_saved}, 
            f'{path_img_dest}models/{name}_{block_name}.pt')
        del net, images_generated_all