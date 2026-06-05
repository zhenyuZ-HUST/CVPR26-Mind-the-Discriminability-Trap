
import random
import argparse  
import numpy as np 
import torch

from lora_base import run_lora_base
from lora_mind import run_lora_mind
    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--shots', default=8, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    #parser.add_argument('--rn', default=False, type=bool)
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='vision')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA')  # ['q', 'k', 'v']
    parser.add_argument('--r', default=16, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=8, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    #################################################################################
    parser.add_argument('--dataset', type=str, default='EuroSAT') #CropDisease, ChestX, EuroSAT, ISIC, MiniImageNet
    parser.add_argument('--image_size', default=224, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=3)
    parser.add_argument('--template', type=str, default='a photo of a {}.')


    args = parser.parse_args()

    return args
    

        
