'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip import blip_decoder
import utils
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval

from models.tokenization_bert import BertTokenizer

from torchvision import transforms
from transformers import BertForMaskedLM
from baseline_attack import *

# @torch.no_grad() Gradient information is necessary if we want to perform backpropagation on the loss.
def evaluate(source_model, target_model, ref_model, data_loader, ref_tokenizer, device, config):
    # evaluate
    source_model.eval() 
    target_model.eval()
    ref_model.eval()
    
    # generate adversarial examples from source model
    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image_attacker = ImageAttacker(epsilon=args.epsilon/255., preprocess=images_normalize, bounding=(0, 1))
    text_attacker = BertAttackFusion_caption(ref_model, ref_tokenizer)
    multi_attacker = MultiModalAttacker(source_model, image_attacker, text_attacker)
    sga_img_attacker = SGAImageAttacker(images_normalize, eps=args.epsilon/255, steps=10, step_size=0.5/255)
    sga_txt_attacker = SGATextAttacker(ref_model, ref_tokenizer)
    sga_attacker = SGAttacker(source_model, sga_img_attacker, sga_txt_attacker)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    # adv_images_without_norm = []
    # adv_images_names = []
    # adv_texts = []

    result = []
    if args.adv == 6:
        if args.scales is not None:
            scales = [float(itm) for itm in args.scales.split(',')]
            print(scales)
        else:
            scales = None
        for (image, texts_group, image_id, text_ids_groups, image_name) in metric_logger.log_every(data_loader, print_freq, header): 
            texts_ids = []
            txt2img = []
            caption = []  # 此时的texts是带有prompt的;用于计算caption loss
            for i in range(len(texts_group)):
                caption += texts_group[i]
                texts_ids += text_ids_groups[i]
                txt2img += [i]*len(text_ids_groups[i])
            image = image.to(device)     
            image, adv_texts = sga_attacker.attack_caption(image, caption, txt2img, device=device,
                                                max_lemgth=30, scales=scales) 
            # adv_images_without_norm.append(image)
            # adv_texts.append(adv_texts)
            # adv_images_names.append(image_name)
            image = images_normalize(image)
            
            # evaluate the target model on adversarial examples
            captions = target_model.generate(image, sample=False, num_beams=target_config['num_beams'], max_length=target_config['max_length'], 
                                    min_length=config['min_length'])
            
            for caption, img_id in zip(captions, image_id):
                result.append({"image_id": img_id, "caption": caption})  # 这里要保存原始image-id, 而不是index
                
            torch.cuda.empty_cache()
        # adv_images_without_norm_all = torch.cat(adv_images_without_norm, dim=0)
        # utils.save_adv_datasets(adv_images_without_norm_all, adv_texts, adv_images_names, args.source_model, args.adv)
    
        return result
    else:
        for image, image_id, caption, image_name in metric_logger.log_every(data_loader, print_freq, header): 
            image = image.to(device)     
            
            if args.adv != 0:
                image, adv_texts = multi_attacker.run_caption(image, caption, adv=args.adv, num_iters=args.num_iters, alpha=args.alpha)  
            # adv_images_without_norm.append(image)
            # adv_texts.append(adv_texts)
            # adv_images_names.append(image_name)
            image = images_normalize(image)
            
            # evaluate the target model on adversarial examples
            captions = target_model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                    min_length=config['min_length'])
            
            for caption, img_id in zip(captions, image_id):
                result.append({"image_id": img_id.item(), "caption": caption})
                
            torch.cuda.empty_cache()
        # adv_images_without_norm_all = torch.cat(adv_images_without_norm, dim=0)
        # utils.save_adv_datasets(adv_images_without_norm_all, adv_texts, adv_images_names, args.source_model, args.adv)
        
        return result


def main(args, source_config, target_config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    # for both source model and target model, the test data is the same
    print("Creating captioning dataset")
    if args.adv == 6:
        test_dataset = create_dataset('adv_caption_coco_sga', target_config)  
    else:
        test_dataset = create_dataset('adv_caption_coco', target_config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([test_dataset], [False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
        
    if args.adv == 6:
        test_loader = create_loader([test_dataset],samplers,
                                 batch_size=[args.batch_size],num_workers=[4],
                                 is_trains=[False], collate_fns=[test_dataset.collate_fn])[0] 
    else:
        test_loader = create_loader([test_dataset],samplers,
                                    batch_size=[args.batch_size],num_workers=[4],
                                    is_trains=[False], collate_fns=[None])[0]         

    #### Model #### 
    print("Creating model")
    source_model = blip_decoder(pretrained=source_config['pretrained'], image_size=source_config['image_size'], vit=source_config['vit'], vit_grad_ckpt=source_config['vit_grad_ckpt'], vit_ckpt_layer=source_config['vit_ckpt_layer'], prompt=source_config['prompt'])
    target_model = blip_decoder(pretrained=target_config['pretrained'], image_size=target_config['image_size'], vit=target_config['vit'], vit_grad_ckpt=target_config['vit_grad_ckpt'], vit_ckpt_layer=target_config['vit_ckpt_layer'], prompt=target_config['prompt'])

    ref_tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)
    
    source_model = source_model.to(device)
    target_model = target_model.to(device)
    ref_model = ref_model.to(device)
    
    source_model_without_ddp = source_model
    target_model_without_ddp = target_model
    if args.distributed:
        source_model = torch.nn.parallel.DistributedDataParallel(source_model, device_ids=[args.gpu])
        source_model_without_ddp = source_model.module   
        target_model = torch.nn.parallel.DistributedDataParallel(target_model, device_ids=[args.gpu])
        target_model_without_ddp = target_model.module 

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, target_config['max_epoch']):

        test_result = evaluate(source_model_without_ddp, target_model_without_ddp, ref_model, test_loader, ref_tokenizer, device, target_config)  
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')  
        # test_result_file = '/home/zhengf/ld/BLIP_sammy/output/caption_coco/capfilt_L/adv6/result/test_epoch0.json'

        if utils.is_main_process():   
            coco_test = coco_caption_eval(target_config['coco_gt_root'],test_result_file,'test')
            
            if args.evaluate:            
                log_stats = {
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
                    
        if args.evaluate: 
            break
        dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    
    utils.set_seed(42)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '14'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print('train_caption_attack')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    parser.add_argument('--batch_size', default=2, type=int, help='Clean-Eval:64, Sep-Attack:32, Co-Attack:16')  
    parser.add_argument('--text_encoder', default='/home/zhengf/.cache/bert-base-uncased')
    parser.add_argument('--adv', default=3, type=int, help='0=Clean-Eval, 3=Sep-Attack, 4=Co-Attack')  
    parser.add_argument('--epsilon', default=2, type=int)
    parser.add_argument('--num_iters', default=10, type=int)  
    parser.add_argument('--alpha', default=3.0, type=float)
    parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
    
    # same model with different base architectures
    parser.add_argument('--source_model', default='capfilt_L', type=str)  
    parser.add_argument('--target_model', default='vit_L', type=str)
    
    args = parser.parse_args()
    
    source_model_config = f'/home/zhengf/ld/BLIP_sammy/configs/caption_coco_{args.source_model}.yaml'
    target_model_config = f'/home/zhengf/ld/BLIP_sammy/configs/caption_coco_{args.target_model}.yaml'

    source_config = yaml.load(open(source_model_config, 'r'), Loader=yaml.Loader)
    target_config = yaml.load(open(target_model_config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(source_config, open(os.path.join(args.output_dir, 'source_config.yaml'), 'w'))    
    yaml.dump(target_config, open(os.path.join(args.output_dir, 'target_config.yaml'), 'w'))    
    
    main(args, source_config, target_config)