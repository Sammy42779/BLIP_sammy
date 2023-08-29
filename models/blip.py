'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = '/home/zhengf/ld/BLIP/configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)    
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        
    def forward(self, image, caption):  # train stage: image and ground-truth
        
        image_embeds = self.visual_encoder(image)   # shape: [32, 577, 768]
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)  # padding=longest: 批次中最长序列长度, 在此基础上不能再超过40
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id  # '[ D E C ]'  把CLS token换成DEC token
        # 这部分的输出应该被解码器忽略, 即在计算损失时, 这部分的预测输出不应该对损失有所贡献
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100) # PAD token的id是0, 把PAD标记替换为-100        
        decoder_targets[:,:self.prompt_length] = -100  # 存储解码器的目标输出, 这是因为此时的caption前缀都加上了prompt='a picture of ', len=4
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss  # decoder_output.logits=prediction_scores
        
        return loss_lm
    
    def get_lm_loss_without_prompt(self, image, caption, loss_reduction='mean'):
        """get lm loss during test stage; therefore no prompts here

        Args:
            images (Tensor): test image batch [with] normalization; shape=[bs,3,384,384]
            caption (List): test text batch; natural language; len=bs
            image_normalize (function): normalization function for images
            loss_reduction (str, optional): loss reduction method. Defaults to 'mean'. 'mean': only one average loss; 'none': loss for each sample
        """
        # image = image_normalize(image)
        image_embeds = self.visual_encoder(image)   # shape: [32, 577, 768]
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        text.input_ids[:,0] = self.tokenizer.bos_token_id  # '[ D E C ]' 
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)    
        # decoder_targets[:,:self.prompt_length] = -100  No prompts
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,  
                                           reduction = loss_reduction, 
                                          )   
        loss_lm = decoder_output.loss 
        return loss_lm
    
    def get_lm_loss(self, image, caption, loss_reduction='mean'):
        """get lm loss during test stage; therefore no prompts here

        Args:
            images (Tensor): test image batch [with] normalization; shape=[bs,3,384,384]
            caption (List): test text batch [with the prompt: 'a picture of ']; natural language; len=bs
            image_normalize (function): normalization function for images
            loss_reduction (str, optional): loss reduction method. Defaults to 'mean'. 'mean': only one average loss; 'none': loss for each sample
        """
        # image = image_normalize(image)
        image_embeds = self.visual_encoder(image)   # shape: [32, 577, 768]
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        text.input_ids[:,0] = self.tokenizer.bos_token_id  # '[ D E C ]' 
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)    
        decoder_targets[:,:self.prompt_length] = -100  # No prompts
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,  
                                           reduction = loss_reduction, 
                                          )   
        loss_lm = decoder_output.loss 
        return loss_lm

    
    def get_multimodal_output(self, image_embeds, image_atts, text_input):  # test stage: image and ground-truth
        # image_embeds = self.visual_encoder(image)   # shape: [32, 577, 768]
        # image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)
        
        # text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(device)  # padding=longest: 批次中最长序列长度, 在此基础上不能再超过40
        
        text_input.input_ids[:,0] = self.tokenizer.bos_token_id  # '[ D E C ]'  把CLS token换成DEC token
        # 这部分的输出应该被解码器忽略, 即在计算损失时, 这部分的预测输出不应该对损失有所贡献
        decoder_targets = text_input.input_ids.masked_fill(text_input.input_ids == self.tokenizer.pad_token_id, -100) # PAD token的id是0, 把PAD标记替换为-100        
        # 此时没有prompt
        # decoder_targets[:,:self.prompt_length] = -100  # 存储解码器的目标输出, 这是因为此时的caption前缀都加上了prompt='a picture of ', len=4
     
        decoder_output = self.text_decoder(text_input.input_ids, 
                                           attention_mask = text_input.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           output_hidden_states=True,
                                           return_dict = True,   
                                          )   
        return decoder_output
        
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        # print(f'%%%%%%%%%%%%%%%%%%%%%%sample:{sample}%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}  # image_embeds是图像的embedding, encoder_attention_mask 是图像的attention mask, 用来指示模型哪些位置的token是有效的，哪些位置的token是无效的(比如填充的部分)
        
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)  # 是输入到模型中的token序列
        input_ids[:,0] = self.tokenizer.bos_token_id  # bos_token_id是 "Beginning of Sentence" token 的 ID; 每个序列的第一个token被替换为 bos_token_id
        input_ids = input_ids[:, :-1]  # 最后一个token被去掉; 可能是为了保持序列的长度，或者是为了去掉原来序列中的结束token

        if sample:
            #nucleus sampling Top-p 采样 随机采样; 随机的文本生成
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search 树搜索
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions  # 自然语言形式
    

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('/home/zhengf/.cache/bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
