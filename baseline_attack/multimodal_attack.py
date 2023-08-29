import torch
from transformers import BatchEncoding
import torch.nn.functional as F

def equal_normalize(x):
    return x

class MultiModalAttacker():
    def __init__(self, net, image_attacker, text_attacker, *args, **kwargs):
        self.net = net  # net本身有一个tokenizer
        self.image_attacker = image_attacker
        self.text_attacker = text_attacker
        # self.ref_tokenizer = ref_tokenizer
        # self.cls = cls
        
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        
        if hasattr(text_attacker, 'sample_batch_size'):
            self.sample_batch_size = text_attacker.sample_batch_size
        if hasattr(image_attacker, 'preprocess'):
            self.image_normalize = image_attacker.preprocess
        else:
            self.image_normalize = equal_normalize

        self.repeat = 1
        if hasattr(image_attacker, 'repeat'):
            self.repeat = image_attacker.repeat
    
    
    def run_caption(self, images, texts, adv, num_iters=10, k=10, max_length=40, alpha=3.0, scales=None):
        """using lm loss to generate adversarial images and texts

        Args:
            images (Tensor): test image batch without normalization; shape=[bs,3,384,384]
            texts (List): test text batch [with the prompt: 'a picture of ']; natural language; len=bs
            adv (Int): 3=Clean Eval; 4=Sep-Attack; 5=Co-Attack
            num_iters (int, optional): iteration number when attacking images. Defaults to 10.
            k (int, optional): candidate number when attacking texts. Defaults to 10.
            max_length (int, optional): max length for texts. Defaults to 40.
            alpha (float, optional): hyperparameter of Co-Attack of balancing two loss items. Defaults to 3.0.
        """
        device = images.device
        # 此时loss的计算不需要embedding, 只需要输入的images和texts
        # Sep-Attack: do not need adv texts, therefore no requirement for prompts
        if adv == 3:
            # print('sep attack image')
            image_attack = self.image_attacker.attack(images, num_iters)  # attack的时候image没有normalize [0,1]之间
            for i in range(num_iters):
                image_diversity = next(image_attack)  # 这不是生成的对抗样本
                # image_diversity是normalize之后的数据, 所以进模型的时候不需要再次normalize
                loss = self.net.get_lm_loss(image_diversity, texts, loss_reduction='mean')
                loss.backward()
            images_adv = next(image_attack)  # image+delta 此时是没有normalize的
            # torch.save(images_adv, '/home/zhengf/ld/BLIP_sammy/output/images_adv.pt')
            # torch.save(images, '/home/zhengf/ld/BLIP_sammy/output/images_ori.pt')
            ## 1. check 扰动的大小 是不是在2/255之间 yes  torch.min(images_adv-images) torch.max(images_adv-images)
        
        elif adv == 4:    
            # image attacker initialization
            image_attack = self.image_attacker.attack(images, num_iters)
            with torch.no_grad():
                # start with text-modal perturbation here the texts are without prompt
                texts_adv = self.text_attacker.attack(self.net, self.image_normalize(images), texts, k, max_length=max_length)  
            for _ in range(num_iters):
                image_diversity = next(image_attack)
                texts_adv_prompt_for_loss = [self.net.prompt + t for t in texts_adv]
                loss_clean_text = self.net.get_lm_loss(image_diversity, texts, loss_reduction='mean')
                loss_adv_text = self.net.get_lm_loss(image_diversity, texts_adv_prompt_for_loss, loss_reduction='mean')
                loss = loss_adv_text + alpha * loss_clean_text
                loss.backward()
            images_adv = next(image_attack)
    
        else:
            images_adv = images
            
        
        # text: need adv texts for loss computation; therefore need to consider prompts
        if adv == 3 or adv == 4:
            with torch.no_grad():
                texts_adv = self.text_attacker.attack(self.net, self.image_normalize(images), texts, k, max_length=max_length)  

        return images_adv, texts_adv