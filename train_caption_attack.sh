
### 

# ### BLIP CapFilt-L
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP/configs/caption_coco_capfilt_L.yaml --evaluate

# ## Sep-Attack
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP/configs/caption_coco_capfilt_L.yaml --evaluate


# ### BLIP ViT-L
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP/configs/caption_coco_vit_L.yaml --evaluate 


# CUDA_VISIBLE_DEVICES=0,8,9,10,11,12,13,14,15 python -m torch.distributed.run --nproc_per_node=9 train_caption_attack.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml --evaluate --adv 3 --batch_size 8 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/capfilt_L/adv3

# CUDA_VISIBLE_DEVICES=0,8,9,10,11,12,13,14,15 python -m torch.distributed.run --nproc_per_node=9 train_caption_attack.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml --evaluate --adv 4 --batch_size 4 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/capfilt_L/adv4


# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml --evaluate --adv 6 --batch_size 1 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/capfilt_L/adv6


# 验证source model + target model结果


# # ViT2capfilt_L adv3
# SOURCE_MODEL=vit_L
# TARGET_MODEL=capfilt_L
# ADV=3
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml --evaluate --adv ${ADV} --batch_size 16 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/${SOURCE_MODEL}2${TARGET_MODEL}/adv${ADV} --source_model ${SOURCE_MODEL} --target_model ${TARGET_MODEL}
# # capfilt_L2ViT adv3
# SOURCE_MODEL=capfilt_L
# TARGET_MODEL=vit_L
# ADV=3
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml --evaluate --adv ${ADV} --batch_size 16 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/${SOURCE_MODEL}2${TARGET_MODEL}/adv${ADV} --source_model ${SOURCE_MODEL} --target_model ${TARGET_MODEL}


# # ViT2capfilt_L adv4
# SOURCE_MODEL=vit_L
# TARGET_MODEL=capfilt_L
# ADV=4
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml --evaluate --adv ${ADV} --batch_size 8 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/${SOURCE_MODEL}2${TARGET_MODEL}/adv${ADV} --source_model ${SOURCE_MODEL} --target_model ${TARGET_MODEL}
# # capfilt_L2ViT adv4
# SOURCE_MODEL=capfilt_L
# TARGET_MODEL=vit_L
# ADV=4
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml --evaluate --adv ${ADV} --batch_size 8 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/${SOURCE_MODEL}2${TARGET_MODEL}/adv${ADV} --source_model ${SOURCE_MODEL} --target_model ${TARGET_MODEL}


# # ViT2capfilt_L adv6
# SOURCE_MODEL=vit_L
# TARGET_MODEL=capfilt_L
# ADV=6
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml --evaluate --adv ${ADV} --batch_size 1 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/${SOURCE_MODEL}2${TARGET_MODEL}/adv${ADV} --source_model ${SOURCE_MODEL} --target_model ${TARGET_MODEL}
# # capfilt_L2ViT adv6
# SOURCE_MODEL=capfilt_L
# TARGET_MODEL=vit_L
# ADV=6
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_capfilt_L.yaml --evaluate --adv ${ADV} --batch_size 1 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/${SOURCE_MODEL}2${TARGET_MODEL}/adv${ADV} --source_model ${SOURCE_MODEL} --target_model ${TARGET_MODEL}



MODEL=vit_L
python -m torch.distributed.run --nproc_per_node=16 train_caption_attack_base.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_${MODEL}.yaml --evaluate --adv 6 --batch_size 1 --epsilon 2 --output_dir /home/zhengf/ld/BLIP_sammy/output/caption_coco/${MODEL}/adv${ADV}
