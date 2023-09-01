

# SOURCE_MODEL=vit_L
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack_save_adv_data.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_${SOURCE_MODEL}.yaml --evaluate --adv 3 --batch_size 8 --epsilon 2 --source_model ${SOURCE_MODEL}

# SOURCE_MODEL=vit_L
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack_save_adv_data.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_${SOURCE_MODEL}.yaml --evaluate --adv 4 --batch_size 4 --epsilon 2 --source_model ${SOURCE_MODEL}

SOURCE_MODEL=vit_L
python -m torch.distributed.run --nproc_per_node=16 train_caption_attack_save_adv_data.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_${SOURCE_MODEL}.yaml --evaluate --adv 6 --batch_size 1 --epsilon 2 --source_model ${SOURCE_MODEL}

# SOURCE_MODEL=capfilt_L
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack_save_adv_data.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_${SOURCE_MODEL}.yaml --evaluate --adv 3 --batch_size 8 --epsilon 2 --source_model ${SOURCE_MODEL}

# SOURCE_MODEL=capfilt_L
# python -m torch.distributed.run --nproc_per_node=16 train_caption_attack_save_adv_data.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_${SOURCE_MODEL}.yaml --evaluate --adv 4 --batch_size 8 --epsilon 2 --source_model ${SOURCE_MODEL}

SOURCE_MODEL=capfilt_L
python -m torch.distributed.run --nproc_per_node=16 train_caption_attack_save_adv_data.py --config /home/zhengf/ld/BLIP_sammy/configs/caption_coco_${SOURCE_MODEL}.yaml --evaluate --adv 6 --batch_size 1 --epsilon 2 --source_model ${SOURCE_MODEL}
