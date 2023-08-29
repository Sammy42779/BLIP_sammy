import torch
import torch.nn as nn
import copy
from transformers import BatchEncoding
import torch.nn.functional as F

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)

class BertAttackFusion_caption():
    # attack text 时候需要image信息
    def __init__(self, ref_net, ref_tokenizer):
        self.ref_net = ref_net
        self.ref_tokenizer = ref_tokenizer
    ## 这里传入了BLIP_decoder, 也就是net
    def attack(self, net, images, texts, k=10, num_perturbation=1, threshold_pred_score=0.3, max_length=40, batch_size=32):
        """generate adversarial texts

        Args:
            net (BLIP_decoder): _description_
            images (Tensor): normalized images; shape=[bs,3,384,384]
            texts (List): ground truth text batch [with the prompt: 'a picture of ']; natural language; len=bs
            k (int, optional): candidate number when attacking texts. Defaults to 10.
            num_perturbation (int, optional): number of word can be perturbed. Defaults to 1.
            threshold_pred_score (float, optional): used to generate adversarial texts. Defaults to 0.3.
            max_length (int, optional): max length for texts. Defaults to 40.
            batch_size (int, optional): number of masked texts to process at one time. Defaults to 32.

        Returns:
            List: final adversarial texts for this batch
        """
        device = self.ref_net.device
        
        # BertForMaskedLM
        # 计算单词重要性, 对比masked texts和image的loss,谁的loss最大, 谁就是重要的单词
        # substitutes 先找到每个单词位置被ref_net预测的替换单词
        texts_with_no_prompt = [text[len(net.prompt):] for text in texts]
        ref_text_inputs = self.ref_tokenizer(texts_with_no_prompt, padding='max_length', truncation=True, max_length=max_length,
                                     return_tensors='pt').to(device)
        mlm_logits = self.ref_net(ref_text_inputs.input_ids, attention_mask=ref_text_inputs.attention_mask).logits  # torch.Size([32, 27, 30522])
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, k, -1)  # seq-len k shape=[bs, longest_len, k=10]; score and prediction_id
        
        final_adverse = []
        for i, text in enumerate(texts_with_no_prompt):
            # word importance eval
            important_scores = self.get_important_scores(images[i], text, net, batch_size)  # 不考虑promt单词的重要性

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.ref_tokenizer, self.ref_net, 1, word_pred_scores,
                                             threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                # 对于一个文本, 替换重要单词后的文本; 这里应该是一个替换文本; 
                # len(replace_text_input)<=10, 这里是texts[i]这个句子所有的替换后的对抗文本
                image_batch = images[i].repeat(len(replace_texts), 1, 1, 1)  # torch.Size([14(texts_len), 3, 384, 384])
                replace_texts_with_prompts = [net.prompt + t for t in replace_texts]
                loss = net.get_lm_loss(image_batch, replace_texts_with_prompts, loss_reduction='none')  # shape=[len(replace_texts)]
                
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.ref_tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, image, text, net, batch_size):

        masked_words = self._get_masked(text)
        # list of text of masked words; no prompt
        masked_texts = [' '.join(words) for words in masked_words]  
        # add promtps for loss computation
        masked_texts = [net.prompt + text for text in masked_texts]

        import_scores = []
        for i in range(0, len(masked_texts), batch_size):  # '[UNK] man with a red helmet on a small moped on a dirt road'
            # texts (batch_size) for masked_texts

            # 和文本保持相同数量, 但是每个文本都是同一个图像
            image_batch = image.repeat(len(masked_texts[i:i+batch_size]), 1, 1, 1)  # torch.Size([14(texts_len), 3, 384, 384])
            masked_losses = net.get_lm_loss(image_batch, masked_texts[i:i+batch_size], loss_reduction='none')

            import_scores += masked_losses
        
        return import_scores  # 14个important score, 显示每个单词的重要性, 14=len(masked_texts)


def get_substitues(substitutes, ref_tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(ref_tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, ref_tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, ref_tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [ref_tokenizer._convert_id_to_token(int(i)) for i in word]
        text = ref_tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
