import numpy as np
import torch
import tqdm
import os
import pickle as pkl
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel


seq_len = 120

model_name = 'bert-base-chinese'
MODEL_PATH = r'/bert_base_Chinese'

tokenizer = BertTokenizer.from_pretrained(model_name)
model_config = BertConfig.from_pretrained(model_name)
model_config.output_hidden_states = True
model_config.output_attentions = True
bert_model = BertModel.from_pretrained(MODEL_PATH, config=model_config)


def get_weibo_matrix(data_type, tokenizer):
    all_img_embed = pkl.load(
        open(r'/XXX/img_emb_resnet34.pkl', 'rb'))
    rumor_content = open('{}/{}_rumor.txt'.format(corpus_dir, data_type)).readlines()
    nonrumor_content = open('{}/{}_nonrumor.txt'.format(corpus_dir, data_type)).readlines()

    text_matrix = []
    image_matrix = []
    labels = []

    n_lines = len(rumor_content)
    for idx in range(2, n_lines, 3):
        one_rumor = rumor_content[idx].strip()
        if one_rumor:
            images = rumor_content[idx-1].split('|')
            corpus_dir = '/home/ubuntu2204/Desktop/My_Fake_News_'
            for image in images:
                img = image.split('/')[-1].split('.')[0]
                if img in all_img_embed:
                    image_matrix.append(all_img_embed[img])
                    labels.append(1)
                    text_tokens = tokenizer.encode_plus(one_rumor,
                                                        truncation=True, max_length=seq_len, return_tensors='pt',
                                                        padding='max_length', return_token_type_ids=True,
                                                        return_attention_mask=True)
                    text_matrix.append(text_tokens)
                    break

    n_lines = len(nonrumor_content)
    for idx in range(2, n_lines, 3):
        one_rumor = nonrumor_content[idx].strip()
        if one_rumor:
            images = nonrumor_content[idx-1].split('|')
            for image in images:
                img = image.split('/')[-1].split('.')[0]
                if img in all_img_embed:
                    image_matrix.append(all_img_embed[img])
                    labels.append(0)
                    text_tokens = tokenizer.encode_plus(one_rumor,
                                                        truncation=True, max_length=seq_len, return_tensors='pt',
                                                        padding='max_length', return_token_type_ids=True,
                                                        return_attention_mask=True)
                    text_matrix.append(text_tokens)
                    break
    return text_matrix, image_matrix, labels


if __name__ == '__main__':
    train_text_matrix, train_image_matrix, train_labels = get_weibo_matrix('train', tokenizer=tokenizer)
    test_text_matrix, test_image_matrix, test_labels = get_weibo_matrix('test', tokenizer=tokenizer)

    train_embeddings = []
    test_embeddings = []
    bert_model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(train_text_matrix):
            tokens_tensor, segments_tensor, attention_mask = i['input_ids'], i['token_type_ids'], i['attention_mask']
            embedding = bert_model(tokens_tensor, token_type_ids=segments_tensor, attention_mask=attention_mask)
            train_embeddings.append(embedding.last_hidden_state)
        train_embeddings = np.array(train_embeddings)

        for j in tqdm.tqdm(test_text_matrix):
            tokens_tensor, segments_tensor, attention_mask = j['input_ids'], j['token_type_ids'], j['attention_mask']
            embedding = bert_model(tokens_tensor, token_type_ids=segments_tensor, attention_mask=attention_mask)
            test_embeddings.append(embedding.last_hidden_state)
        test_embeddings = np.array(test_embeddings)

    train_text_matrix = np.array(train_text_matrix)
    train_image_matrix = np.array(train_image_matrix)
    train_labels = np.array(train_labels)

    test_text_matrix = np.array(test_text_matrix)
    test_image_matrix = np.array(test_image_matrix)
    test_labels = np.array(test_labels)
    print('train text embeddings: ', train_embeddings.shape)
    print('train image embeddings: ', train_image_matrix.shape)
    print('train labels: ', train_labels.shape)

    print('test text embeddings: ', test_embeddings.shape)
    print('test image embeddings: ', test_image_matrix.shape)
    print('test labels: ', test_labels.shape)

    matrix_save_dir = '/XXXX/weibo_dataset_3scale'

    np.save('{}/train_text_embed'.format(matrix_save_dir), train_embeddings)
    np.save('{}/train_image_embed'.format(matrix_save_dir), train_image_matrix)
    np.save('{}/train_label'.format(matrix_save_dir), train_labels)

    np.save('{}/test_text_embed'.format(matrix_save_dir), test_embeddings)
    np.save('{}/test_image_embed'.format(matrix_save_dir), test_image_matrix)
    np.save('{}/test_label'.format(matrix_save_dir), test_labels)
