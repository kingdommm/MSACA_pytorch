import numpy as np
import torch
import tqdm
import pandas as pd
import pickle as pkl
from transformers import BertTokenizer, BertConfig
from transformers import BertModel

seq_len = 512
MODEL_PATH = '/home/ubuntu2204/Desktop/My_Fake_News_Detection/bert_base_uncased'

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model_config = BertConfig.from_pretrained(MODEL_PATH)
model_config.output_hidden_states = True
model_config.output_attentions = True
bert_model = BertModel.from_pretrained(MODEL_PATH, config=model_config)


def get_fake_news_net_matrix(data_type, tokenizer):
    corpus_dir = '/XXXX/gossipcop'
    all_img_embed = pkl.load(
        open('/XXXX/Images/img_emb_resnet34_3scale.pkl', 'rb'))
    gossip_content = pd.read_csv('{}/gossip_{}.csv'.format(corpus_dir, data_type))['content'].tolist()
    gossip_image_name = pd.read_csv('{}/gossip_{}.csv'.format(corpus_dir, data_type))['image'].tolist()
    gossip_label = pd.read_csv('{}/gossip_{}.csv'.format(corpus_dir, data_type))['label'].tolist()

    gossip_text_matrix = []
    gossip_image_matrix = []
    gossip_labels = []

    num_content = len(gossip_content)
    for idx in range(num_content):
        one_rumor = gossip_content[idx].strip()
        if one_rumor:
            image = gossip_image_name[idx].split('.')[0]
            if image in all_img_embed:
                gossip_image_matrix.append(all_img_embed[image])
                gossip_labels.append(gossip_label[idx])
                text_tokens = tokenizer.encode_plus(one_rumor,
                                                    truncation=True, max_length=seq_len, return_tensors='pt',
                                                    padding='max_length', return_token_type_ids=True,
                                                    return_attention_mask=True)
                gossip_text_matrix.append(text_tokens)

    num_content = len(politi_content)
    for idx in range(num_content):
        one_rumor = politi_content[idx].strip()
        if one_rumor:
            image = politi_image_name[idx].split('.')[0]
            if image in all_img_embed:
                politi_image_matrix.append(all_img_embed[image])
                politi_labels.append(politi_label[idx])
                text_tokens = tokenizer.encode_plus(one_rumor,
                                                    truncation=True, max_length=seq_len, return_tensors='pt',
                                                    padding='max_length', return_token_type_ids=True,
                                                    return_attention_mask=True)
                politi_text_matrix.append(text_tokens)

    return gossip_text_matrix, gossip_image_matrix, gossip_labels


if __name__ == '__main__':
    gossip_train_text_matrix, gossip_train_image_matrix, gossip_train_labels = get_fake_news_net_matrix('train', tokenizer=tokenizer)
    gossip_test_text_matrix, gossip_test_image_matrix, gossip_test_labels = get_fake_news_net_matrix('test', tokenizer=tokenizer)

    gossip_train_embeddings = []
    gossip_test_embeddings = []
    bert_model.eval()

    with torch.no_grad():
        for i in tqdm.tqdm(gossip_train_text_matrix):
            tokens_tensor, segments_tensor, attention_mask = i['input_ids'], i['token_type_ids'], i['attention_mask']
            embedding = bert_model(tokens_tensor, token_type_ids=segments_tensor, attention_mask=attention_mask)
            gossip_train_embeddings.append(embedding.last_hidden_state)
        gossip_train_embeddings = np.array(gossip_train_embeddings)

        for j in tqdm.tqdm(gossip_test_text_matrix):
            tokens_tensor, segments_tensor, attention_mask = j['input_ids'], j['token_type_ids'], j['attention_mask']
            embedding = bert_model(tokens_tensor, token_type_ids=segments_tensor, attention_mask=attention_mask)
            gossip_test_embeddings.append(embedding.last_hidden_state)
        gossip_test_embeddings = np.array(gossip_test_embeddings)

    gossip_train_text_matrix = np.array(gossip_train_text_matrix)
    gossip_train_image_matrix = np.array(gossip_train_image_matrix)
    gossip_train_labels = np.array(gossip_train_labels)
    gossip_test_text_matrix = np.array(gossip_test_text_matrix)
    gossip_test_image_matrix = np.array(gossip_test_image_matrix)
    gossip_test_labels = np.array(gossip_test_labels)

    print('gossip_train text embeddings: ', gossip_train_embeddings.shape)
    print('gossip_train image embeddings: ', gossip_train_image_matrix.shape)
    print('gossip_train labels: ', gossip_train_labels.shape)

    print('gossip_test text embeddings: ', gossip_test_embeddings.shape)
    print('gossip_test image embeddings: ', gossip_test_image_matrix.shape)
    print('gossip_test labels: ', gossip_test_labels.shape)

    matrix_save_dir = '/XXXX/gossipcop_dataset_3scale'
    np.save('{}/gossip_train_text_embed'.format(matrix_save_dir), gossip_train_embeddings)
    np.save('{}/gossip_train_image_embed'.format(matrix_save_dir), gossip_train_image_matrix)
    np.save('{}/gossip_train_label'.format(matrix_save_dir), gossip_train_labels)

    np.save('{}/gossip_test_text_embed'.format(matrix_save_dir), gossip_test_embeddings)
    np.save('{}/gossip_test_image_embed'.format(matrix_save_dir), gossip_test_image_matrix)
    np.save('{}/gossip_test_label'.format(matrix_save_dir), gossip_test_labels)
