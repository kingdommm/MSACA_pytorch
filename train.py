import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from dataset import FeatureDataset
from MSACA_model import VaeModule, DetectionModule


# Configs
DEVICE = "cuda:0"
NUM_WORKER = 0
BATCH_SIZE = 64
LR = 5e-4
L2 = 1e-5  # 1e-5
VAE_WEIGHT = 0.2
NUM_EPOCH = 100


def prepare_data(text, image, label):
    nr_index = [i for i, l in enumerate(label) if l == 1]
    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    return fixed_text, matched_image


def train():

    # ---  Load Config  ---
    device = torch.device(DEVICE)
    num_workers = NUM_WORKER
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    vae_weight = VAE_WEIGHT
    num_epoch = NUM_EPOCH

    # ---  Load Data  ---
    dataset_dir = '/XXXX/weibo_dataset_3scale'
    train_set = FeatureDataset(
        "{}/{}train_text_embed.npy".format(dataset_dir, dataset),
        "{}/{}train_image_embed.npy".format(dataset_dir, dataset),
        "{}/{}train_label.npy".format(dataset_dir, dataset)
    )
    test_set = FeatureDataset(
        "{}/{}test_text_embed.npy".format(dataset_dir, dataset),
        "{}/{}test_image_embed.npy".format(dataset_dir, dataset),
        "{}/{}test_label.npy".format(dataset_dir, dataset)
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )

    # ---  Build Model & Trainer  ---
    vae_module = VaeModule()
    vae_module.to(device)
    detection_module = DetectionModule()
    detection_module.to(device)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    optim_task_detection = torch.optim.Adam(
        detection_module.parameters(), lr=lr, weight_decay=l2)

    # ---  Model Training  ---
    best_acc = 0
    for epoch in range(num_epoch):
        vae_module.train()
        detection_module.train()
        corrects_pre_detection = 0
        loss_detection_total = 0
        detection_count = 0

        for text, image, label in tqdm(train_loader):

            text = text.to(device)
            image_224 = image.permute(1, 0, 2)[0].to(device)
            image_112 = image.permute(1, 0, 2)[1].to(device)
            image_56 = image.permute(1, 0, 2)[2].to(device)
            label = label.to(device)
            fixed_text, matched_image = prepare_data(text, image, label)
            matched_image_224 = matched_image[0].to(device)
            matched_image_112 = matched_image[1].to(device)
            matched_image_56 = matched_image[2].to(device)

            # ---  TASK2 Detection  ---

            kl_text_224, kl_image_224, z1_text_match_224, z2_image_match_224 = vae_module(fixed_text, matched_image_224)
            kl_text_112, kl_image_112, z1_text_match_112, z2_image_match_112 = vae_module(fixed_text, matched_image_112)
            kl_text_56, kl_image_56, z1_text_match_56, z2_image_match_56 = vae_module(fixed_text, matched_image_56)

            loss_vae = (kl_text_224 + kl_text_112 + kl_text_56) / 3 + (kl_image_224 + kl_image_112 + kl_image_56) / 3

            kl_text_224, kl_image_224, z1_given_text_224, z2_given_image_224 = vae_module(text, image_224)
            kl_text_112, kl_image_112, z1_given_text_112, z2_given_image_112 = vae_module(text, image_112)
            kl_text_56, kl_image_56, z1_given_text_56, z2_given_image_56 = vae_module(text, image_56)
            pre_detection = detection_module(text, image_224, z1_given_text_224, z2_given_image_224,
                                                          text, image_112, z1_given_text_112, z2_given_image_112,
                                                          text, image_56, z1_given_text_56, z2_given_image_56)

            loss_detection = loss_func_detection(pre_detection, label) + vae_weight * loss_vae

            optim_task_detection.zero_grad()
            loss_detection.backward()
            optim_task_detection.step()

            pre_label_detection = pre_detection.argmax(1)
            corrects_pre_detection += pre_label_detection.eq(label.view_as(pre_label_detection)).sum().item()

            # ---  Record  ---

            loss_detection_total += loss_detection.item() * text.shape[0]
            detection_count += text.shape[0]

        loss_detection_train = loss_detection_total / detection_count
        acc_detection_train = corrects_pre_detection / detection_count

        # ---  Test  ---

        acc_detection_test, loss_detection_test, precision_scores_f, recall_scores_f, f1_scores_f, precision_scores_t, recall_scores_t, f1_scores_t = \
            test(vae_module, detection_module, test_loader)

        # ---  Output  ---

        print('---  TASK Detection  ---')
        print(
            "EPOCH = %d \n acc_detection_train = %.3f \n acc_detection_test = %.3f \n loss_detection_train = %.3f \n loss_detection_test = %.3f \n" %
            (epoch + 1, acc_detection_train, acc_detection_test, loss_detection_train, loss_detection_test)
        )

        print('precision_scores_rumor:', precision_scores_f, 'recall_scores_rumor:', recall_scores_f,
              'f1_scores_rumor:', f1_scores_f)
        print('precision_scores_non_rumor:', precision_scores_t, 'recall_scores_non_rumor:', recall_scores_t,
              'f1_scores_non_rumor:', f1_scores_t)


def test(vae_module, detection_module, test_loader):
    vae_module.eval()
    detection_module.eval()
    device = torch.device(DEVICE)
    vae_weight = VAE_WEIGHT
    loss_func_detection = torch.nn.CrossEntropyLoss()

    detection_count = 0
    loss_detection_total = 0
    detection_label_all = []
    detection_pre_label_all = []

    with torch.no_grad():
        for text, image, label in tqdm(test_loader):
            text = text.to(device)
            image_224 = image.permute(1, 0, 2)[0].to(device)
            image_112 = image.permute(1, 0, 2)[1].to(device)
            image_56 = image.permute(1, 0, 2)[2].to(device)
            label = label.to(device)

            # ---  TASK1 Similarity  ---

            kl_text_224, kl_image_224, _, _ = vae_module(text, image_224)
            kl_text_112, kl_image_112, _, _ = vae_module(text, image_112)
            kl_text_56, kl_image_56, _, _ = vae_module(text, image_56)

            loss_vae = (kl_text_224 + kl_text_112 + kl_text_56) / 3 + (kl_image_224 + kl_image_112 + kl_image_56) / 3

            # ---  TASK2 Detection  ---
            _, _, z1_given_text_224, z2_given_image_224 = vae_module(text, image_224)
            _, _, z1_given_text_112, z2_given_image_112 = vae_module(text, image_112)
            _, _, z1_given_text_56, z2_given_image_56 = vae_module(text, image_56)

            pre_detection = detection_module(text, image_224, z1_given_text_224, z2_given_image_224,
                                                          text, image_112, z1_given_text_112, z2_given_image_112,
                                                          text, image_56, z1_given_text_56, z2_given_image_56)
            loss_detection = loss_func_detection(pre_detection, label) + vae_weight * loss_vae
            pre_label_detection = pre_detection.argmax(1)

            # ---  Record  ---

            loss_detection_total += loss_detection.item() * text.shape[0]
            detection_count += text.shape[0]

            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            detection_label_all.append(label.detach().cpu().numpy())

        loss_detection_test = loss_detection_total / detection_count

        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)

        acc_detection_test = accuracy_score(detection_label_all, detection_pre_label_all)
        f1_scores_1 = f1_score(detection_label_all, detection_pre_label_all, pos_label=1)
        recall_scores_1 = recall_score(detection_label_all, detection_pre_label_all, pos_label=1)
        precision_scores_1 = precision_score(detection_label_all, detection_pre_label_all, pos_label=1)

        f1_scores_2 = f1_score(detection_label_all, detection_pre_label_all, pos_label=0)
        recall_scores_2 = recall_score(detection_label_all, detection_pre_label_all, pos_label=0)
        precision_scores_2 = precision_score(detection_label_all, detection_pre_label_all, pos_label=0)

    return acc_detection_test, loss_detection_test, precision_scores_1, recall_scores_1, f1_scores_1, \
        precision_scores_2, recall_scores_2, f1_scores_2


if __name__ == "__main__":
    train()
