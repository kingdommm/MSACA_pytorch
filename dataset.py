import torch
import numpy as np
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, t_file, i_file, l_file):
        test_text = np.load(t_file, allow_pickle=True)
        test_text_data = []
        for i in test_text:
            test_text_data.append(i.clone().squeeze(0))
        test_text = torch.stack(test_text_data)
        test_img = np.load(i_file, allow_pickle=True)
        test_label = np.load(l_file, allow_pickle=True)
        self.test_data_text = test_text
        self.test_data_img = torch.from_numpy(test_img).squeeze().float()
        self.test_labels = torch.from_numpy(test_label).long()

    def __len__(self):
        return self.test_data_text.shape[0]

    def __getitem__(self, item):
        return self.test_data_text[item], self.test_data_img[item], self.test_labels[item]
