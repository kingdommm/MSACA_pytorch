import math
import random
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


class FastCNN(nn.Module):
    def __init__(self, channel=128, kernel_size=(1, 2, 4, 8)):
        super(FastCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(
                    nn.Conv1d(768, channel, kernel_size=kernel),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)
                )
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_out = []
        for module in self.fast_cnn:
            x_out.append(module(x).squeeze())
        for index, items in enumerate(x_out):
            if len(x_out[index].shape) == 1:
                x_out[index] = x_out[index].unsqueeze(0)
        x_out = torch.cat(x_out, 1)
        return x_out


class EncodingPart(nn.Module):
    def __init__(
        self,
        cnn_channel=128,
        cnn_kernel_size=(1, 2, 4, 8),
        shared_image_dim=128,
        shared_text_dim=128
    ):
        super(EncodingPart, self).__init__()
        self.shared_text_encoding = FastCNN(
            channel=cnn_channel,
            kernel_size=cnn_kernel_size
        )
        self.shared_text_linear = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class Encoder(nn.Module):
    def __init__(self, z_dim=64):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        kl_divergence = -0.5 * torch.sum(1 + torch.log(sigma ** 2) - (mu ** 2) - (sigma ** 2))
        return Independent(Normal(loc=mu, scale=sigma), 1), kl_divergence


class VaeModule(nn.Module):
    def __init__(self, shared_dim=128, sim_dim=64):
        super(VaeModule, self).__init__()
        self.encoding_module = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_encoding, image_encoding = self.encoding_module(text, image)
        text_aligned = self.text_aligner(text_encoding)
        image_aligned = self.image_aligner(image_encoding)
        p_z1_given_text, kl_text = self.encoder_text(text_aligned)
        p_z2_given_image, kl_image = self.encoder_image(image_aligned)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        return kl_text, kl_image, z1, z2


class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=64, prime_dim=16):
        super(UnimodalDetection, self).__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )

    def forward(self, z1, z2):
        text_prime = self.text_uni(z1)
        image_prime = self.image_uni(z2)
        return text_prime, image_prime


class CrossModule4Batch(nn.Module):
    def __init__(self, corre_dim=64, corre_out_dim=64):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text_encoding, image_encoding):
        text_in = text_encoding.unsqueeze(2)
        image_in = image_encoding.unsqueeze(1)
        corre_dim = text_encoding.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        correlation_out = self.c_specific_2(correlation_p)
        return correlation_out


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        self.vaemodule = VaeModule()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()
        self.uni_repre = UnimodalDetection()
        self.cross_repre = CrossModule4Batch()
        self.cor_att = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 7),
            nn.BatchNorm1d(7),
            nn.ReLU()
        )

    def forward(self, z1_given_text_224, z2_given_image_224, z1_given_text_112, z2_given_image_112, z1_given_text_56, z2_given_image_56):
        p_z1_224, _ = self.encoder_text(z1_given_text_224)
        z1_224 = p_z1_224.rsample()
        p_z2_224, _ = self.encoder_image(z2_given_image_224)
        z2_224 = p_z2_224.rsample()
        text_zprime_224, image_zprime_224 = self.uni_repre(z1_224, z2_224)
        zcorrelation_224 = self.cross_repre(z1_224, z2_224)

        p_z1_112, _ = self.encoder_text(z1_given_text_112)
        z1_112 = p_z1_112.rsample()
        p_z2_112, _ = self.encoder_image(z2_given_image_112)
        z2_112 = p_z2_112.rsample()
        text_zprime_112, image_zprime_112 = self.uni_repre(z1_112, z2_112)
        zcorrelation_112 = self.cross_repre(z1_112, z2_112)

        p_z1_56, _ = self.encoder_text(z1_given_text_56)
        z1_56 = p_z1_56.rsample()
        p_z2_56, _ = self.encoder_image(z2_given_image_56)
        z2_56 = p_z2_56.rsample()
        text_zprime_56, image_zprime_56 = self.uni_repre(z1_56, z2_56)
        zcorrelation_56 = self.cross_repre(z1_56, z2_56)

        cor = torch.cat((text_zprime_224, image_zprime_224, image_zprime_112, image_zprime_56, zcorrelation_224,
                         zcorrelation_112, zcorrelation_56), dim=1)
        att = torch.softmax(self.cor_att(cor), dim=1)
        return att


class DetectionModule(nn.Module):
    def __init__(self, feature_dim=256, h_dim=64):
        super(DetectionModule, self).__init__()
        self.ambiguity_module = AmbiguityLearning()
        self.encoding_module = EncodingPart()
        self.vae_module = VaeModule()
        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule4Batch()
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h_dim, 2)
        )

    def forward(self, text_224, image_224, z1_given_text_224, z2_given_image_224,
                text_112, image_112, z1_given_text_112, z2_given_image_112,
                text_56, image_56, z1_given_text_56, z2_given_image_56):
        _, _, z1_224, z2_224 = self.vae_module(text_224, image_224)
        text_prime_224, image_prime_224 = self.uni_repre(z1_224, z2_224)
        correlation_224 = self.cross_module(z1_224, z2_224)

        _, _, z1_112, z2_112 = self.vae_module(text_112, image_112)
        text_prime_112, image_prime_112 = self.uni_repre(z1_112, z2_112)
        correlation_112 = self.cross_module(z1_112, z2_112)

        _, _, z1_56, z2_56 = self.vae_module(text_56, image_56)
        text_prime_56, image_prime_56 = self.uni_repre(z1_56, z2_56)
        correlation_56 = self.cross_module(z1_56, z2_56)

        att = self.ambiguity_module(z1_given_text_224, z2_given_image_224, z1_given_text_112, z2_given_image_112, z1_given_text_56, z2_given_image_56)
        text_final = att[:, 0].unsqueeze(1) * text_prime_224
        img_final_224 = att[:, 1].unsqueeze(1) * image_prime_224
        img_final_112 = att[:, 2].unsqueeze(1) * image_prime_112
        img_final_56 = att[:, 3].unsqueeze(1) * image_prime_56
        cor_final_224 = att[:, 4].unsqueeze(1) * correlation_224
        cor_final_112 = att[:, 5].unsqueeze(1) * correlation_112
        cor_final_56 = att[:, 6].unsqueeze(1) * correlation_56

        final_corre = torch.cat([text_final, img_final_224, img_final_112, img_final_56,
                                 cor_final_224, cor_final_112, cor_final_56], 1)
        pre_label = self.classifier_corre(final_corre)
        return pre_label

