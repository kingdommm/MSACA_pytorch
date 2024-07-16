import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import tqdm
import pickle as pkl


def img_gen(images_dir_list, save_emb_dir, output_format='pkl'):
    resnet_model = models.resnet34(pretrained=True)
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
    resnet_model.eval()
    img2emb = {}
    for image_dir in images_dir_list:
        print('--------------{}--------------'.format(image_dir))
        for imgs in tqdm.tqdm(os.listdir(image_dir)):
            img_name = imgs.split('.')[0]
            img_type = imgs.split('.')[1]
            if img_type == 'gif':
                continue
            if img_type == 'txt':
                continue
            im = Image.open('{}/{}'.format(image_dir, imgs)).convert('RGB')
            im_224 = transforms.Resize((224, 224))(im)
            im_112 = transforms.Resize((112, 112))(im)
            im_56 = transforms.Resize((56, 56))(im)
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            im_224 = trans(im_224).unsqueeze_(dim=0)
            im_112 = trans(im_112).unsqueeze_(dim=0)
            im_56 = trans(im_56).unsqueeze_(dim=0)
            with torch.no_grad():
                out_224 = resnet_model(im_224)
                out_112 = resnet_model(im_112)
                out_56 = resnet_model(im_56)
            out = torch.cat((out_224, out_112, out_56), dim=0)
            out_np = list(out.numpy())
            img2emb[img_name] = out_np

    if output_format == 'pkl':
        pkl.dump(img2emb, open(save_emb_dir, 'wb'))
    elif output_format == 'txt':
        f_img = open(save_emb_dir, 'w')
        for img in img2emb:
            line = '{},{}\n'.format(img, ','.join(str(i) for i in img2emb[img]))
            f_img.write(line)
        f_img.close()
    else:
        raise ValueError('the output_format only support pkl or txt!')


def run(dataset='GossipCop'):
    if dataset == 'GossipCop':
        main_dir = '/XXXX/AAAI_dataset/Images'
        image_files_dir_list = ['{}/gossip_test'.format(main_dir),
                                '{}/gossip_train'.format(main_dir)]
        save_image_emb_file = '{}/img_emb_resnet34_3scale.pkl'.format(main_dir)
        emb_output_format = 'pkl'
        img_gen(image_files_dir_list, save_image_emb_file, emb_output_format)
    elif dataset == 'weibo':
        main_dir = '/XXXX/MM17-WeiboRumorSet'
        image_files_dir_list = ['{}/nonrumor_images'.format(main_dir),
                                '{}/rumor_images'.format(main_dir)]
        save_image_emb_file = '{}/img_emb_resnet34_3scale.pkl'.format(main_dir)
        emb_output_format = 'pkl'
        img_gen(image_files_dir_list, save_image_emb_file, emb_output_format)
    else:
        raise ValueError('ERROR! dataset must be weibo or GossipCop.')


if __name__ == '__main__':
    run('Weibo')
