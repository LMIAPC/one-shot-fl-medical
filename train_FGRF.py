"""This implementation is adapted from:
# Author: [cloneofsimo]
# Repository: https://github.com/cloneofsimo/minRF
# License: Apache License, Version 2.0, January 2004 (http://www.apache.org/licenses/LICENSE-2.0)
# Modifications: [We proposed Feature-Guided Rectified Flow Model to reduce privacy leakage.
See more descriptions in our paper.]
"""
# Clients train local FG-RF models
import argparse
import random

import torch
import os
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from models.dit import DiT_Llama
from models.rf import RF


def parser_args():
    parser = argparse.ArgumentParser(description="FG-RF to generate images on each client")
    parser.add_argument('--dataset', type=str, default='brain_tumor',
                        choices=['TB', 'skin_cancer', 'brain_tumor'],
                        help='Choose between "TB", "skin_cancer", "brain_tumor"')
    parser.add_argument('--num_classes', type=int, default=4, help="number of classes")
    parser.add_argument('--channels', type=int, default=1, help="number of channels")
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=800, help='number of epochs')
    parser.add_argument('--cuda', type=int, default=1, help='choose cuda number')
    parser.add_argument('--client', type=int, default=0, help='client id')
    # parser.add_argument('--data_dir', type=str,
    #                     default='/mnt/lv01/myf/medical_dataset/brain_tumor/Training',
    #                     help='Path to the original training data directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()

    torch.cuda.set_device(args.cuda)
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # define seed
    seed = 66
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # create generator
    generator = torch.Generator()
    generator.manual_seed(seed)

    # define model saved paths
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    metadata_save_path = os.path.join(save_dir, f'{args.dataset}_client_{args.client}_metadata.pth')

    if args.dataset == 'brain_tumor':
        #
        dataset_name = "brain_tumor"
        metadata = torch.load(metadata_save_path)
        features = metadata["features"]  # [32, 1, 224, 224]

        labels = metadata["labels"]

        train_dataset = TensorDataset(features, labels)

        channels = args.channels

        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
        model = DiT_Llama(
            channels, args.image_size, dim=256, n_layers=10, n_heads=8, patch_size=8, num_classes=args.num_classes
        ).cuda()

    elif args.dataset == 'skin_cancer':
        dataset_name = "skin_cancer"

        metadata = torch.load(metadata_save_path)
        features = metadata["features"]  # [32, 3, 224, 224]

        # tensor_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomRotation(20),
        #     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        #     # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),  # 随机擦除
        # ])
        #
        # augmented_tensor = tensor_transforms(features)

        labels = metadata["labels"]

        train_dataset = TensorDataset(features, labels)

        channels = args.channels
        print(f'channels: {channels}')

        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
        model = DiT_Llama(
            channels, args.image_size, dim=256, n_layers=10, n_heads=8, patch_size=8, num_classes=args.num_classes
        ).cuda()

    else:
        dataset_name = "TB"
        metadata = torch.load(metadata_save_path)
        features = metadata["features"]  # [32, 1, 224, 224]

        labels = metadata["labels"]

        train_dataset = TensorDataset(features, labels)
        channels = args.channels

        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
        model = DiT_Llama(
            channels, args.image_size, dim=256, n_layers=10, n_heads=8, patch_size=8, num_classes=args.num_classes
        ).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    wandb.init(project=f"rf_{dataset_name}", mode="offline")

    save_temp = os.path.join('./generated_contents/temp', args.dataset, f'client_{args.client}')
    os.makedirs(save_temp, exist_ok=True)
    save_model = os.path.join('./generated_contents/models', args.dataset, f'client_{args.client}')
    os.makedirs(save_model, exist_ok=True)

    for epoch in range(args.epochs):  # 100
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.cuda(), c.cuda()

            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)

            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

        wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

        torch.save(rf.model.state_dict(), os.path.join(save_model, f"ckpt_latest.pt"))
        if epoch % 50 == 0:
            torch.save(rf.model.state_dict(), os.path.join(save_model, f"ckpt_{epoch}.pt"))

        rf.model.eval()
        with torch.no_grad():
            # cond = torch.arange(0, 16).cuda() % 10
            # uncond = torch.ones_like(cond) * 10
            # cond = torch.arange(0, 16).cuda() % 4
            cond = torch.arange(0, args.num_classes).cuda()
            uncond = torch.ones_like(cond) * args.num_classes

            # init_noise = torch.randn(16, channels, 32, 32).cuda()
            init_noise = torch.randn(args.num_classes, channels, args.image_size, args.image_size).cuda()  # 16yuanlai
            images = rf.sample(init_noise, cond, uncond)
            # image sequences to gif
            gif = []
            for image in images:
                # unnormalize
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=2)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            # gif[0].save(
            #     # f"contents/sample_{epoch}.gif",
            #     os.path.join(save_temp, f"sample_{epoch}.gif"),
            #     save_all=True,
            #     append_images=gif[1:],
            #     duration=100,
            #     loop=0,
            # )

            last_img = gif[-1]
            last_img.save(os.path.join(save_temp, f"sample_{epoch}_last.png"))

        rf.model.train()
