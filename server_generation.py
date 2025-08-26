# Server image generation.
import argparse
import sys
"""This implementation is adapted from:
# Author: [cloneofsimo]
# Repository: https://github.com/cloneofsimo/minRF
# License: Apache License, Version 2.0, January 2004 (http://www.apache.org/licenses/LICENSE-2.0)
# Modifications: [We proposed Feature-Guided Rectified Flow Model to reduce privacy leakage.
See more descriptions in our paper.]
"""
import torch
import os
import torch.optim as optim
from models.dit import DiT_Llama
from models.rf import RF


def parser_args():
    parser = argparse.ArgumentParser(description="Server image generation")
    parser.add_argument('--dataset', type=str, default='brain_tumor',
                        choices=['TB', 'skin_cancer', 'brain_tumor'],
                        help='Choose between "TB", "skin_cancer", "brain_tumor"')
    parser.add_argument('--num_classes', type=int, default=4, help="number of classes")
    parser.add_argument('--channels', type=int, default=1, help="number of channels")
    parser.add_argument('--image_size', type=int, default=128, help='image_size')
    # parser.add_argument('--up_bound', type=int, default=1100, help='up bound of generation loop')
    parser.add_argument('--cuda', type=int, default=3, help='choose cuda number')
    parser.add_argument('--specified', action='store_true',
                        help='If set, use specified model weights')
    parser.add_argument('--model_weights', type=str,
                        default='/mnt/lv01/myf/workspace/Tumor_project/gen_data/caption',
                        help='path of specified model weights')
    parser.add_argument('--client', type=int, default=0, help='client id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()

    torch.cuda.set_device(args.cuda)
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    if args.dataset == 'brain_tumor':
        dataset_name = "brain_tumor"
        channels = args.channels
        print(f'channels: {channels}')
        model = DiT_Llama(
            channels, args.image_size, dim=256, n_layers=10, n_heads=8, patch_size=8, num_classes=args.num_classes
        ).cuda()

    elif args.dataset == 'skin_cancer':
        dataset_name = "skin_cancer"
        channels = args.channels
        print(f'channels: {channels}')
        model = DiT_Llama(
            channels, args.image_size, dim=256, n_layers=10, n_heads=8, patch_size=8, num_classes=args.num_classes
        ).cuda()

    else:
        dataset_name = "TB"
        channels = args.channels
        print(f'channels: {channels}')
        model = DiT_Llama(
            channels, args.image_size, dim=256, n_layers=10, n_heads=8, patch_size=8, num_classes=args.num_classes
        ).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    # wandb.init(project=f"rf_{dataset_name}", mode="offline")

    # load model
    if args.specified:
        model_name = args.model_weights
    else:
        model_name = f"./generated_contents/models/{args.dataset}/client_{args.client}/ckpt_latest.pt"

    ckpt = torch.load(model_name)
    print('load', model_name)
    rf.model.load_state_dict(ckpt)

    ####2025.6
    ### The number of the generated images is tha same as the dataset of each client.
    if args.dataset == 'brain_tumor':
        if args.client == 0:
            gen_nums = [1023, 1011, 1038, 1030]  # brain client0
        elif args.client == 1:
            gen_nums = [1030, 1022, 1023, 1015]  # brain client1
        else:
            gen_nums = [1030, 1007, 1023, 1037]  # brain client2
    if args.dataset == 'TB':
        if args.client == 0:
            gen_nums = [1033, 1033]  # TB client0
        elif args.client ==1:
            gen_nums = [1008, 1008]  # TB client1
        else:
            gen_nums = [1008, 1006]  # TB client2
    if args.dataset == 'skin_cancer':
        if args.client == 0:
            gen_nums = [1025, 1004, 1042, 1004, 1016, 1251, 1007]  # skin client0
        elif args.client ==1:
            gen_nums = [1041, 1004, 1025, 1005, 1012, 1050, 1015]  # skin client1
        else:
            gen_nums = [1033, 1003, 1033, 1003, 1019, 1302, 1007]  # skin client2

    print(gen_nums)

    rf.model.eval()
    with torch.no_grad():
        # define the number of classes and samples per class
        num_classes = args.num_classes
        samples_per_class = 10

        for class_id in range(num_classes):
            print(f"Generating images for class {class_id}...")
            batch_data = []  # save all batch samples per class

            for i in range(1000, gen_nums[class_id]):
                # Create a condition variable for the current category.
                cond = torch.tensor([class_id] * samples_per_class).cuda()
                uncond = torch.ones_like(cond) * num_classes

                # initialize noise
                init_noise = torch.randn(samples_per_class, channels, args.image_size, args.image_size).cuda()

                # sample features
                images = rf.sample(init_noise, cond, uncond)

                # obtain the last one
                final_images = images[-1]
                batch_data.append(final_images.cpu())

            #     # Save the images of the current category (Save as image format)
            #     class_dir = (f"./generated_contents/generated_features/{args.dataset}"
            #                  f"/client_{args.client}/class_{class_id}")
            #     os.makedirs(class_dir, exist_ok=True)
            #     print(f"Saving images to: {class_dir}")
            #
            #     for idx, image in enumerate(final_images):
            #         # Inverse normalization
            #         image = image * 0.5 + 0.5
            #         image = image.clamp(0, 1)
            #
            #         img = image.permute(1, 2, 0).cpu().numpy()
            #         img = img.squeeze()
            #         img = (img * 255).astype(np.uint8)
            #
            #         # save images
            #         img = Image.fromarray(img)
            #         img.save(os.path.join(class_dir, f"{i}_image_{idx}.png"))
            # print(f"Finished generating and saving images for class {class_id}.")

            # Save all batch data for the current category. (tensor format)
            class_dir = f"./generated_contents/generated_features/{args.dataset}/client_{args.client}"
            os.makedirs(class_dir, exist_ok=True)

            save_path = os.path.join(class_dir, f"class_{class_id}_generated_data.pth")
            torch.save(batch_data, save_path)  # Save all batch data at once.
            print(f"Finished generating images for class {class_id}.")

    rf.model.train()
