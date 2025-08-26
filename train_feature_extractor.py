import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
import argparse

from helpers.dataset_2025 import load_client_dataset
from models.feature_extractor import resnet18_new
from helpers.utils import load_model, test
import wandb


# You can comment this
if torch.cuda.is_available():
    print(f"CUDA is available. Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(torch.cuda.current_device())  # Display the index of the currently used GPU
else:
    print("CUDA is not available.")


def parse_args():
    parser = argparse.ArgumentParser(description='Feature extractor training for one-shot federated classification')
    parser.add_argument('--dataset', type=str, default='brain_tumor',
                        choices=['TB', 'skin_cancer', 'brain_tumor'],
                        help='Choose between "TB", "skin_cancer", "brain_tumor"')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization) (default: 1e-5)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--num_classes', type=int, default=4, help="number of classes")
    parser.add_argument('--num_users', type=int, default=3, help="number of users: K")
    parser.add_argument('--channels', type=int, default=1, help="number of picture channels")
    parser.add_argument('--seed', type=int, default=66, help="seed")
    parser.add_argument('--cuda', type=int, default=1, help="cuda device number")
    parser.add_argument('--dis_epoch', type=int, default=100, help="number of epochs for distillation")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha for distillation")
    parser.add_argument('--dis_beta', type=float, default=0.5, help="beta for distillation")
    parser.add_argument('--T', type=int, default=3, help="temperature for distillation")
    parser.add_argument('--label_s', type=float, default=0.1, help="factor of label smoothing")
    parser.add_argument('--wandb_file', type=str, default='one-shot', help='Log file name',
                        required=False)
    args = parser.parse_args()
    return args


def kd_train_extractor(model, client_datasets, test_sets, epochs, save_path, num_users, args):
    """
    This function is used for training local feature extractor, moreover; it can also be used in kd if teacher models
    is trained by pictures
    :param model: extractor
    :param train_loader: train loader
    :param test_loader: test loader
    :param epochs: number of training epochs
    :param save_path: path to save models
    :param num_users: number of users
    :param args: args
    :return:
    """
    # create client loaders
    client_loaders = []
    for i in range(len(client_datasets)):
        # labels = client_datasets[i].labels # when using Custom dataset class
        # labels = torch.tensor(labels)

        if args.dataset == 'TB':
            labels = client_datasets[i].tensors[1]
        else:
            indices = client_datasets[i].indices
            original_dataset = client_datasets[i].dataset
            while isinstance(original_dataset, Subset):
                original_dataset = original_dataset.dataset

            labels = original_dataset.tensors[1][indices]

        label_counts = torch.bincount(labels)
        class_weights = 1.0 / label_counts.float()
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        client_loader = DataLoader(client_datasets[i],
                                   batch_size=args.batch_size,
                                   sampler=sampler,
                                   num_workers=4,
                                   worker_init_fn=seed_worker,
                                   generator=generator
                                   )
        client_loaders.append(client_loader)

        # data loader_v2
        # client_loaders.append(DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True, num_workers=4))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # clients train locally
    for i in range(num_users):

        # define features save path
        metadata_save_path = os.path.join(save_path, f'{args.dataset}_client_{i}_metadata.pth')

        for epoch in range(epochs):
            # model training
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(client_loaders[i]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(client_loaders[i].dataset)
            epoch_acc = running_corrects.double() / len(client_loaders[i].dataset)

            # val_loss, val_acc = validation(model, val_dataset, val_loader, criterion)

            print(f'Client {i}: Epoch {epoch}/{epochs - 1}')
            print(f'Client {i} Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')
            # print(f'Client {i} Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}')

            # wandb.log({'Client': i, 'Train Loss': epoch_loss, 'Train Accuracy': epoch_acc})
            # wandb.log({'Client': i, 'Val Loss': val_loss, 'Val Accuracy': val_acc})

            # save model
            if epoch == epochs - 1:
                torch.save(model.state_dict(), os.path.join(save_path, f'{args.dataset}_client_{i}_local_model.pth'))
                save_test_dir = os.path.join(save_path, f'{args.dataset}_client_{i}_local_model.pth')

        # Extract features!
        print(f"Extracting features for Client {i}...")
        model.eval()
        features_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in client_loaders[i]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                _, features, _ = model(inputs, return_features=True)
                # _, features = model(inputs, return_features=True)

                features_list.append(features.cpu())
                labels_list.append(labels.cpu())

        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        metadata = {
            "features": all_features,
            "labels": all_labels
        }

        torch.save(metadata, metadata_save_path)

        print(f"Metadata for Client {i} saved at {metadata_save_path}.")

        test_loader = DataLoader(test_sets[i], shuffle=False, batch_size=args.batch_size, num_workers=4)
        # test model
        model_test = load_model(model, save_test_dir)
        accu = test(model_test, test_loader)

        print(f'Client {i} Test Accuracy: {accu * 100:.2f}%')
        # wandb.log({'Client': i, 'Test Accuracy': accu * 100})



if __name__ == '__main__':
    args = parse_args()
    # torch.cuda.set_device(args.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="one-shot-classification",
        name=args.wandb_file,
        mode="offline"
    )

    # define seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # create generator
    generator = torch.Generator()
    generator.manual_seed(seed)


    # Set a random seed for each worker
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    # ************************************************************************ #

    extractor = resnet18_new(args.num_classes, in_channels=args.channels)
    extractor = extractor.to(device)

    # define model saved paths
    ### !
    save_dir = "./saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ## pre-train feature extractor, using model resnet18_new from models.extractor
    # load dataset
    if args.dataset == 'skin_cancer':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    client_datasets, test_sets = load_client_dataset(args, transform, transform)

    # train feature extractor
    kd_train_extractor(extractor, client_datasets, test_sets, args.num_epochs, save_dir, args.num_users, args)
    print("Local Extractor Training Finish!")

