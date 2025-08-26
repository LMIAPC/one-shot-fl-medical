import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms
import argparse

from helpers.dataset_2025 import load_client_dataset, load_auxiliary_datasets, get_torch_data
from models.resnet import resnet18
from models.feature_extractor import resnet18_new
from helpers.utils import (load_model, aggregate_models, kl_loss, apply_label_smoothing, validation,
                           count_data_and_labels, print_client_data_statistics, print_client_data_statistics_test)
import wandb


if torch.cuda.is_available():
    print(f"CUDA is available. Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(torch.cuda.current_device())  # 显示当前使用的 GPU 索引
else:
    print("CUDA is not available.")


def parse_args():
    parser = argparse.ArgumentParser(description='Feature extractor training for one-shot federated classification')
    parser.add_argument('--train_type', type=str, default='student',
                        choices=['student', 'teacher'],
                        help="choose two stages of knowledge distillation: teacher or student train")
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
    parser.add_argument('--seed', type=int, default=66, help="brain: 66, others: 1228")
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


def kd_train_local(model, test_sets, epochs, save_path, num_users, args):
    """
    This function is used for locally training (teacher training) for knowledge distillation
    :param model: Client local models, e.g. Resnet18
    :param test_loader: test loader
    :param epochs: number of epochs
    :param save_path: path to save model
    :param num_users: number of users
    :param args: args
    :return:
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # clients train locally (teacher model training)
    for i in range(num_users):
        ## train features
        # load features
        metadata_save_path = os.path.join(save_path, f'{args.dataset}_client_{i}_metadata.pth')

        metadata = torch.load(metadata_save_path)
        features = metadata["features"]  # [32, 64, 224, 224]
        labels = metadata["labels"]

        # tensor_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.RandomVerticalFlip(),
        #     transforms.RandomRotation(15),
        #     # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        #     # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),  # 随机擦除
        # ])
        #
        # augmented_tensor = tensor_transforms(features)

        train_dataset = TensorDataset(features, labels)

        label_counts = torch.bincount(labels)
        class_weights = 1.0 / label_counts.float()
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        client_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   sampler=sampler,
                                   num_workers=4,
                                   worker_init_fn=seed_worker,
                                   generator=generator
                                   )

        # ### if train pictures
        # labels = client_datasets[i].tensors[1] # TB dataset

        # indices = client_datasets[i].indices  # Subset 存储的是索引
        # original_dataset = client_datasets[i].dataset
        # while isinstance(original_dataset, Subset):  # 处理嵌套 Subset
        #     original_dataset = original_dataset.dataset
        #
        # labels = original_dataset.tensors[1][indices] # brain/HAM10000 dataset

        # labels = torch.tensor(labels)
        # # 计算每个类别的样本数量
        # label_counts = torch.bincount(labels)
        # class_weights = 1.0 / label_counts.float()
        # sample_weights = class_weights[labels]
        #
        # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        #
        # client_loader = DataLoader(client_datasets[i],
        #                            batch_size=args.batch_size,
        #                            sampler=sampler,
        #                            num_workers=4,
        #                            worker_init_fn=seed_worker,  # 初始化 worker 随机性
        #                            generator=generator  # 控制随机性
        #                            )

        print(f'client {i} starts teacher model training!')
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(client_loader):
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

            epoch_loss = running_loss / len(client_loader.dataset)
            epoch_acc = running_corrects.double() / len(client_loader.dataset)

            # val_loss, val_acc = validation(model, val_dataset, val_loader, criterion)

            print(f'Client {i}: Epoch {epoch}/{epochs - 1}')
            print(f'Client {i} Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')
            # print(f'Client {i} Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}')

            # wandb.log({'Client': i, 'Train Loss': epoch_loss, 'Train Accuracy': epoch_acc})
            # wandb.log({'Client': i, 'Val Loss': val_loss, 'Val Accuracy': val_acc})

            if epoch == epochs - 1:
                torch.save(model.state_dict(),
                           os.path.join(save_path, f'{args.dataset}_client_{i}_local_model_teacher.pth'))
                save_test_dir = os.path.join(save_path, f'{args.dataset}_client_{i}_local_model_teacher.pth')

        # # test model
        # model_test = load_model(model, save_test_dir)
        # accu = test(model_test, test_loader)
        #
        # print(f'Client {i} Test Accuracy: {accu * 100:.2f}%')
        # wandb.log({'Client': i, 'Test Accuracy': accu * 100})


def distill_student_model(classifier, extractor, save_path, args, test_sets, num_epochs=20,
                          alpha=0.5, temperature=3, label_smoothing=0.1, beta=0.5):
    """
    Dual-Layer knowledge distillation in the server side (train student model).
    :param test_sets: test dataset
    :param classifier: Resnet-18
    :param extractor: feature extractor
    :param save_path: path to save model
    :param args: args
    :param num_epochs: number of distillation epochs
    :param alpha: 0.5
    :param temperature: 3
    :param label_smoothing: 0.1
    :param beta: 0.5
    :return:
    """
    # load each pre-trained client model
    client_models = []
    for i in range(args.num_users):
        classifier.load_state_dict(
            torch.load(os.path.join(save_path,
                                    f'{args.dataset}_client_{i}_local_model_teacher.pth')))  # teacher: feature
        classifier.eval()
        client_models.append(classifier)

    # aggregate models
    teacher_model, teacher_state_dict = aggregate_models(client_models, classifier)

    print("knowledge distillation")

    # define student model
    student_model = classifier

    # student_model.load_state_dict(teacher_state_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(student_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.AdamW(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(student_model.parameters(), lr=0.0001, alpha=0.99, weight_decay=args.weight_decay)

    feature_loss_fn = nn.MSELoss()

    ######
    # load synthetic features
    all_images = []
    all_labels = []
    for i in range(args.num_users):
        data_dir = f"./generated_contents/generated_features/{args.dataset}/client_{i}"
        images, labels = get_torch_data(data_dir=data_dir, num_classes=args.num_classes)
        # tensor_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     # transforms.RandomVerticalFlip(),
        #     transforms.RandomRotation(degrees=15), #20
        #     # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        #     # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        # ])
        # # augmented_tensor = tensor_transforms(images)
        # # all_images.extend(augmented_tensor)
        all_images.extend(images)
        print(f'client {i} has {images.shape[0]}')

        all_labels.extend([label for label in labels for _ in range(10)])
        # for label in labels:
        #     all_labels.extend([label] * 10)

    print(f'Total number of images: {len(all_images)}')
    dataset = TensorDataset(torch.stack(all_images), torch.tensor(all_labels))
    all_labels = torch.tensor(all_labels)

    # dataloader with WeightedRandomSampler
    label_counts = torch.bincount(all_labels)
    class_weights = 1.0 / label_counts.float()
    sample_weights = class_weights[all_labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    auxiliary_loader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=sampler,
                                  num_workers=4,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker,  #
                                  generator=generator
                                  )

    # auxiliary_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    ######

    # train student model with picture format features
    # auxiliary_loader = load_auxiliary_datasets(args)

    print(len(auxiliary_loader))

    # training
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in auxiliary_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # obtain the prediction of the teacher model
            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(images, return_features=True)

            # student outputs
            student_logits_distill, student_features = student_model(images, return_features=True)

            # add label smoothing
            smoothed_labels = apply_label_smoothing(labels, num_classes=args.num_classes, smoothing=label_smoothing)
            smoothed_labels = smoothed_labels.to(device)

            # obtain middel-layer features
            teacher_mid_features = teacher_features[-1].detach()
            student_mid_features = student_features[-1]

            feature_loss = feature_loss_fn(student_mid_features, teacher_mid_features)
            loss = kl_loss(teacher_logits, student_logits_distill, smoothed_labels, criterion, temperature, alpha)

            total_loss = loss + beta * feature_loss

            _, preds = torch.max(student_logits_distill, 1)

            # loss.backward()
            total_loss.backward()
            optimizer.step()

            # running_loss += loss.item() * images.size(0)
            # running_corrects += torch.sum(preds == labels.data)
            running_loss += total_loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(auxiliary_loader.dataset)
        epoch_acc = running_corrects.double() / len(auxiliary_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # wandb.log({'Epoch': epoch + 1, 'Loss': epoch_loss, 'Accuracy': epoch_acc})

        # save the model
        if epoch == num_epochs - 1:
            print('Saving model...')
            torch.save(student_model.state_dict(), os.path.join(save_path, f'{args.dataset}_student_model.pth'))

    # final test!
    for i in range(args.num_users):
        # The name for extractor model weights doesn't include 'fea'
        extractor.load_state_dict(torch.load(os.path.join(save_path, f'{args.dataset}_client_{i}_local_model.pth')))
        extractor.eval()

        student_model.load_state_dict(torch.load(os.path.join(save_path, f'{args.dataset}_student_model.pth')))
        student_model.eval()

        test_dataset = test_sets[i]
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)

                _, features, _ = extractor(data, return_features=True)
                logits = student_model(features)

                _, predicted = torch.max(logits, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if total == 0:
            return 0.0
        accuracy = (correct / total) * 100

        print(f'client {i} test accuracy: {accuracy:.2f}')


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


    # *************************************************** #
    extractor = resnet18_new(args.num_classes, in_channels=args.channels)
    extractor = extractor.to(device)

    save_dir = "./saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # using Resnet 18 as classifier
    classifier = resnet18(args.num_classes, in_channels=args.channels)
    classifier = classifier.to(device)

    ########
    if args.train_type == 'teacher':
        # client train local models using features for the following knowledge distillation
        # load dataset
        if args.dataset == 'skin_cancer':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像大小以匹配模型输入,原来是224
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像大小以匹配模型输入,原来是224
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(0.5, 0.5)
            ])
        _, test_sets = load_client_dataset(args, transform, transform)
        kd_train_local(classifier, test_sets, args.dis_epoch, save_dir, args.num_users, args)
        print("Local Training Finish!")

    if args.train_type == 'student':
        # knowledge distillation (student model)
        if args.dataset == 'skin_cancer':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像大小以匹配模型输入,原来是224
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(0.5, 0.5)
            ])

        _, test_sets = load_client_dataset(args, transform, transform)
        distill_student_model(classifier, extractor, save_dir, args, test_sets,
                              num_epochs=args.dis_epoch,
                              alpha=args.alpha, beta=args.dis_beta,
                              temperature=args.T,
                              label_smoothing=args.label_s)

