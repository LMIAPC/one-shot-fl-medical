import os
from collections import defaultdict, Counter
import pandas as pd
from torch.utils.data import Dataset, Subset, TensorDataset, WeightedRandomSampler, random_split
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random
##
# define seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# create generator
generator = torch.Generator()
generator.manual_seed(seed)


# set random seed for each worker
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomDataset(Dataset):
    def __init__(self, data_dir, args, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.filepaths = []
        self.labels = []
        self.args = args

        # Define the allowed image formats
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

        for label, class_name in enumerate(self.classes):
            print(f'Processing class {class_name}:')
            class_dir = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if os.path.splitext(file_name)[1].lower() in self.allowed_extensions:
                    self.filepaths.append(file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        if self.args.channels == 1:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_label(data_dir, args, transform=None):
    print(data_dir)
    client_dirs = [d for d in os.listdir(data_dir) if
                   d.startswith("client") and os.path.isdir(os.path.join(data_dir, d))]
    all_images = []
    all_labels = []

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')  # Add more extensions if needed

    for i, client in enumerate(client_dirs):
        client_data_dir = os.path.join(data_dir, client)
        # classes_dir = os.listdir(client_data_dir)
        classes_dir = [d for d in os.listdir(client_data_dir) if os.path.isdir(os.path.join(client_data_dir, d))]
        images = []
        labels = []
        for label, class_name in enumerate(classes_dir):
            print(f'client {i}: {class_name}')
            class_dir = os.path.join(client_data_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if file_name.lower().endswith(valid_extensions):  # Filter non-image files
                    if args.channels == 1:
                        image = Image.open(file_path).convert('L')
                    else:
                        image = Image.open(file_path).convert('RGB')
                    if transform:
                        image = transform(image)
                    images.append(image)
                    labels.append(label)
        print(f'number of client_{i}: {len(labels)}')
        all_images.extend(images)
        all_labels.extend(labels)
    # print(f'total number of images: {len(all_images)}')
    return all_images, all_labels


def get_data_label_brain(data_dir, args, transform=None):
    """
    load brain tumor dataset
    :param data_dir: the training directory of brain tumor dataset
    :param args: the arguments
    :param transform: the transformation
    :return: images and labels
    """
    # classes_dir = os.listdir(data_dir)
    # classes_dir = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')  # Add more extensions if needed
    label_map={
        'pituitary_tumor': 0,
        'no_tumor': 1,
        'glioma_tumor': 2,
        'meningioma_tumor': 3
    }
    for class_name, label in label_map.items():
        print(class_name, label)
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if file_name.lower().endswith(valid_extensions):  # Filter non-image files
                if args.channels == 1:
                    image = Image.open(file_path).convert('L')
                else:
                    image = Image.open(file_path).convert('RGB')
                if transform:
                    image = transform(image)
                images.append(image)
                labels.append(label)
    print(f'Total images: {len(labels)}')
    return images, labels


# load all clients of TB dataset
def get_TB_data(data_dir, transform=None):
    print(data_dir)
    client_list = ['ChinaSet', 'IndiaSet', 'MontgomerySet']
    all_images = []
    all_labels = []

    for client in client_list:
        client_data_dir = os.path.join(data_dir, client)
        images, labels = get_TB_data_single(client_data_dir, transform)
        print(f'Client {client} has {len(images)}')
        all_images.extend(images)
        all_labels.extend(labels)

    print(f'Total images: {len(all_images)}')
    return all_images, all_labels


# load single client of TB dataset
def get_TB_data_single(data_dir, transform=None):
    images = []
    labels = []

    label_map = {
        'Normal': 0,
        'TB': 1
    }

    for class_name, label in label_map.items():
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            image = Image.open(file_path).convert('L')
            if transform:
                image = transform(image)
            images.append(image)
            labels.append(label)
    print(len(labels))
    return images, labels


def get_torch_data(data_dir="/mnt/lv02/myf/test/gen_data/model_feature/brain_tumor/client_0",
                   num_classes=4):
    """
    load generated features in "tensor format"!
    :param data_dir: the directory of generated features
    :param num_classes: the number of classes
    :return: images and labels
    """
    # For storing the combined data and labels.
    all_data = []
    all_labels = []

    for class_id in range(num_classes):
        file_path = os.path.join(data_dir, f"class_{class_id}_generated_data.pth")
        class_data = torch.load(file_path)

        all_data.extend(class_data)
        all_labels.extend([class_id] * len(class_data))

    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.tensor(all_labels)

    # print(f"Combined data shape: {all_data.shape}")
    # print(f"Combined labels shape: {all_labels.shape}")
    return all_data, all_labels


def load_csv_dataset(main_folder_path, transform=None):
    """
    read HAM dataset with csv format
    :param main_folder_path: the main folder path
    :param transform: the transformation
    :return: images and labels
    """
    images = []
    image_paths = []
    labels = []

    csv_path = os.path.join(main_folder_path, 'HAM10000_metadata.csv')

    metadata = pd.read_csv(csv_path)

    images_folder = os.path.join(main_folder_path, 'Skin_Cancer')

    label_mapping = {
        'bkl': 0,
        'df': 1,
        'mel': 2,
        'vasc': 3,
        'bcc': 4,
        'nv': 5,
        'akiec': 6
    }

    for idx, row in metadata.iterrows():
        # Obtain the image IDs along with the labels.
        image_name = row['image_id'] + '.jpg'
        label_name = row['dx']

        if label_name in label_mapping:
            label = label_mapping[label_name]
        else:
            print(f"Warning：Cannot find the mapping for label {label_name}. Skipping this image.")
            continue

        # obtain the path of images
        image_path = os.path.join(images_folder, image_name)

        image = Image.open(image_path).convert('RGB')
        if image is not None:
            if transform:
                image = transform(image)
                images.append(image)
                labels.append(label)
            else:
                images.append(image)
                labels.append(label)

    if len(images) == 0:
        print("Failed to load any images. Please check the file paths and formats.")

    return images, labels


def shard_non_iid_partition_new(labels, num_clients=10, shards_per_client=2, random_seed=42):
    """
    Apply the shards method to divide the dataset into Non-IID data for clients.
    :param random_seed:random seed
    :param labels: dataset labels
    :param dataset: PyTorch Dataset (with labels)
    :param num_clients: number of clients
    :param shards_per_client: The number of shards per client.
    :return: Dictionary of client data indices
    ! Insure num_shards≥num_classes
    """
    # set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
    # Get the indices of all the data.
    data_indices = np.arange(len(labels))
    labels = np.array([labels[i] for i in data_indices])  # Get the labels for all samples.

    # Group data indices by class.
    class_indices = defaultdict(list)
    for idx, label in zip(data_indices, labels):
        class_indices[label].append(idx)

    # Split the data of each class into shards.
    num_shards = num_clients * shards_per_client
    all_shards = []
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        shards = np.array_split(indices, num_shards // len(class_indices))
        all_shards.extend(shards)

    # Shuffle the shards and assign them to clients.
    np.random.shuffle(all_shards)
    client_data = {i: [] for i in range(num_clients)}
    for i, shard in enumerate(all_shards):
        client_id = i % num_clients
        client_data[client_id].extend(shard)

    print(f"The data has been successfully partitioned across {num_clients} clients, "
          f"with each client containing {shards_per_client} shards！")
    return client_data


def load_client_dataset(args, train_transform, test_transform):
    """
    load client datasets
    :param args: the arguments
    :param train_transform: training data transform
    :param test_transform:  testing data transform
    :return: training and test datasets of each client
    """
    # select datasets
    if args.dataset == 'brain_tumor':
        # load train dataset

        # change to your dataset path!
        data_folder = "/mnt/diskB/myf/datasets/brain_tumor/Training"

        images, labels = get_data_label_brain(data_folder, args, transform=train_transform)
        datasets = TensorDataset(torch.stack(images), torch.tensor(labels))
        if len(images) == 0:
            raise ValueError("No images were found. Please check the image file paths or folder structure.")

        # split dataset(non-iid)
        client_datas = shard_non_iid_partition_new(labels, num_clients=3, shards_per_client=15)
        # Construct a corresponding sub-dataset for each client.
        client_datasets = {}
        for client_id, indices in client_datas.items():
            client_datasets[client_id] = Subset(datasets, indices)

        ## load test dataset
        # change to your own path!
        test_folder = '/mnt/diskB/myf/datasets/brain_tumor/Testing'
        test_images, test_labels = get_data_label_brain(test_folder, args, transform=test_transform)
        test_datasets = TensorDataset(torch.stack(test_images), torch.tensor(test_labels))
        if len(test_images) == 0:
            raise ValueError("No images were found. Please check the image file paths or folder structure.")

        # Split the test set into three equal subsets.
        total_size = len(test_datasets)
        split_size = total_size // 3  # The size of each subset

        # Calculate the size of each subset
        # If the total number cannot be evenly divided by 3, the last subset will be slightly larger.
        split_sizes = [split_size] * 3
        split_sizes[-1] += total_size - sum(split_sizes)  # The remainder is added to the last subset

        # set random seed
        generator_split = torch.Generator().manual_seed(42)
        # use random_split to split dataset
        subset1, subset2, subset3 = random_split(test_datasets, split_sizes, generator_split)
        test_sets = [subset1, subset2, subset3]

        return client_datasets, test_sets

    if args.dataset == "skin_cancer":
        # load train dataset

        # change to your dataset path!
        data_folder = "/mnt/diskB/myf/datasets/Skin-Cancer"
        images, labels = load_csv_dataset(data_folder, transform=train_transform)

        # Count the number of categories
        label_counts = Counter(labels)
        print(f'Information for H10000 Dataset: {label_counts}')
        datasets = TensorDataset(torch.stack(images), torch.tensor(labels))
        if len(images) == 0:
            raise ValueError("No images were found. Please check the image file paths or folder structure.")

        # split datasets
        testsets = []
        client_datas = shard_non_iid_partition_new(labels, num_clients=3, shards_per_client=30)
        # Construct corresponding subsets for each client
        client_datasets = {}
        for client_id, indices in client_datas.items():
            # Get the current client's data subset.
            client_subset = Subset(datasets, indices)

            # Get all tensors.
            all_tensors = datasets.tensors

            # The first tensor is the features, and the second tensor is the labels.
            features_tensor = all_tensors[0]
            labels_tensor = all_tensors[1]

            # Get the labels for the corresponding client.
            subset_labels = labels_tensor[indices]

            #
            label_counts = Counter(subset_labels.numpy())
            sorted_label_counts = dict(sorted(label_counts.items()))

            # print(f"Client {client_id}: Class distribution before splitting: {dict(label_counts)}")
            # print(f"Client {client_id}: Class distribution before splitting: {sorted_label_counts}")

            # Split the training and validation sets using stratified sampling.
            min_count = min(label_counts.values())  # Find the size of the smallest category
            # Calculate a reasonable test_size to prevent any classes from disappearing.
            test_size = max(0.1, 1 / min_count)

            train_indices, val_indices = train_test_split(
                indices,  # Data Index
                # range(len(indices)),
                test_size=test_size,  # Test set ratio
                stratify=subset_labels.numpy(),  # Stratify by labels.
                # shuffle=True,
                random_state=42  # Use a random seed to ensure reproducibility.
            )

            # create train dataset and test dataset
            train_dataset = Subset(datasets, list(train_indices))
            val_dataset = Subset(datasets, list(val_indices))

            # define Transform
            # train_transform = transforms.Compose([
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomRotation(10),
            #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # ])

            # transformed_dataset = TransformTensorDataset(train_dataset, train_transform)

            # save train dataset and test dataset
            client_datasets[client_id] = train_dataset
            testsets.append(val_dataset)

        return client_datasets, testsets

    if args.dataset == "TB":
        # load original data
        # change to your dataset path!
        image0, labels0 = get_TB_data_single(
            "/mnt/diskB/myf/datasets/TB_dataset/ChinaSet",
            transform=train_transform)
        image1, labels1 = get_TB_data_single(
            "/mnt/diskB/myf/datasets/TB_dataset/IndiaSet",
            transform=train_transform)
        image2, labels2 = get_TB_data_single(
            "/mnt/diskB/myf/datasets/TB_dataset/MontgomerySet",
            transform=train_transform)
        # client0
        X_train0, X_test0, y_train0, y_test0 = train_test_split(image0, labels0, test_size=0.1,
                                                                random_state=42, stratify=labels0)
        # client1
        X_train1, X_test1, y_train1, y_test1 = train_test_split(image1, labels1, test_size=0.1,
                                                                random_state=42, stratify=labels1)
        # client2
        X_train2, X_test2, y_train2, y_test2 = train_test_split(image2, labels2, test_size=0.1,
                                                                random_state=42, stratify=labels2)

        client0_dataset = TensorDataset(torch.stack(X_train0), torch.tensor(y_train0))
        client1_dataset = TensorDataset(torch.stack(X_train1), torch.tensor(y_train1))
        client2_dataset = TensorDataset(torch.stack(X_train2), torch.tensor(y_train2))

        client_datasets = [client0_dataset, client1_dataset, client2_dataset]

        test_dataset0 = TensorDataset(torch.stack(X_test0), torch.tensor(y_test0))
        # test_loader0 = DataLoader(test_dataset0, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_dataset1 = TensorDataset(torch.stack(X_test1), torch.tensor(y_test1))
        # test_loader1 = DataLoader(test_dataset1, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_dataset2 = TensorDataset(torch.stack(X_test2), torch.tensor(y_test2))
        # test_loader2 = DataLoader(test_dataset2, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # test_loader = [test_loader0, test_loader1, test_loader2]
        test_sets = [test_dataset0, test_dataset1, test_dataset2]

        return client_datasets, test_sets


def load_auxiliary_datasets(args):
    """
    if generated features are stored in pictures format, then using this function to load training datasets for
    knowledge distillation. In our experiments, the generated brain tumor features are stored in pictures format,
    which is used in subsequent knowledge distillation.
    :param args:
    :return: train dataloader
    """
    if args.dataset == 'brain_tumor':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # train_transform = transforms.Compose([
        #     # transforms.RandomResizedCrop(size=224),
        #     # transforms.Resize((224, 224)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.RandomRotation(degrees=15),
        #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.ToTensor(),  # 转换为Tensor
        #     transforms.Normalize(mean=[0.5],
        #                          std=[0.5])
        # ])
        # test_transform = transforms.Compose([
        #     transforms.Resize(size=256),
        #     transforms.CenterCrop(size=224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5],
        #                          std=[0.5])
        # ])

        images, labels = get_data_label(
            # the path of generated features
            '/mnt/diskB/myf/workspace/one-shot/gen_dataset/gen_data_224/brain_tumor',
            args,
            transform=transform)

        auxiliary_dataset = TensorDataset(torch.stack(images), torch.tensor(labels))

        labels = torch.tensor(labels)

        # Calculate the number of samples per class.
        label_counts = torch.bincount(labels)
        class_weights = 1.0 / label_counts.float()
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True)

        aux_loader = DataLoader(auxiliary_dataset,
                                batch_size=args.batch_size,
                                sampler=sampler,
                                num_workers=4,
                                worker_init_fn=seed_worker,
                                generator=generator
                                )

    if args.dataset == "skin_cancer":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # train_transform = transforms.Compose([
        #     # transforms.RandomResizedCrop(size=224),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.RandomRotation(degrees=15),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5],
        #                          std=[0.5])
        # ])
        # test_transform = transforms.Compose([
        #     transforms.Resize(size=256),
        #     transforms.CenterCrop(size=224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5],
        #                          std=[0.5])
        # ])
        images, labels = get_data_label(
            # the path of generated features
            '/mnt/diskB/myf/workspace/one-shot/gen_dataset/gen_data_224/skin_cancer',
            args,
            transform=transform)
        auxiliary_dataset = TensorDataset(torch.stack(images), torch.tensor(labels))

        labels = torch.tensor(labels)

        # Calculate the number of samples per class.
        label_counts = torch.bincount(labels)
        class_weights = 1.0 / label_counts.float()
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True)

        aux_loader = DataLoader(auxiliary_dataset,
                                batch_size=args.batch_size,
                                sampler=sampler,
                                num_workers=4,
                                worker_init_fn=seed_worker,
                                generator=generator
                                )
    if args.dataset == 'TB':
        # train_transform = transforms.Compose([
        #     # transforms.RandomResizedCrop(size=224),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomRotation(degrees=15),
        #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5],
        #                          std=[0.5])
        # ])
        # test_transform = transforms.Compose([
        #     transforms.Resize(size=256),
        #     transforms.CenterCrop(size=224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5],
        #                          std=[0.5])
        # ])

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        images, labels = get_data_label(
            # the path of generated features
            '/mnt/diskB/myf/workspace/one-shot/gen_dataset/gen_data_224/TB',
            args,
            transform=transform)
        auxiliary_dataset = TensorDataset(torch.stack(images), torch.tensor(labels))

        labels = torch.tensor(labels)

        # Calculate the number of samples per class.
        label_counts = torch.bincount(labels)
        class_weights = 1.0 / label_counts.float()
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

        aux_loader = DataLoader(auxiliary_dataset,
                                batch_size=args.batch_size,
                                sampler=sampler,
                                num_workers=4,
                                worker_init_fn=seed_worker,
                                generator=generator
                                )
    return aux_loader
