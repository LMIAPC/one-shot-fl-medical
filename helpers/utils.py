import os
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, Subset
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def averaging(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def train_local_model(model, train_loader, criterion, optimizer, epochs=1):
    # best_model_wts = None
    # best_val_loss = float('inf')

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # # Check if this is the best model so far
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model_wts = model.state_dict()

        print(f"Epoch {epoch + 1}/{epochs}: Loss: {epoch_loss}")

    return model.state_dict()

def load_model(model, load_path):
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")
    else:
        print(f"No model found at {load_path}")
    # model.eval()
    return model

def aggregate_models(client_models, model):
    teacher_model = model
    teacher_state_dict = teacher_model.state_dict()

    for key in teacher_state_dict.keys():
        teacher_state_dict[key] = torch.mean(
            torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))]), dim=0)

    teacher_model.load_state_dict(teacher_state_dict)
    # return teacher_model
    return teacher_model, teacher_state_dict #new


def validation(model, val_loader, criterion):
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = val_running_corrects.double() / len(val_loader.dataset)
    return val_loss, val_acc


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    if total == 0:
        return 0.0
    return correct / total


def kl_loss(teacher_logits, student_logits, labels, criterion, temperature, alpha):
    # distillation loss
    distill_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                            F.softmax(teacher_logits / temperature, dim=1), reduction='batchmean') * (
                           temperature ** 2)
    # classification loss
    classification_loss = criterion(student_logits, labels)

    loss = alpha * distill_loss + (1.0 - alpha) * classification_loss

    return loss


def apply_label_smoothing(labels, num_classes, smoothing=0.1):
    """
    Apply label smoothing to target labels.
    :param labels: Tensor of target labels, shape (batch_size,)
    :param num_classes: Number of classes in the dataset.
    :param smoothing: Smoothing factor, between 0 and 1.
    :return: Smoothed labels, shape (batch_size, num_classes)
    """
    with torch.no_grad():
        # Create a one-hot encoding of labels
        one_hot = torch.zeros((labels.size(0), num_classes), device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        # Apply label smoothing
        smoothed_labels = one_hot * (1 - smoothing) + (smoothing / num_classes)

    return smoothed_labels


def count_data_and_labels(client_data):
    client_stats = {}

    for client_id, dataset in client_data.items():
        if isinstance(dataset, Subset):
            indices = dataset.indices

            original_dataset = dataset.dataset
            while isinstance(original_dataset, Subset):
                original_dataset = original_dataset.dataset

            if isinstance(original_dataset, TensorDataset):
                data = original_dataset.tensors[0][indices]
                labels = original_dataset.tensors[1][indices]
            else:
                raise TypeError(f"Unexpected dataset type: {type(original_dataset)}")

        elif isinstance(dataset, TensorDataset):
            data = dataset.tensors[0]
            labels = dataset.tensors[1]

        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

        total_data = len(data)
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label.item()] += 1

        client_stats[client_id] = {
            'total_data': total_data,
            'label_counts': dict(sorted(label_counts.items()))
        }

    return client_stats


def print_client_data_statistics(client_datasets):
    for client_id, dataset in client_datasets.items():
        if isinstance(dataset, Subset):
            indices = dataset.indices

            original_dataset = dataset.dataset
            while isinstance(original_dataset, Subset):
                original_dataset = original_dataset.dataset

            if isinstance(original_dataset, TensorDataset):
                data = original_dataset.tensors[0][indices]
                labels = original_dataset.tensors[1][indices]
            else:
                raise TypeError(f"Unexpected dataset type: {type(original_dataset)}")

        elif isinstance(dataset, TensorDataset):
            data = dataset.tensors[0]
            labels = dataset.tensors[1]

        label_counts = Counter(labels.numpy())

        print(f"Client {client_id}:")
        print(f"  Total images: {len(labels)}")
        # print(f"  Class distribution: {dict(label_counts)}")
        print(f"  Class distribution: {dict(sorted(label_counts.items()))}")
        print("---------------------------")


def print_client_data_statistics_test(client_datasets):
    for client_id, dataset in enumerate(client_datasets):
        if isinstance(dataset, Subset):
            indices = dataset.indices

            original_dataset = dataset.dataset
            while isinstance(original_dataset, Subset):
                original_dataset = original_dataset.dataset

            if isinstance(original_dataset, TensorDataset):
                data = original_dataset.tensors[0][indices]
                labels = original_dataset.tensors[1][indices]
            else:
                raise TypeError(f"Unexpected dataset type: {type(original_dataset)}")

        elif isinstance(dataset, TensorDataset):
            data = dataset.tensors[0]
            labels = dataset.tensors[1]

        label_counts = Counter(labels.numpy())

        print(f"Client {client_id}:")
        print(f"  Total images: {len(labels)}")
        # print(f"  Class distribution: {dict(label_counts)}")
        print(f"  Class distribution: {dict(sorted(label_counts.items()))}")
        print("---------------------------")
