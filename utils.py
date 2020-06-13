# -*- coding: utf-8 -*-

import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as pbar
import copy


# CIFAR10 mean and std for normalization
mean_cifar10, std_cifar10 = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Returns a DataLoader of CIFAR10 test set
def testloader_cifar10(path, batch_size, shuffle=True):
  transform = transforms.Compose([transforms.ToTensor(),
      transforms.Normalize(mean=mean_cifar10, std=std_cifar10)])

  test_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10(root=path, train=False, transform=transform, download=True),
      batch_size=batch_size, shuffle=shuffle
  )

  return test_loader


# Shows the accuracy of the model
def test_model(model, device, test_loader):

    model = model.to(device).eval()
    logs = {'Accuracy': 0.0}

    # Iterate over data
    for image, label in pbar(test_loader):
        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            prediction = model(image)
            accuracy = torch.sum(torch.max(prediction, 1)[1] == label.data).item()
            logs['Accuracy'] += accuracy

    logs['Accuracy'] /= len(test_loader.dataset)

    return logs['Accuracy']


def normalize(img, dataset='cifar10'):  # img of size (3,H,W)
  mean = mean_cifar10 if dataset=='cifar10' else [0,0,0]
  std = std_cifar10 if dataset=='cifar10' else [1,1,1]
  for channel in range(3):
    img[channel] = (img[channel] - mean[channel]) / std[channel]
  return img


min_rgb = {'cifar10': normalize(torch.tensor([0.,0.,0.], dtype=torch.double), dataset='cifar10'), 'imagenet': normalize(torch.tensor([0.,0.,0.], dtype=torch.double), dataset='imagenet')}
max_rgb = {'cifar10': normalize(torch.tensor([1.,1.,1.], dtype=torch.double), dataset='cifar10'), 'imagenet': normalize(torch.tensor([1.,1.,1.], dtype=torch.double), dataset='imagenet')}


def denormalize(img, dataset='cifar10'):  # img of size (3,H,W)
  mean = mean_cifar10 if dataset=='cifar10' else [0,0,0]
  std = std_cifar10 if dataset=='cifar10' else [1,1,1]
  for channel in range(3):
    img[channel] = img[channel] * std[channel] + mean[channel]
  return img


def de_scale(x, dataset='cifar10'):  # x: tensor of size 1xCxHxW
  std = std_cifar10 if dataset=='cifar10' else [1,1,1]
  x[0][0] *= std[0]
  x[0][1] *= std[1]
  x[0][2] *= std[2]
  return x


# Keep tensor values between two tensors values
def clip_image_values(x, minv, maxv):
    x = torch.max(x, minv)
    x = torch.min(x, maxv)
    return x


# Clamp normalized image values in the desired range (no-normalized)
def clamp(img, inf, sup, dataset='cifar10'):
  mean = mean_cifar10 if dataset=='cifar10' else [0,0,0]
  std = std_cifar10 if dataset=='cifar10' else [1,1,1]
  for channel in range(3):
    lim_inf = (inf-mean[channel]) / std[channel]
    lim_sup = (sup-mean[channel]) / std[channel]
    img[0][channel] = clip_image_values(img[0][channel], torch.full_like(img[0][channel], lim_inf), torch.full_like(img[0][channel], lim_sup))
  return img


# For SparseFool
def valid_bounds(img, delta=255, dataset='cifar10'):
  mean = mean_cifar10 if dataset=='cifar10' else [0,0,0]
  std = std_cifar10 if dataset=='cifar10' else [1,1,1]
  # Deepcopy of the image as a numpy int array of range [0, 255]
  im = copy.deepcopy(np.transpose(denormalize(img.cpu().detach().numpy()[0], dataset=dataset), (1,2,0)))
  im *= 255
  im = (np.around(im)).astype(np.int)

  # General valid bounds [0, 255]
  valid_lb = np.zeros_like(im)
  valid_ub = np.full_like(im, 255)

  # Compute the bounds
  lb = im - delta
  ub = im + delta

  # Validate that the bounds are in [0, 255]
  lb = np.maximum(valid_lb, np.minimum(lb, im))
  ub = np.minimum(valid_ub, np.maximum(ub, im))

  # Round and change types to uint8
  lb = lb.astype(np.uint8)
  ub = ub.astype(np.uint8)

  # Convert to tensors and normalize
  lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
  ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)

  return lb, ub


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


# Convert a tensor to a plt displayable numpy array in range [0,1]
def displayable(img, dataset='cifar10'):  # tensor of size 1xCxHxW
  return np.transpose(denormalize(img.squeeze().numpy(), dataset=dataset), (1,2,0))
