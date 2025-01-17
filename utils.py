# -*- coding: utf-8 -*-

import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as pbar
import copy
from PIL import Image
import requests


# CIFAR10 mean and std
mean_cifar10, std_cifar10 = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
mean_ImageNet, std_ImageNet = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# ImageNet mean and std
mean_ImageNet, std_ImageNet = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# CIFAR10 classes
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


# Shows the accuracy of a model
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


# Normalize an image using the corresponding mean and std
def normalize(img, dataset='cifar10'):  # img of size (3,H,W)
  mean = mean_cifar10 if dataset=='cifar10' else mean_ImageNet
  std = std_cifar10 if dataset=='cifar10' else std_ImageNet
  for channel in range(3):
    img[channel] = (img[channel] - mean[channel]) / std[channel]
  return img

# Denormalize an image to obtain the original pixel values in the range [0,1]
def denormalize(img, dataset='cifar10'):  # img of size (3,H,W)
  mean = mean_cifar10 if dataset=='cifar10' else mean_ImageNet
  std = std_cifar10 if dataset=='cifar10' else std_ImageNet
  for channel in range(3):
    img[channel] = img[channel] * std[channel] + mean[channel]
  return img

# De-scale an image using the std of the dataset
def de_scale(x, dataset='cifar10'):  # x: tensor of size 1xCxHxW
  std = std_cifar10 if dataset=='cifar10' else std_ImageNet
  x[0][0] *= std[0]
  x[0][1] *= std[1]
  x[0][2] *= std[2]
  return x

# Min and max RGB values of a pixel normalized with CIFAR10 or ImageNet mean and std
min_rgb = {'cifar10': normalize(torch.tensor([0.,0.,0.], dtype=torch.double), dataset='cifar10'), 'imagenet': normalize(torch.tensor([0.,0.,0.], dtype=torch.double), dataset='imagenet')}
max_rgb = {'cifar10': normalize(torch.tensor([1.,1.,1.], dtype=torch.double), dataset='cifar10'), 'imagenet': normalize(torch.tensor([1.,1.,1.], dtype=torch.double), dataset='imagenet')}



# Keep tensor values between two tensors values
def clip_image_values(x, minv, maxv):
    x = torch.max(x, minv)
    x = torch.min(x, maxv)
    return x


# Clamp normalized image values in the desired range
def clamp(img, inf, sup, dataset='cifar10'):
  mean = mean_cifar10 if dataset=='cifar10' else mean_ImageNet
  std = std_cifar10 if dataset=='cifar10' else std_ImageNet
  for channel in range(3):
    lim_inf = (inf-mean[channel]) / std[channel]
    lim_sup = (sup-mean[channel]) / std[channel]
    img[0][channel] = clip_image_values(img[0][channel], torch.full_like(img[0][channel], lim_inf), torch.full_like(img[0][channel], lim_sup))
  return img


# Compute valid bounds for SparseFool
def valid_bounds(img, delta=255, dataset='cifar10'):
  mean = mean_cifar10 if dataset=='cifar10' else mean_ImageNet
  std = std_cifar10 if dataset=='cifar10' else std_ImageNet
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


# Get the corresponding index considering the shape of an array
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


# Convert a tensor to a plt displayable numpy array in range [0,1]
def displayable(img, dataset='cifar10'):  # tensor of size 1xCxHxW
  return np.transpose(denormalize(img.squeeze().numpy(), dataset=dataset), (1,2,0))


# Load an image from an url and normalize using ImageNet mean and std
def image_loader(url):
  """load image, returns CUDA tensor"""
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_ImageNet, std=std_ImageNet),
  ])

  img = Image.open(requests.get(url, stream=True).raw)

  return transform(img)
