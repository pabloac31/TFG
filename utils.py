# -*- coding: utf-8 -*-

import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as pbar


mean_cifar10, std_cifar10 = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]


def normalize_cifar10(img):
  for channel in range(3):
    img[channel] = (img[channel] - mean_cifar10[channel]) / std_cifar10[channel]
  return img


def denormalize_cifar10(img):
  for channel in range(3):
    img[channel] = img[channel] * std_cifar10[channel] + mean_cifar10[channel]
  return img


def clamp_cifar10(img, inf, sup):
  im = img.detach().cpu().numpy()
  for channel in range(3):
    lim_inf = (inf-mean_cifar10[channel]) / std_cifar10[channel]
    lim_sup = (sup-mean_cifar10[channel]) / std_cifar10[channel]
    for i, arr in enumerate(im[0][channel]):
      for j, pixel in enumerate(arr):
        if pixel < lim_inf:
          im[0][channel][i][j] = lim_inf
        elif pixel > lim_sup:
          im[0][channel][i][j] = lim_sup
  return (torch.from_numpy(im).to(device))


def testloader_cifar10(path, batch_size):
  transform = transforms.Compose([transforms.ToTensor(),
      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

  test_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10(root=path, train=False, transform=transform, download=True),
      batch_size=batch_size, shuffle=True
  )

  return test_loader


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


def plot_examples(epsilons, examples):
  cnt = 0
  plt.figure(figsize=(8,10))
  for i in range(len(epsilons)):
    for j in range(len(examples[i])):
      cnt += 1
      plt.subplot(len(epsilons), len(examples[0]), cnt)
      plt.xticks([], [])
      plt.yticks([], [])
      if j == 0:
        plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
      orig,adv,ex = examples[i][j]
      plt.title("{} -> {}".format(orig, adv))
      plt.imshow(ex, cmap="gray")
  plt.tight_layout()
  plt.show()
