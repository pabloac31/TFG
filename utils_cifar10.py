# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from adversarial_methods import *

def testloader(path, batch_size):
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


def test_fgsm(model, device, test_loader, epsilon, iters=10000):
  
  # Initialize the network and set the model in evaluation mode. 
  model = model.to(device).eval()

  # Accuracy counter
  correct = 0
  confidence = 0
  total_time = 0
  robustness = 0
  adv_examples = []

  i = 0

  # Loop all examples in test set
  for data, target in pbar(test_loader):
    if i >= iters:
      break
    i += 1

    # Send the data and label to the device
    data, target = data.to(device), target.to(device)
  
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # If the initial prediction is wrong, dont botter attacking
    if init_pred.item() != target.item():
      continue

    # Calculate the loss
    loss = F.nll_loss(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Call FGSM attack
    time_ini = time.time()
    perturbed_data = fgsm(data, epsilon, data_grad)
    time_end = time.time()
    total_time += time_end-time_ini

    # Update model robustness
    delta = math.sqrt(32*32*3*epsilon**2)  # fgsm has a fixed delta(f,x) robustness
    im_np = data.squeeze().detach().cpu().numpy()
    robustness += delta / np.linalg.norm(im_np.flatten(), ord=2)

    # Re-classify the perturbed image
    output = model(perturbed_data)

    # Check for success
    final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    confidence += F.softmax(output, dim=1).max(1, keepdim=True)[0].item()  # adv. confidence
    if final_pred.item() == target.item():
      correct += 1
    else:
      # Save some adv examples for visualization later
      if len(adv_examples) < 5:
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append( (init_pred.item(), final_pred.item(), im_np, adv_ex) )

  # Calculate final accuracy for this epsilon
  final_acc = correct / float(iters)  # len(test_loader)
  avg_confidence = confidence / float(iters)
  avg_time = total_time / float(iters)
  model_robustness = robustness / float(iters)
  print("Epsilon: {}\nTest Accuracy = {} / {} = {}\nAverage confidence = {}\nAverage time = {}\nAverage magnitude of perturbations = {}\nModel robustness = {}"
    .format(epsilon, correct, iters, final_acc, avg_confidence, avg_time, delta, model_robustness))  # len(test_loader)

  # Return the accuracy and adversarial examples
  return final_acc, adv_examples

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


def clamp_norm(img, inf, sup):
  mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
  im = img.detach().cpu().numpy()
  for channel in range(3):
    lim_inf = (inf-mean[channel]) / std[channel]
    lim_sup = (sup-mean[channel]) / std[channel]
    for i, arr in enumerate(im[0][channel]):
      for j, pixel in enumerate(arr):
        if pixel < lim_inf:
          im[0][channel][i][j] = lim_inf
        elif pixel > lim_sup:
          im[0][channel][i][j] = lim_sup
  return (torch.from_numpy(im).to(device))
