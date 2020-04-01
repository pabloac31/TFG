import tensorflow as tf
import torch

import matplotlib.pyplot as plt
import numpy as np
import time
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import torchvision.transforms.functional as TF

from tqdm import tqdm as pbar

from utils import *
from adversarial_methods import *


def test_fgsm(model, device, img, epsilon):
  
  model = model.to(device).eval()
  
  image = Image.open(img)
  x = TF.to_tensor(image)

  x = normalize_cifar10(x)

  x = x.unsqueeze_(0).to(device)

  label = torch.tensor([1]).to(device)

  x.requires_grad = True

  y = model(x)
  init_pred = y.max(1, keepdim=True)[1]
  print("Original image prediction: ", init_pred.item())

  if init_pred.item() != label.item():
    print("Image misclassified...")
    return

  # Calculate the loss
  loss = F.nll_loss(y, label)

  # Zero all existing gradients
  model.zero_grad()

  # Calculate gradients of model in backward pass
  loss.backward()

  # Collect datagrad
  x_grad = x.grad.data

  # Call FGSM attack
  adv_x = fgsm(x, epsilon, x_grad)  #  +-28 pixeles en [0,255]

  y_adv = model(adv_x)
  adv_pred = y_adv.max(1, keepdim=True)[1]
  print("Adversarial image prediction: ", adv_pred.item())

  if adv_pred.item() == label.item():
    print("Attack failed... try with a greater epsilon")

  else:
    print("Succesful attack!")

  adv_ex = adv_x.squeeze().detach().cpu().numpy()

  adv_ex = denormalize_cifar10(adv_ex)

  adv_ex = np.transpose(adv_ex, (1,2,0))

  plt.imshow(np.asarray(image))
  plt.title('Original image')
  plt.show()
  plt.imshow(adv_ex)
  plt.title('Adversarial image')
  plt.show()


def full_test_fgsm(model, device, test_loader, epsilon, iters=10000):
  
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
