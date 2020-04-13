import tensorflow as tf
import torch

import matplotlib.pyplot as plt
import numpy as np
import time
import math
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from PIL import Image
import torchvision.transforms.functional as TF

from tqdm import tqdm as pbar

from utils import *


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

  # Call FGSM attack
  adv_x = fgsm(model, x, epsilon, y, label)

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


def fgsm(model, image, epsilon, output, label):
  # Calculate the loss
  loss = F.nll_loss(output, label)
  # Zero all existing gradients
  model.zero_grad()
  # Calculate gradients of model in backward pass
  loss.backward()
  # Collect datagrad
  data_grad = image.grad.data
  # Collect the element-wise sign of the data gradient
  sign_data_grad = data_grad.sign()
  # Create the perturbed image by adjusting each pixel of the input image
  perturbed_image = image + epsilon*sign_data_grad
  # Adding clipping to maintain [0,1] range
  perturbed_image = clamp_cifar10(perturbed_image, 0, 1)
  # Return the perturbed image
  return perturbed_image


def deepfool(model, device, image, num_classes=10, overshoot=0.02, max_iter=50, lambda_fac=1.5):

  # Get the output of the original image
  output = model(image)
  # Get the input image shape
  input_shape = image.size()
  # Array with the class probabilities of the image
  f_image = output.data.cpu().numpy().flatten()
  # Classes ordered by probability (descending)
  I = f_image.argsort()[::-1]
  # We consider only 'num_classes' classes
  I = I[0:num_classes]
  # Get the predicted label
  label = I[0]

  # Start from a copy of the original image
  pert_image = copy.deepcopy(image)   # tensor of size (1,3,H,W)

  # Initialize variables
  r_tot = torch.zeros(input_shape).to(device) # adversarial perturbation
  k_i = label  # current label
  loop_i = 0

  while k_i == label and loop_i < max_iter:

    # Get the output for the current image
    x = pert_image.clone().detach().requires_grad_(True)
    fs = model(x)

    pert = torch.Tensor([np.inf])[0].to(device)
    w = torch.zeros(input_shape).to(device)

    # Calculate grad(f_label(x_i))
    fs[0, I[0]].backward(retain_graph=True)
    grad_orig = grad_orig = copy.deepcopy(x.grad.data)

    for k in range(1, num_classes):  # for k != label
      # Reset gradients
      zero_gradients(x)

      # Calculate grad(f_k(x_i))
      fs[0, I[k]].backward(retain_graph=True)
      cur_grad = copy.deepcopy(x.grad.data)

      # Set new w_k and new f_k
      w_k = cur_grad - grad_orig
      f_k = (fs[0, I[k]] - fs[0, I[0]]).data

      # Calculate hyperplane-k distance
      pert_k = torch.abs(f_k) / w_k.norm()  # Frobenious norm (2-norm)

      # determine which w_k to use
      if pert_k < pert:
        pert = pert_k + 0.
        w = w_k + 0.

    # compute r_i and r_tot
    r_i = torch.clamp(pert, min=1e-4) * w / w.norm()  # Added 1e-4 for numerical stability
    r_tot = r_tot + r_i

    # Update perturbed image
    pert_image = pert_image + r_i  # x_(i+1) <- x_i + r_i

    # Adding overshoot
    check_fool = image + (1 + overshoot) * r_tot
    x = check_fool.clone().detach().requires_grad_(True)
    # output for x_(i+1)
    fs = model(x)
    # label assigned to x_(i+1)
    k_i = torch.argmax(fs.data).item()

    loop_i += 1

  # Compute final perturbed image output
  x = pert_image.clone().detach().requires_grad_(True)
  fs = model(x)
  # Compute final gradient
  (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
  grad = copy.deepcopy(x.grad.data)
  grad = grad / grad.norm()

  # Include lambda_fac in the adversarial perturbation
  r_tot = lambda_fac * r_tot

  # Update adverarial image (pert_image = image + r_tot)
  p_im = image.detach().cpu().numpy() + r_tot.detach().cpu().numpy() # for deepcopy
  pert_image = torch.from_numpy(p_im).to(device)
  # Adding clipping to maintain [0,1] range !!!!
  #pert_image = clamp_cifar10(image + r_tot, 0, 1)

  return grad, pert_image, r_tot, loop_i


def linear_solver(x_0, normal, boundary_point, lb, ub):

  # Initialize variables
  input_shape = x_0.size()
  coord_vec = copy.deepcopy(normal)

  # Obtain plane normal vector and boundary point
  plane_normal = copy.deepcopy(coord_vec).view(-1)
  plane_point = copy.deepcopy(boundary_point).view(-1)

  x_i = copy.deepcopy(x_0)   # x_i <- x_0

  # "Linearized" classifier
  f_k = torch.dot(plane_normal, x_0.view(-1) - plane_point)
  sign_true = f_k.sign().item()

  beta = 0.001 * sign_true
  current_sign = sign_true

  while current_sign == sign_true and coord_vec.nonzero().size()[0] > 0:  # while w^T(x_i - x_B) != 0

    # Update f_k
    f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta

    # Maximum |w_j|
    pert = f_k.abs() / coord_vec.abs().max()

    mask = torch.zeros_like(coord_vec)
    mask[np.unravel_index(torch.argmax(coord_vec.abs().cpu()), input_shape)] = 1.

    # Update r_i
    r_i = torch.clamp(pert, min=1e-4) * mask * coord_vec.sign()

    # Update perturbation with the desired constraints
    x_i = x_i + r_i
    x_i = clip_image_values(x_i, lb, ub)

    # Update predictions
    f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
    current_sign = f_k.sign().item()

    coord_vec[r_i != 0] = 0

  return x_i.detach().cpu().numpy()  # for deepcopy


def sparsefool(model, device, x_0, label, lb, ub, lambda_=3., max_iter=20, epsilon=0.02):

  # Initialize variables
  x_i = copy.deepcopy(x_0)
  fool_im = copy.deepcopy(x_i)
  fool_label = label
  loops = 0

  while fool_label == label and loops < max_iter:

    # Compute l2 adversarial perturbation (using DeepFool)
    normal, x_adv,_,_ = deepfool(model, device, x_i, lambda_fac=lambda_)

    # Update x_i using the linear solver
    x_i = linear_solver(x_i, normal, x_adv, lb, ub)
    x_i = torch.from_numpy(x_i).to(device)  # necessary for deepcopy

    # Adding epsilon to compute fool_im
    fool_im = x_0 + (1 + epsilon) * (x_i - x_0)
    # Clip values using lb, ub
    fool_im = clip_image_values(fool_im, lb, ub)
    # Obtain current prediction
    x = fool_im.clone().detach().requires_grad_(True)
    fool_label = torch.argmax(model(x).data).item()

    loops += 1

  return fool_im


def universal_perturbation(dataloader, model, device, delta=0.2, xi=10, max_iter_uni=10, p=2, num_classes=10, overshoot=0.02, max_iter_df=10):

  # Initialize the network and set the model in evaluation mode.
  model = model.to(device).eval()

  fooling_rate = 0.0
  itr = 0
  num_images = 1000

  x,_ = next(iter(dataloader))
  input_shape = x.size()
  v = torch.zeros(input_shape).to(device)  # torch.Size([1, 3, 32, 32])

  while fooling_rate < 1-delta and itr < max_iter_uni:

    print('Starting pass number ', itr)

    fooling_rate = 0.0

    # Go through the data set and compute the perturbation increments sequentially
    i = 0
    for image, label in pbar(dataloader):
      if i >= num_images:
        break
      i = i+1

      image, label = image.to(device), label.to(device)

      image_v = image.add(v).to(device)
      image_v.requires_grad = True

      if model(image_v).max(1, keepdim=True)[1] == model(image).max(1, keepdim=True)[1]:

        # Compute adversarial perturbation
        _, _, dr, loop_i = deepfool(model, device, image_v, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

        # Make sure it converged...
        if loop_i < max_iter_df-1:
          v = v.add(dr)

          # Project on the lp ball centered at 0 and of radius xi
          if p == 2:
            v = v * min(1, xi / v.norm())
          elif p == np.inf:
            v = torch.sign(v) * torch.min(torch.abs(v), torch.full_like(v, xi))

    itr += 1

    # Update fooling rate
    i = 0
    for image, label in dataloader:
      if i >= num_images:
        break
      i += 1
      image, label = image.to(device), label.to(device)
      image_v = image.add(v).to(device)
      if model(image + v).max(1, keepdim=True)[1].item() != model(image).max(1, keepdim=True)[1].item():
        fooling_rate += 1
    fooling_rate /= num_images
    print('fooling_rate: ', fooling_rate)

  return v


def perturb(p, img):
  # Elements of p should be in range [0,1]
  img_size = img.size(2)  # H (= W)
  p_img = img.clone()
  xy = (p[0:2].copy() * img_size).astype(int)  # pixel x-y coordinates
  xy = np.clip(xy, 0, img_size-1)
  rgb = normalize_cifar10(p[2:5]).copy()
  rgb = clip_image_values(torch.from_numpy(rgb), normalize_cifar10(torch.tensor([0.,0.,0.], dtype=torch.double)), normalize_cifar10(torch.tensor([1.,1.,1.], dtype=torch.double)))
  p_img[0,:,xy[0],xy[1]] = rgb
  return p_img


def evaluate(model, device, candidates, img, label):
  preds = []
  model = model.to(device).eval()
  with torch.no_grad():
    for i, xs in enumerate(candidates):
      p_img = perturb(xs, img).to(device)
      preds.append(F.softmax(model(p_img).squeeze(), dim=0)[label].item())
  return np.array(preds)


def evolve(candidates, F=0.5, strategy="clip"):
  gen2 = candidates.copy()
  num_candidates = len(candidates)
  for i in range(num_candidates):
    x1, x2, x3 = candidates[np.random.choice(num_candidates, 3, replace=False)]
    x_next = (x1 + F * (x2 - x3))
    if strategy == "clip":
        gen2[i] = np.clip(x_next, 0, 1)
    elif strategy == "resample":
        x_oob = np.logical_or((x_next < 0), (1 < x_next))
        x_next[x_oob] = np.random.random(5)[x_oob]
        gen2[i] = x_next
  return gen2


def one_pixel_attack(model, device, img, label, target_label=None, iters=100, pop_size=400, verbose=True):
  # Targeted: maximize target_label if given (early stop > 50%)
  # Untargeted: minimize true_label otherwise (early stop < 5%)
  candidates = np.random.random((pop_size,5))
  candidates[:,2:5] = np.clip(np.random.normal(0.5, 0.5, (pop_size, 3)), 0, 1)
  is_targeted = target_label is not None
  label = target_label if is_targeted else label
  fitness = evaluate(model, device, candidates, img, label)

  def is_success():
      return (is_targeted and fitness.max() > 0.5) or ((not is_targeted) and fitness.min() < 0.05)

  for iteration in range(iters):
      # Early Stopping
      if is_success():
          break
      if verbose and iteration%1 == 0: # Print progress
          print("Target Probability [Iteration {}]:".format(iteration), fitness.max() if is_targeted else fitness.min())
      # Generate new candidate solutions
      new_gen_candidates = evolve(candidates, strategy="resample")
      # Evaluate new solutions
      new_gen_fitness = evaluate(model, device, new_gen_candidates, img, label)
      # Replace old solutions with new ones where they are better
      successors = new_gen_fitness > fitness if is_targeted else new_gen_fitness < fitness
      candidates[successors] = new_gen_candidates[successors]
      fitness[successors] = new_gen_fitness[successors]
  best_idx = fitness.argmax() if is_targeted else fitness.argmin()
  best_solution = candidates[best_idx]
  best_score = fitness[best_idx]

  return is_success(), best_solution, best_score


ddef attack_model(model, device, test_loader, method, params, iters=10000):

  # Initialize the network and set the model in evaluation mode.
  model = model.to(device).eval()

  # Stat counters
  correct = 0
  confidence = 0
  total_time = 0
  ex_robustness = 0
  model_robustness = 0
  adv_examples = []

  i = 0

  # Loop (iters) examples in test set
  for data, target in pbar(test_loader):
    if i >= iters:
      break
    i += 1

    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    if method in ['fgsm', 'deepfool', 'sparsefool']:
        data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # If the initial prediction is wrong, dont botter attacking
    if init_pred.item() != target.item():
      continue

    if method == 'fgsm':
        # Call FGSM attack
        time_ini = time.time()
        perturbed_data = fgsm(model, data, params["epsilon"], output, target)
        time_end = time.time()
        total_time += time_end-time_ini

    elif method == 'deepfool':
        # Call DeepFool attack
        time_ini = time.time()
        perturbed_data = deepfool(model, device, data, params["num_classes"], params["overshoot"], params["max_iter"], params["lambda_fac"])[1]
        time_end = time.time()
        total_time += time_end-time_ini

    elif method == 'sparsefool':
        # Generate lower and upper bounds
        delta = params["delta"]
        lb, ub =  valid_bounds_cifar10(data, delta)
        lb = lb[None, :, :, :].to(device)
        ub = ub[None, :, :, :].to(device)
        # Call SparseFool attack
        time_ini = time.time()
        perturbed_data = sparsefool(model, device, data, target.item(), lb, ub, params["lambda_"], params["max_iter"], params["epsilon"])
        time_end = time.time()
        total_time += time_end-time_ini

    elif method == 'one_pixel_attack':
        # Call one pixel attack
        time_ini = time.time()
        _, best_sol, score = one_pixel_attack(model, device, data, target.item(), params["target_label"], params["iters"], params["pop_size"], params["verbose"])
        perturbed_data = perturb(best_sol, data)
        time_end = time.time()
        total_time += time_end-time_ini


    # Update model robustness
    p_norm = 2
    im_np = data.squeeze().detach().cpu().numpy()
    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
    adv_rob = np.linalg.norm((adv_ex - im_np).flatten(), ord=p_norm)
    ex_robustness += adv_rob
    model_robustness += adv_rob / np.linalg.norm(im_np.flatten(), ord=p_norm)

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
        adv_examples.append( (init_pred.item(), final_pred.item(), im_np, adv_ex) )

  # Calculate stats
  final_acc = correct / float(iters)  # len(test_loader)
  avg_confidence = confidence / float(iters)
  avg_time = total_time / float(iters)
  avg_ex_robustness = ex_robustness / float(iters)
  model_robustness = model_robustness / float(iters)
  print("\n======== RESULTS ========")
  print("Test Accuracy = {} / {} = {}\nAverage confidence = {}\nAverage time = {}\nAverage magnitude of perturbations = {}\nModel robustness = {}"
    .format(correct, iters, final_acc, avg_confidence, avg_time, avg_ex_robustness, model_robustness))

  # Return adversarial examples
  return adv_examples
