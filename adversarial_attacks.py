import time
import math
import statistics

import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torchvision.transforms.functional as TF

from PIL import Image
from utils import *


""""""""""""""""""" FAST GRADIENT SIGN METHOD """""""""""""""""""

def fgsm(model, image, label, output, epsilon, clip=True, dataset='cifar10'):
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
  # Create the perturbation (considering data normalization)
  std = std_cifar10 if dataset=='cifar10' else [1,1,1]
  adv_pert = sign_data_grad
  adv_pert[0][0] = adv_pert[0][0] * (epsilon / std[0])
  adv_pert[0][1] = adv_pert[0][1] * (epsilon / std[1])
  adv_pert[0][2] = adv_pert[0][2] * (epsilon / std[2])
  # Create the perturbed image by adjusting each pixel of the input image
  perturbed_image = image + adv_pert
  # Adding clipping to maintain [0,1] range
  if clip:
    perturbed_image = clamp(perturbed_image, 0, 1, dataset)
  # Return the perturbed image and the perturbation
  return perturbed_image, adv_pert


""""""""""""""""""""""""""" DEEPFOOL """""""""""""""""""""""""""

def deepfool(model, device, im, num_classes=10, overshoot=0.02, lambda_fac=1.01, max_iter=50, p=2, clip=False, dataset='cifar10'):

  image = copy.deepcopy(im)

  # Get the input image shape
  input_shape = image.size()

  # Get the output of the original image
  output = model(image)

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
    grad_orig = copy.deepcopy(x.grad.data)

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
      if p == 2:
        pert_k = torch.abs(f_k) / w_k.norm()  # Frobenious norm (2-norm)
      elif p == np.inf:
        pert_k = torch.abs(f_k) / w_k.norm(1) # 1-norm

      # determine which w_k to use
      if pert_k < pert:
        pert = pert_k + 0.
        w = w_k + 0.

    # compute r_i and r_tot
    if p == 2:
      r_i = torch.clamp(pert, min=1e-4) * w / w.norm()  # Added 1e-4 for numerical stability
    elif p == np.inf:
      r_i = torch.clamp(pert, min=1e-4) * torch.sign(w)

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
  r_tot = lambda_fac * r_tot  # for SparseFool

  # Adding clipping to maintain [0,1] range
  if clip:
    pert_image = clamp(image + r_tot, 0, 1, dataset)

  else:
    pert_image = (image + r_tot).clone().detach()

  return grad, pert_image, r_tot, loop_i


""""""""""""""""""""""""" SPARSEFOOL """""""""""""""""""""""""

def linear_solver(x_0, normal, boundary_point, lb, ub):

  # Initialize variables
  input_shape = x_0.size()
  coord_vec = copy.deepcopy(normal)

  # Obtain plane normal vector and boundary point
  plane_normal = copy.deepcopy(coord_vec).view(-1)
  plane_point = copy.deepcopy(boundary_point).view(-1)

  x_i = copy.deepcopy(x_0)   # x(0) <- x_0

  # "Linearized" classifier
  f_k = torch.dot(plane_normal, x_0.view(-1) - plane_point)
  sign_true = f_k.sign().item()

  beta = 0.001 * sign_true
  current_sign = sign_true

  #print('sign_true', sign_true)

  while current_sign == sign_true and coord_vec.nonzero().size()[0] > 0:  # while w^T(x_i - x_B) != 0

    # Update f_k
    f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta

    # Maximum |w_j|
    pert = f_k.abs() / coord_vec.abs().max()

    mask = torch.zeros_like(coord_vec)
    mask[unravel_index(torch.argmax(coord_vec.abs()), input_shape)] = 1.

    # Update r_i
    r_i = torch.clamp(pert, min=1e-4) * mask * (-sign_true * coord_vec.sign())  # added -sign_true !!!

    # Update perturbation with the desired constraints
    x_i = x_i + r_i
    x_i = clip_image_values(x_i, lb, ub)

    # Update predictions
    f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
    current_sign = f_k.sign().item()

    coord_vec[r_i != 0] = 0

  return x_i.detach()  # for deepcopy


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

    # Adding epsilon to compute fool_im
    fool_im = x_0 + (1 + epsilon) * (x_i - x_0)
    # Clip values using lb, ub
    fool_im = clip_image_values(fool_im, lb, ub)
    # Obtain current prediction
    x = fool_im.clone().detach().requires_grad_(True)
    fool_label = torch.argmax(model(x).data).item()

    loops += 1

  r = fool_im - x_0
  return fool_im, r, loops


""""""""""""""" UNIVERSAL PERTUBATION """""""""""""""

def show_univ_examples(num_images, model, device, v):
  f = plt.figure()
  i = 0
  for img, label in adv_loader:

    if i >= num_images:
      break
    i += 1

    img = img.to(device)
    label = label.to(device)

    y = model(img)
    init_pred = y.max(1, keepdim=True)[1]

    f.add_subplot(3,num_images,i)
    plt.axis('off')
    f.text(.03 + (0.8/num_images)*i, .62, cifar10_classes[label.item()], ha='center')
    plt.imshow(displayable(img.cpu()))

    adv_x = img + v
    y_adv = model(adv_x)
    adv_pred = y_adv.max(1, keepdim=True)[1]

    f.add_subplot(3,num_images,2*num_images+i)
    plt.axis('off')
    f.text(.03 + (0.8/num_images)*i, .1, cifar10_classes[adv_pred.item()], ha='center')
    plt.imshow(displayable(adv_x.cpu()))

  f.add_subplot(3,num_images,num_images + np.ceil(num_images/2))
  plt.axis('off')
  f.text(0.7, 0.5, 'Univ. perturbation', ha='center')
  plt.imshow(displayable(v.cpu()))

  plt.show(block=True)


def univ_fool_rate(model, device, dataset, v, batch_size=250):

    model = model.to(device).eval()
    dataset = dataset.to(device)

    num_images = dataset.size(0)

    # Perturb the dataset with the universal perturbation v
    dataset_perturbed = (dataset + v).to(device)

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))
    fooling_rate = 0.0

    # Compute the estimated labels in batches
    for ii in pbar(range(num_batches)):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)

        with torch.no_grad():

          est_labels_orig = torch.argmax(model(dataset[m:M, :, :, :]), axis=1)
          est_labels_pert = torch.argmax(model(dataset_perturbed[m:M, :, :, :]), axis=1)

        fooling_rate += torch.sum(est_labels_pert != est_labels_orig).item()

    # Compute the fooling rate
    fooling_rate = fooling_rate / float(num_images)
    return fooling_rate


def universal_perturbation(dataset, labels, model, device, delta=0.2, xi=10, max_iter_uni=10, p=2, num_classes=10, overshoot=0.02, max_iter_df=10, v_ini=None):

  time_ini = time.time()

  # Initialize the network and set the model in evaluation mode.
  model = model.to(device).eval()
  dataset = dataset.to(device)
  labels = labels.to(device)

  v = v_ini.clone() if v_ini is not None else torch.zeros((1, dataset.size()[1], dataset.size()[2], dataset.size()[3])).to(device)
  fooling_rate = 0.0
  num_images = dataset.size()[0]

  v_best = v.clone()
  fool_rate_best = 0.0

  itr = 0
  while fooling_rate < 1-delta and itr < max_iter_uni:

    # Shuffle the dataset
    order = np.arange(num_images)
    np.random.shuffle(order)
    dataset[np.arange(num_images)] = dataset[order]
    labels[np.arange(num_images)] = labels[order]

    print('Starting pass number ', itr)

    # Go through the data set and compute the perturbation increments sequentially
    for k in pbar(range(num_images)):
      cur_img = dataset[k:(k+1), :, :, :]
      label = labels[k]

      pred_label = model(cur_img).max(1, keepdim=True)[1].item()

      if pred_label == label.item() and model(cur_img + v).max(1, keepdim=True)[1].item() == pred_label:

        # Compute adversarial perturbation
        _, _, dr, loop_i = deepfool(model, device, cur_img + v, num_classes=num_classes, overshoot=overshoot, lambda_fac=1+overshoot, max_iter=max_iter_df, p=p)

        # Make sure it converged...
        if loop_i < max_iter_df-1:
          v = v + dr

          # Project on the lp ball centered at 0 and of radius xi
          if p == 2:
            v = v * min(1, xi / v.norm())
          elif p == np.inf:
            v = torch.sign(v) * torch.min(torch.abs(v), torch.full_like(v, xi))

    itr += 1

    fooling_rate = univ_fool_rate(model, device, dataset, v, batch_size=100)
    print('FOOLING RATE = ', fooling_rate)

    if fooling_rate > fool_rate_best:
      v_best = v.clone()
      fool_rate_best = fooling_rate

  time_end = time.time()
  total_time = time_end-time_ini
  print('Total time: {:.2f}'.format(total_time))
  print('Total iters:', itr)
  print('Norm of the univ. perturbation: {:.4f}'.format(torch.norm(v, 2 if p==2 else float('inf')).item()))

  return v_best, fool_rate_best



""""""""""""""" ONE PIXEL ATTACK """""""""""""""

def perturb(p, img, dataset='cifar10'):
  # Elements of p should be in range [0,1]
  img_size = img.size(2)  # H (= W)
  p_img = img.clone()

  for pert in p:
    # Convert x-y coordinates to range [0,img_size)
    xy = (pert[0:2].copy() * img_size).astype(int)
    xy = np.clip(xy, 0, img_size-1)

    # Normalize RGB pixel values and clip to maintain the correct range
    rgb = normalize(pert[2:5].copy(), dataset=dataset)
    rgb = clip_image_values(torch.from_numpy(rgb), min_rgb[dataset], max_rgb[dataset])

    # Change correponding pixels of the image
    p_img[0,:,xy[0],xy[1]] = rgb

  return p_img


def evaluate(model, device, candidates, img, label, dataset='cifar10'):
  preds = []
  # model = model.to(device).eval()  # already in eval mode when called
  with torch.no_grad():
    for i, xs in enumerate(candidates):
      # Calculate the perturbed image
      p_img = perturb(xs, img, dataset).to(device)
      # Append the probability of the target label
      preds.append(F.softmax(model(p_img).squeeze(), dim=0)[label].item())
  return np.array(preds)


def evolve(candidates, d, F=0.5, strategy="clip"):
  gen2 = candidates.copy()
  num_candidates = len(candidates)
  for i in range(num_candidates):
    for p in range(d):
      # Apply usual DE formula
      rdm_idx = np.random.choice(num_candidates * d, 3, replace=False)
      x1, x2, x3 = candidates[rdm_idx % num_candidates, rdm_idx // num_candidates]
      x_next = (x1 + F * (x2 - x3))
      if strategy == "clip":
          gen2[i,p] = np.clip(x_next, 0, 1)
      elif strategy == "resample":
          x_oob = np.logical_or((x_next < 0), (1 < x_next))
          x_next[x_oob] = np.random.random(5)[x_oob]
          gen2[i,p] = x_next
  return gen2


def one_pixel_attack(model, device, img, label, d=1, target_label=None, iters=100, pop_size=400, verbose=True, dataset='cifar10'):

  # Targeted: maximize target_label if given (early stop > 50%)
  # Untargeted: minimize true_label otherwise (early stop < 10%)

  # Initialize population
  candidates = np.random.random((pop_size, d, 5))
  # RGB values in range [0,1] from a Gaussian distribution N(0.5,0.5)
  candidates[:,:,2:5] = np.clip(np.random.normal(0.5, 0.5, (pop_size, d, 3)), 0, 1)

  # Select label for a targeted / non-targeted attack
  is_targeted = target_label is not None
  label = target_label if is_targeted else label

  # Initial score (prob. of each perturbation)
  fitness = evaluate(model, device, candidates, img, label, dataset)

  scores = []

  def is_success():
      return (is_targeted and fitness.max() > 0.5) or ((not is_targeted) and fitness.min() < 0.1)

  for iteration in range(iters):
      # Early Stopping
      if is_success():
          break

      if verbose and iteration%1 == 0: # Print progress
          print("Target Probability [Iteration {}]: {:.4f}".format(iteration, fitness.max() if is_targeted else fitness.min()))
          scores.append(fitness.max() if is_targeted else fitness.min())

      # Generate new candidate solutions
      new_gen_candidates = evolve(candidates, d, strategy="resample")
      # Evaluate new solutions
      new_gen_fitness = evaluate(model, device, new_gen_candidates, img, label, dataset)
      # Replace old solutions with new ones where they are better
      successors = new_gen_fitness > fitness if is_targeted else new_gen_fitness < fitness
      candidates[successors] = new_gen_candidates[successors]
      fitness[successors] = new_gen_fitness[successors]

  # Select best candidate
  best_idx = fitness.argmax() if is_targeted else fitness.argmin()
  best_solution = candidates[best_idx]
  best_score = fitness[best_idx]

  return perturb(best_solution, img), iteration, scores


# Test the desired method in one image
def test_method(model, device, img, label, method, params):

  img = img.clone()

  model = model.to(device).eval()

  x = img.to(device)
  label = label.to(device)

  if method in ['fgsm', 'deepfool', 'sparsefool']:
    x.requires_grad = True

  y = model(x)
  init_pred = y.max(1, keepdim=True)[1]
  x_conf = F.softmax(y, dim=1).max(1, keepdim=True)[0].item()

  if init_pred.item() != label.item():
    print("Wrong classification...")
    return

  # Call method
  if method == 'fgsm':
    adv_x, pert_x = fgsm(model, x, label, y, params["epsilon"], params["clip"])

  elif method == 'deepfool':
    _, adv_x, pert_x, n_iter = deepfool(model, device, x, params["num_classes"], overshoot=params["overshoot"], max_iter=params["max_iter"], p=params["p"], clip=params["clip"])

  elif method == 'sparsefool':
    # Generate lower and upper bounds
    delta = params["delta"]
    lb, ub =  valid_bounds(x, delta, dataset='cifar10')
    lb = lb[None, :, :, :].to(device)
    ub = ub[None, :, :, :].to(device)
    adv_x, pert_x, n_iter = sparsefool(model, device, x, label.item(), lb, ub, params["lambda_"], params["max_iter"], params["epsilon"])

  elif method == 'one_pixel_attack':
    adv_x, n_iter, scores = one_pixel_attack(model, device, x, label.item(), params["dim"], params["target_label"], params["iters"], params["pop_size"], params["verbose"])
    pert_x = adv_x - x

  y_adv = model(adv_x)
  adv_pred = y_adv.max(1, keepdim=True)[1]
  adv_x_conf = F.softmax(y_adv, dim=1).max(1, keepdim=True)[0].item()

  if adv_pred.item() == label.item():
    print("Attack failed...")

  else:
    print("Succesful attack!")

  f = plt.figure()
  f.add_subplot(1,3,1)
  plt.title('Original image')
  plt.axis('off')
  f.text(.25, .3, cifar10_classes[label.item()] + ' ({:.2f}%)'.format(x_conf*100), ha='center')
  plt.imshow(displayable(img))
  f.add_subplot(1,3,2)
  plt.title('Perturbation')
  plt.axis('off')
  plt.imshow(displayable(pert_x.cpu().detach()))
  f.add_subplot(1,3,3)
  plt.title('Adv. image')
  plt.axis('off')
  f.text(.8, .3, cifar10_classes[adv_pred.item()] + ' ({:.2f}%)'.format(adv_x_conf*100), ha='center')
  plt.imshow(displayable(adv_x.cpu().detach()))
  plt.show(block=True)

  if method in ['deepfool',  'sparsefool', 'one_pixel_attack']:
    print('Number of iterations needed: ', n_iter)

  if method == 'sparsefool':
    pert_pixels = pert_x.flatten().nonzero().size(0)
    print('Number of perturbed pixels: ', pert_pixels)

  if method == 'one_pixel_attack':
    return scores


# Performs an attack and shows the results achieved by some method
def attack_model(model, device, test_loader, method, params, p=2, iters=10000, dataset='cifar10'):

  # Initialize the network and set the model in evaluation mode.
  model = model.to(device).eval()

  # Initialize stat counters
  correct = 0
  incorrect = 0
  confidence = 0
  total_time = 0
  ex_robustness = 0
  model_robustness = 0
  method_iters = 0
  n_pert_pixels = []
  adv_examples = []

  i = 0

  # Loop (iters) examples in test set
  for data, target in pbar(test_loader):
    if i >= iters:
      break
    i += 1

    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor (important for some attacks)
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
        perturbed_data, _ = fgsm(model, data, target, output, params["epsilon"], params["clip"], dataset)
        time_end = time.time()
        total_time += time_end-time_ini

    elif method == 'deepfool':
        # Call DeepFool attack
        time_ini = time.time()
        _, perturbed_data, _, n_iter = deepfool(model, device, data, params["num_classes"], overshoot=params["overshoot"], max_iter=params["max_iter"], p=params["p"], clip=params["clip"])
        time_end = time.time()
        total_time += time_end-time_ini
        method_iters += n_iter

    elif method == 'sparsefool':
        # Generate lower and upper bounds
        delta = params["delta"]
        lb, ub =  valid_bounds(data, delta, dataset='cifar10')
        lb = lb[None, :, :, :].to(device)
        ub = ub[None, :, :, :].to(device)
        # Call SparseFool attack
        time_ini = time.time()
        perturbed_data, perturbation, n_iter = sparsefool(model, device, data, target.item(), lb, ub, params["lambda_"], params["max_iter"], params["epsilon"])
        time_end = time.time()
        total_time += time_end-time_ini
        method_iters += n_iter
        n_pert_pixels.append(perturbation.flatten().nonzero().size(0))

    elif method == 'one_pixel_attack':
        # Call one pixel attack
        time_ini = time.time()
        perturbed_data, n_iter, _ = one_pixel_attack(model, device, data, target.item(), params["dim"], params["target_label"], params["iters"], params["pop_size"], params["verbose"])
        time_end = time.time()
        total_time += time_end-time_ini
        method_iters += n_iter


    # Update model robustness
    # multiply by std to make it independent of the normalization used
    difference = de_scale(perturbed_data-data, dataset)
    if p == 2:
      adv_rob = torch.norm(difference)  # Frobenius norm (p=2)
      model_robustness += adv_rob / torch.norm(de_scale(data, dataset))
    elif p == np.inf:
      adv_rob = torch.norm(difference, float('inf'))  # Inf norm (p=inf)
      model_robustness += adv_rob / torch.norm(de_scale(data, dataset), float('inf'))
    ex_robustness += adv_rob

    # Re-classify the perturbed image
    output = model(perturbed_data)

    # Check for success
    final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    if final_pred.item() == target.item():
      correct += 1

    else:
      incorrect += 1
      # Update average confidence
      confidence += F.softmax(output, dim=1).max(1, keepdim=True)[0].item()
      # Save some adv examples for visualization later
      if len(adv_examples) < 5:
        adv_examples.append( (init_pred.item(), final_pred.item(), data.detach().cpu(), perturbed_data.detach().cpu()) )

  # Calculate stats
  final_acc = correct / float(iters)  # len(test_loader)
  avg_confidence = confidence / float(incorrect)
  avg_time = total_time / float(correct+incorrect)
  avg_ex_robustness = ex_robustness / float(correct+incorrect)
  model_robustness = model_robustness / float(correct+incorrect)
  print("\n======== RESULTS ========")
  print("Test Accuracy = {} / {} = {:.4f}\nAverage confidence = {:.4f}\nAverage time = {:.4f}\nAverage magnitude of perturbations = {:.4f}\nModel robustness = {:.4f}"
    .format(correct, iters, final_acc, avg_confidence, avg_time, avg_ex_robustness, model_robustness))

  if method in ['deepfool', 'sparsefool', 'one_pixel_attack']:
    print("Avg. iters = {:.2f}".format(method_iters / float(correct+incorrect)))

  if method == 'sparsefool':
    print("Median num. of pixels perturbed = ", statistics.median(n_pert_pixels))
    print("Average num. of pixels perturbed = {:.2f}".format(statistics.mean(n_pert_pixels)))

  # Return adversarial examples
  return adv_examples
