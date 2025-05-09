"""
Utility functions for neural style transfer and image processing.
This module contains functions extracted from testingproposedmethods.ipynb.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import time
import functools

import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pytorch_msssim import ssim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_image_tf(tensor):
    tensor = np.array(tensor * 255, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    else:
        raise Exception()

    return PIL.Image.fromarray(tensor)


def load_image(image_path):
    max_res = 512

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_side = max(shape)
    scaling_factor = max_res / long_side

    new_shape = tf.cast(shape * scaling_factor, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]

    return image

def imshow(image, title = None):
    if isinstance(image, np.ndarray):
        if len(image.shape) > 3:
            image = image[0]
    elif isinstance(image, tf.Tensor):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis = 0)
        image = image.numpy()

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')

def image_loader(image_path, imsize=512):
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def get_feature_map(model, image):
    feature_map = []
    for name, layer in model.named_children():
        image = layer(image)
        feature_map.append((name, image))

    outputs = list()
    for name, feature in feature_map:
        feature = feature.squeeze(0)
        gray_scale = torch.sum(feature, 0)
        gray_scale = gray_scale / feature.shape[0]
        outputs.append((name, gray_scale.data.numpy()))

    return outputs

def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach().numpy().squeeze(0)
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def display_generated(img, title = None):
    with torch.no_grad():
        img_cloned = img.clone()
        img_cloned.clamp_(0, 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(tensor_to_image(img_cloned))
        plt.title(title)
        plt.axis('off')
        plt.show()

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach()
        self.std = std.clone().detach()

    def forward(self, img):
        return (img - self.mean) / self.std

def content_loss(target, content):
    return 0.5 * F.mse_loss(target, content)

def gram_matrix(features):
    batch_size, channels, height, width = features.size()
    features_reshaped = features.view(batch_size, channels, -1)

    gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))

    return gram / (channels * height * width)

def style_loss(style_features, target_features, layer_weights):
    assert len(style_features) == len(target_features) == len(layer_weights), "Number of layers must match"

    L_style = torch.tensor(0.0, device = device, requires_grad = True)

    for l in range(len(style_features)):
        style_gram = gram_matrix(style_features[l])
        target_gram = gram_matrix(target_features[l])

        E_l = F.mse_loss(target_gram, style_gram)

        L_style = L_style + layer_weights[l] * E_l

    return L_style

def extract_features(model, input_img, layers_of_interest):
    features = {}
    x = input_img.clone()

    for name, module in model.named_children():
        x = module(x)
        if name in layers_of_interest:
            features[name] = x.detach() if not input_img.requires_grad else x

    return features


def create_model_and_losses(cnn, style_img, content_img, style_layer_weights=None):
    normal_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    normal_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    normalization = Normalization(normal_mean, normal_std).to(device)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential(normalization)

    if style_layer_weights is None:
        style_layer_weights = [0.2] * len(style_layers)

    assert len(style_layer_weights) == len(style_layers), \
        f"Expected {len(style_layers)} weights but got {len(style_layer_weights)}"

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            name = f'other_{i}'

        model.add_module(name, layer)

    content_features = extract_features(model, content_img, content_layers)
    style_features = extract_features(model, style_img, style_layers)
    style_feature_list = [style_features[layer] for layer in style_layers]

    return model, style_feature_list, content_features[content_layers[0]], style_layer_weights, content_layers, style_layers

def run_style_transfer(content_img, style_img, num_iterations, alpha, beta, style_layer_weights = None):
    content_img = content_img.to(device)
    style_img = style_img.to(device)

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)
    model, style_feature_list, content_feature, style_layer_weights, content_layers, style_layers = create_model_and_losses(
        cnn, style_img, content_img, style_layer_weights
    )

    model.eval()
    generate_img = content_img.clone()
    generate_img.requires_grad_(True)
    optimizer = optim.LBFGS([generate_img])

    epoch = [0]

    while epoch[0] < num_iterations:
        def closure():
            with torch.no_grad():
                generate_img.clamp_(0, 1)

            optimizer.zero_grad()
            gen_content_features = extract_features(model, generate_img, content_layers)
            gen_style_features = extract_features(model, generate_img, style_layers)

            c_loss = content_loss(content_feature, gen_content_features[content_layers[0]])
            gen_style_feature_list = [gen_style_features[layer] for layer in style_layers]
            s_loss = style_loss(style_feature_list, gen_style_feature_list, style_layer_weights)
            loss = alpha * c_loss + beta * s_loss
            loss.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0 or epoch[0] == num_iterations:
                print(f"Iteration {epoch[0]}/{num_iterations}:")
                print(f"Style Loss: {beta * s_loss.item():.4f} Content Loss: {alpha * c_loss.item():.4f}")
                display_generated(generate_img, title=f'Iteration {epoch[0]}')

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        generate_img.clamp_(0, 1)

    return generate_img


def multi_layer_content_loss(content_features, target_features, layer_weights):
    assert len(content_features) == len(target_features) == len(layer_weights), "Number of layers must match"

    L_content = torch.tensor(0.0, device=device, requires_grad = True)

    for l in range(len(content_features)):
        layer_loss = F.mse_loss(target_features[l], content_features[l])

        L_content = L_content + layer_weights[l] * layer_loss

    return L_content


def create_model_and_losses1(cnn, style_img, content_img, style_layer_weights = None, content_layer_weights = None):
    normal_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    normal_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    normalization = Normalization(normal_mean, normal_std).to(device)

    content_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            name = f'other_{i}'

        model.add_module(name, layer)
        
    content_features = extract_features(model, content_img, content_layers)
    style_features = extract_features(model, style_img, style_layers)

    style_feature_list = [style_features[layer] for layer in style_layers]
    content_feature_list = [content_features[layer] for layer in content_layers]

    return model, style_feature_list, content_feature_list, style_layer_weights, content_layer_weights, content_layers, style_layers

def run_style_transfer1(content_img, style_img, num_iterations, alpha, beta, style_layer_weights = None, content_layer_weights = None):
    content_img = content_img.to(device)
    style_img = style_img.to(device)

    cnn = models.vgg19(weights = models.VGG19_Weights.DEFAULT).features.eval().to(device)
    model, style_feature_list, content_feature_list, style_layer_weights, content_layer_weights, content_layers, style_layers = create_model_and_losses1(
        cnn, style_img, content_img, style_layer_weights, content_layer_weights
    )

    generate_img = content_img.clone()
    generate_img.requires_grad_(True)
    optimizer = optim.LBFGS([generate_img])

    epoch = [0]

    while epoch[0] < num_iterations:
        def closure():
            with torch.no_grad():
                generate_img.clamp_(0, 1)

            optimizer.zero_grad()

            gen_content_features = extract_features(model, generate_img, content_layers)
            gen_style_features = extract_features(model, generate_img, style_layers)

            gen_content_feature_list = [gen_content_features[layer] for layer in content_layers]
            gen_style_feature_list = [gen_style_features[layer] for layer in style_layers]

            c_loss = multi_layer_content_loss(content_feature_list, gen_content_feature_list, content_layer_weights)
            s_loss = style_loss(style_feature_list, gen_style_feature_list, style_layer_weights)
            loss = alpha * c_loss + beta * s_loss
            loss.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0 or epoch[0] == num_iterations:
                print(f"Iteration {epoch[0]}/{num_iterations}:")
                print(f"Style Loss: {beta * s_loss.item():.4f} Content Loss: {alpha * c_loss.item():.4f}")
                display_generated(generate_img, title=f'Iteration {epoch[0]}')

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        generate_img.clamp_(0, 1)

    return generate_img


def compute_laplacian(x):
    orig_shape = x.shape
    if x.dim() == 3:
        x = x.unsqueeze(0)

    kernel = torch.tensor([[0, -1,  0],
                           [-1, 4, -1],
                           [0, -1,  0]],
                          dtype=x.dtype, device=x.device)
    C = x.shape[1]
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    lap = F.conv2d(x, kernel, padding=1, groups=C)
    return lap.squeeze(0) if len(orig_shape) == 3 else lap

def laplacian_loss(pred, target):
    return (pred - target).pow(2).mean()

def run_style_transfer2(content_img, style_img, num_iterations, alpha, beta, lambda_lap, style_layer_weights=None, content_layer_weights=None):
    content_img = content_img.to(device)
    style_img   = style_img.to(device)

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)
    model, style_feature_list, content_feature_list, style_layer_weights, content_layer_weights, content_layers, style_layers = create_model_and_losses1(
        cnn, style_img, content_img, style_layer_weights, content_layer_weights)

    generated = content_img.clone().requires_grad_(True)
    optimizer = optim.LBFGS([generated])

    iteration = [0]
    while iteration[0] < num_iterations:
        def closure():
            with torch.no_grad():
                generated.clamp_(0, 1)

            optimizer.zero_grad()

            gen_c_feats = extract_features(model, generated, content_layers)
            gen_s_feats = extract_features(model, generated, style_layers)
            gen_c_list  = [gen_c_feats[l] for l in content_layers]
            gen_s_list  = [gen_s_feats[l] for l in style_layers]

            ml_content = multi_layer_content_loss(
                content_feature_list, gen_c_list, content_layer_weights
            )

            lap_gen     = compute_laplacian(generated)
            lap_content = compute_laplacian(content_img)
            lap_loss    = laplacian_loss(lap_gen, lap_content)

            total_content = ml_content + lambda_lap * lap_loss
            s_loss = style_loss(
                style_feature_list, gen_s_list, style_layer_weights
            )

            loss = alpha * total_content + beta * s_loss
            loss.backward()

            iteration[0] += 1
            if iteration[0] % 50 == 0 or iteration[0] == num_iterations:
                print(f"Iteration {iteration[0]}/{num_iterations}:")
                print(f"  Style Loss: {beta * s_loss.item():.4f}")
                print(f"  Content Loss: {alpha * total_content.item():.4f}")
                print(f"    • Multi‐layer: {alpha * ml_content.item():.4f}")
                print(f"    • Laplacian:  {alpha * lambda_lap * lap_loss.item():.4f}")
                display_generated(generated, title=f"Iteration {iteration[0]}")

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        generated.clamp_(0, 1)
    return generated


def sobel_edge_loss(content, generated):

    device = content.device
    channels = content.shape[1]

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)

    content_gx = F.conv2d(content, sobel_x, padding=1, groups=channels)
    content_gy = F.conv2d(content, sobel_y, padding=1, groups=channels)

    generated_gx = F.conv2d(generated, sobel_x, padding=1, groups=channels)
    generated_gy = F.conv2d(generated, sobel_y, padding=1, groups=channels)

    diff_x = content_gx - generated_gx
    diff_y = content_gy - generated_gy

    loss = torch.mean(diff_x**2 + diff_y**2)

    return loss


def run_style_transfer3(content_img, style_img, num_iterations, alpha, beta,
                        style_layer_weights = None, content_layer_weights = None,
                        gamma = 0.9):  
    content_img = content_img.to(device)
    style_img = style_img.to(device)

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)
    (model, style_feature_list, content_feature_list,
     style_layer_weights, content_layer_weights,
     content_layers, style_layers) = create_model_and_losses1(
        cnn, style_img, content_img, style_layer_weights, content_layer_weights
    )

    generate_img = content_img.clone()
    generate_img.requires_grad_(True)
    optimizer = optim.LBFGS([generate_img])

    epoch = [0]

    while epoch[0] < num_iterations:
        def closure():
            with torch.no_grad():
                generate_img.clamp_(0, 1)

            optimizer.zero_grad()

            gen_content_features = extract_features(model, generate_img, content_layers)
            gen_style_features = extract_features(model, generate_img, style_layers)

            gen_content_feature_list = [gen_content_features[layer] for layer in content_layers]
            gen_style_feature_list = [gen_style_features[layer] for layer in style_layers]
            c_loss = multi_layer_content_loss(content_feature_list,
                                              gen_content_feature_list,
                                              content_layer_weights)

            s_loss = style_loss(style_feature_list,
                                gen_style_feature_list,
                                style_layer_weights)
            e_loss = sobel_edge_loss(content_img, generate_img)
            loss = alpha * c_loss + beta * s_loss + gamma * e_loss
            loss.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0 or epoch[0] == num_iterations:
                print(f"Iteration {epoch[0]}/{num_iterations}:")
                print(f"   Style Loss: {beta * s_loss.item():.4f} "
                      f"Content Loss: {alpha * c_loss.item():.4f} "
                      f"Edge Loss: {gamma * e_loss.item():.4f} "
                      f"Total: {loss.item():.4f}")
                display_generated(generate_img, title=f'Iteration {epoch[0]}')

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        generate_img.clamp_(0, 1)

    return generate_img


def total_variation_loss(img):

    diff_x = img[:, :, :, 1:] - img[:, :, :, :-1]
    diff_y = img[:, :, 1:, :] - img[:, :, :-1, :]

    tv_loss = torch.sum(diff_x.abs().pow(2)) + torch.sum(diff_y.abs().pow(2))

    batch_size, channels, height, width = img.size()
    tv_loss = tv_loss / (batch_size * channels * height * width)

    return tv_loss

def run_style_transfer4(content_img, style_img, num_iterations,
                                alpha, beta, gamma, delta,
                                style_layer_weights=None, content_layer_weights=None):
    content_img = content_img.to(device)
    style_img = style_img.to(device)

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)
    (model, style_feature_list, content_feature_list,
     style_layer_weights, content_layer_weights,
     content_layers, style_layers) = create_model_and_losses1(
        cnn, style_img, content_img, style_layer_weights, content_layer_weights
    )

    generate_img = content_img.clone()
    generate_img.requires_grad_(True)
    optimizer = optim.LBFGS([generate_img])

    epoch = [0]

    while epoch[0] < num_iterations:
        def closure():
            with torch.no_grad():
                generate_img.clamp_(0, 1)

            optimizer.zero_grad()
            gen_content_features = extract_features(model, generate_img, content_layers)
            gen_style_features = extract_features(model, generate_img, style_layers)

            gen_content_feature_list = [gen_content_features[layer] for layer in content_layers]
            gen_style_feature_list = [gen_style_features[layer] for layer in style_layers]
            c_loss = multi_layer_content_loss(content_feature_list,
                                             gen_content_feature_list,
                                             content_layer_weights)
            s_loss = style_loss(style_feature_list,
                               gen_style_feature_list,
                               style_layer_weights)
            e_loss = sobel_edge_loss(content_img, generate_img)
            t_loss = total_variation_loss(generate_img)
            loss = (alpha * c_loss +
                   beta * s_loss +
                   gamma * e_loss +
                   delta * t_loss)

            loss.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0 or epoch[0] == num_iterations:
                print(f"Iteration {epoch[0]}/{num_iterations}:")
                print(f"   Content Loss: {alpha * c_loss.item():.4f} "
                      f"Style Loss: {beta * s_loss.item():.4f} "
                      f"Edge Loss: {gamma * e_loss.item():.4f} "
                      f"TV Loss: {delta * t_loss.item():.4f} "
                      f"Total: {loss.item():.4f}")
                display_generated(generate_img, title=f'Iteration {epoch[0]}')

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        generate_img.clamp_(0, 1)

    return generate_img

def run_style_transfer5(content_img, style_img, num_iterations,
                        alpha, beta, gamma, delta,
                        style_layer_weights=None, content_layer_weights=None):
    content_img = content_img.to(device)
    style_img = style_img.to(device)

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)
    (model, style_feature_list, content_feature_list,
     style_layer_weights, content_layer_weights,
     content_layers, style_layers) = create_model_and_losses1(
         cnn, style_img, content_img,
         style_layer_weights, content_layer_weights
    )

    generate_img = content_img.clone().requires_grad_(True)
    optimizer = optim.LBFGS([generate_img])
    epoch = [0]

    while epoch[0] < num_iterations:
        def closure():
            with torch.no_grad():
                generate_img.clamp_(0, 1)
            optimizer.zero_grad()
            gen_c_feats = extract_features(model, generate_img, content_layers)
            gen_s_feats = extract_features(model, generate_img, style_layers)
            c_loss = multi_layer_content_loss(
                content_feature_list,
                [gen_c_feats[l] for l in content_layers],
                content_layer_weights
            )
            s_loss = style_loss(
                style_feature_list,
                [gen_s_feats[l] for l in style_layers],
                style_layer_weights
            )
            e_loss = sobel_edge_loss(content_img, generate_img)
            t_loss = total_variation_loss(generate_img)
            total_loss = (
                alpha * c_loss +
                beta * s_loss +
                gamma * e_loss +
                delta * t_loss
            )
            total_loss.backward()
            epoch[0] += 1
            return total_loss
        optimizer.step(closure)

    with torch.no_grad():
        generate_img.clamp_(0, 1)
        gen_c_feats = extract_features(model, generate_img, content_layers)
        gen_s_feats = extract_features(model, generate_img, style_layers)
        final_c = multi_layer_content_loss(
            content_feature_list,
            [gen_c_feats[l] for l in content_layers],
            content_layer_weights
        )
        final_s = style_loss(
            style_feature_list,
            [gen_s_feats[l] for l in style_layers],
            style_layer_weights
        )
        final_e = sobel_edge_loss(content_img, generate_img)
        final_t = total_variation_loss(generate_img)
        final_total = (
            alpha * final_c +
            beta * final_s +
            gamma * final_e +
            delta * final_t
        ).item()


    display_generated(generate_img)
    return generate_img

