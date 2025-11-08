import torch
import torch.nn as nn
import time
import numpy as np
import logging


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_nonzero_parameters(model):
    total_nonzero = 0
    total_params = 0

    for param in model.parameters():
        total_params += param.numel()
        total_nonzero += torch.count_nonzero(param).item()

    sparsity = 100.0 * (1 - total_nonzero / total_params)
    return total_nonzero, total_params, sparsity


def count_flops(model, input_size=(1, 3, 32, 32), device='cpu'):
    model = model.to(device)
    model.eval()

    flops_dict = {}

    def conv2d_flops_hook(module, input, output):
        batch_size, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups

        kernel_flops = kernel_h * kernel_w * (in_channels // groups)
        bias_flops = 1 if module.bias is not None else 0
        output_size = out_h * out_w * out_channels

        flops = batch_size * output_size * (kernel_flops + bias_flops)
        flops_dict[module] = flops

    def linear_flops_hook(module, input, output):
        batch_size = input[0].size(0)
        in_features = module.in_features
        out_features = module.out_features

        flops = batch_size * in_features * out_features
        if module.bias is not None:
            flops += batch_size * out_features

        flops_dict[module] = flops

    def batchnorm_flops_hook(module, input, output):
        batch_size, num_features = output.shape[0], output.shape[1]
        output_size = output.numel() / batch_size
        flops = batch_size * output_size * 2
        flops_dict[module] = flops

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_flops_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flops_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(batchnorm_flops_hook))

    with torch.no_grad():
        dummy_input = torch.randn(input_size).to(device)
        model(dummy_input)

    total_flops = sum(flops_dict.values())

    for hook in hooks:
        hook.remove()

    return total_flops


def measure_inference_time(model, input_size=(1, 3, 32, 32), device='cpu', num_runs=100, warmup_runs=10):
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time


def get_model_summary(model, input_size=(1, 3, 32, 32), device='cpu'):
    total_params, trainable_params = count_parameters(model)
    nonzero_params, _, sparsity = count_nonzero_parameters(model)
    flops = count_flops(model, input_size, device)
    avg_time, std_time = measure_inference_time(model, input_size, device)

    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'nonzero_parameters': nonzero_params,
        'sparsity_percentage': sparsity,
        'flops': flops,
        'inference_time_ms': avg_time,
        'inference_time_std_ms': std_time
    }

    return summary


def print_model_summary(summary, model_name='Model'):
    logging.info(f'{model_name}: Params={summary["total_parameters"]:,} Sparsity={summary["sparsity_percentage"]:.2f}% FLOPs={summary["flops"]/1e6:.2f}M Time={summary["inference_time_ms"]:.2f}ms')


def compare_models(summaries_dict):
    logging.info(f'{"Model":<20} {"Params":>15} {"Sparsity":>10} {"FLOPs(M)":>12} {"Time(ms)":>10}')
    logging.info('=' * 70)
    for name, summary in summaries_dict.items():
        logging.info(f'{name:<20} {summary["total_parameters"]:>15,} {summary["sparsity_percentage"]:>9.2f}% {summary["flops"]/1e6:>11.2f} {summary["inference_time_ms"]:>9.2f}')
