import torch
import numpy as np
#import nvidia_smi

from globals import device


def get_edges(t):
    """
    This function is taken from: https://github.com/NVIDIA/pix2pixHD.
    :param t:
    :return:
    """
    edge = torch.cuda.ByteTensor(t.size()).zero_()
    # comparing with the left pixels
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).type(
        torch.uint8
    )
    # comparing with the right pixels
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).type(
        torch.uint8
    )
    # comparing with the lower pixels
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).type(
        torch.uint8
    )
    # comparing with upper  pixels
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).type(
        torch.uint8
    )
    return edge.float()


def label_to_tensor(label, height, width, count=0):
    if count == 0:
        arr = np.zeros((10, height, width))
        arr[label] = 1

    else:
        arr = np.zeros((count, 10, height, width))
        arr[:, label, :, :] = 1

    return torch.from_numpy(arr.astype(np.float32))


def save_checkpoint(path_to_save, optim_step, model, optimizer, loss, lr, additional_info=None):
    """
    Save a model checkpoint including state dicts, current loss, learning rate and additional info.
    
    Args:
        path_to_save: Directory to save the checkpoint
        optim_step: Current optimization step
        model: The model to save
        optimizer: The optimizer to save
        loss: Current loss value
        lr: Current learning rate
        additional_info: Optional dict containing additional information to save
    """
    name = path_to_save + f"/optim_step={optim_step}.pt"
    checkpoint = {
        "loss": loss,
        "lr": lr,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    # Add additional info if provided
    if additional_info is not None:
        checkpoint.update(additional_info)

    torch.save(checkpoint, name)
    print(f'In [save_checkpoint]: save state dict done at: "{name}"')


def load_checkpoint(path_to_load, optim_step, model, optimizer, resume_train=True):
    """
    Load a model checkpoint including state dicts, loss, learning rate and any additional info.
    
    Args:
        path_to_load: Directory to load the checkpoint from
        optim_step: Optimization step to load
        model: The model to load into
        optimizer: The optimizer to load into
        resume_train: Whether to put model in train mode
    """
    name = path_to_load + f"/optim_step={optim_step}.pt"
    checkpoint = torch.load(name, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    loss = checkpoint["loss"]
    lr = checkpoint.get("lr", None)  # backward compatibility
    
    # Extract any additional info if present
    additional_info = {k: v for k, v in checkpoint.items() 
                      if k not in ["model_state_dict", "optimizer_state_dict", "loss", "lr"]}

    print(f'In [load_checkpoint]: load state dict done from: "{name}"')

    # Set model mode
    if resume_train:
        model.train()
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    if optimizer is not None:
        return model.to(device), optimizer, loss, lr, additional_info
    return model.to(device), None, loss, lr, additional_info


def show_memory_usage():
    # nvidia_smi.nvmlInit()
    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # GPU number
    # mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # print('=' * 50)
    # print(f'mem: {mem_res.used / (1024 ** 3)} (GiB)')  # usage in GiB
    # print(f"mem usage: {100 * (mem_res.used / mem_res.total):.3f}%")  # percentage
    # print('=' * 50)
    pass


def tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()
