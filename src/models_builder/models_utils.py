import torch
from torch_geometric.nn import MessagePassing


def apply_message_gradient_capture(layer, name):
    """
    # Example how get Tensors
    # for name, layer in self.gnn.named_children():
    #     if isinstance(layer, MessagePassing):
    #         print(f"{name}: {layer.get_message_gradients()}")
    """
    original_message = layer.message
    layer.message_gradients = {}

    def capture_message_gradients(x_j, *args, **kwargs):
        x_j = x_j.requires_grad_()
        if not layer.training:
            return original_message(x_j=x_j, *args, **kwargs)

        def save_message_grad(grad):
            layer.message_gradients[name] = grad.detach()
        x_j.register_hook(save_message_grad)
        return original_message(x_j=x_j, *args, **kwargs)
    layer.message = capture_message_gradients

    def get_message_gradients():
        return layer.message_gradients
    layer.get_message_gradients = get_message_gradients


def apply_decorator_to_graph_layers(model):
    # TODO Kirill add more options
    """
    Example how use this def
    apply_decorator_to_graph_layers(gnn)
    """
    for name, layer in model.named_children():
        if isinstance(layer, MessagePassing):
            apply_message_gradient_capture(layer, name)
        elif isinstance(layer, torch.nn.Module):
            apply_decorator_to_graph_layers(layer)

