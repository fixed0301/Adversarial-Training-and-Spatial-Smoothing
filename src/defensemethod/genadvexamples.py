import torch
import torchattacks
import copy

def gen_adv(input, label , model):
    model = model
    model_cp = copy.deepcopy(model) # Copy the Model because to not change the params
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    attack = torchattacks.Pixle(model_cp, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50)
    adv_input = attack(input , label)
    return adv_input