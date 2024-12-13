import torch
import torchattacks
from tqdm import tqdm

import sys

from src.metrics.plot import plot_metrics_acc_batch, plot_metrics_loss_batch

from src.metrics import acc

# Calculates accuracy between truth labels and predictions.
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def adversarialattack(model_path, testloader, device, model, modelname):
    torch.cuda.empty_cache()

    testloader = testloader
    model_point = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.load_state_dict(model_point["state_dict"])
    loss_func = torch.nn.CrossEntropyLoss().to(device)

    val_loss, val_accuracy = 0, 0
    val_accuracy_batch, val_loss_batch = 0, 0

    loss_per_batch = []
    acc_per_batch = []

    model.eval()
    for i, (images, labels) in tqdm(enumerate(testloader, 0)):
        images, labels = images.to(device), labels.to(device)

        attack = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50)
        adv_images = attack(images, labels)

        with torch.no_grad():
            predictions_adv = model(adv_images).to(device)

        val_loss_batch = loss_func(predictions_adv, labels)
        val_accuracy_batch = accuracy_fn(y_true=labels, y_pred=predictions_adv.argmax(dim=1))

        # Save losses, accuarcy & plot
        loss_per_batch.append(val_loss_batch.item())
        plot_metrics_loss_batch(modelname, loss_per_batch, True)
        acc_per_batch.append(val_accuracy_batch)
        plot_metrics_acc_batch(modelname, acc_per_batch, True)

        val_loss += loss_func(predictions_adv, labels)
        val_accuracy += accuracy_fn(y_true=labels, y_pred=predictions_adv.argmax(dim=1))

    val_loss /= len(testloader)
    val_accuracy /= len(testloader)
    print("Loss: ", val_loss, "Acc: ", val_accuracy)
    Path_checkpoint = "../src/model/metrics/" + modelname + "test_with_attack_Checkpoint.pth"

    checkpoint = {
        "Loss": val_loss,
        "Acc": val_accuracy,
    }
    torch.save(checkpoint, Path_checkpoint)
    return {"Loss: ": val_loss.item(), "Acc.: ": val_accuracy}