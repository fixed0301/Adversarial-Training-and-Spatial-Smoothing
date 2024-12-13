import torch
import numpy as np
from tqdm import tqdm

import sys

from src.metrics.plot import plot_metrics_acc_batch, plot_metrics_loss_batch
from src.metrics import acc


def test(model, val_batches, device, path, modelname):
    torch.cuda.empty_cache()
    val_loss, accuracy = 0, 0
    loss_func = torch.nn.CrossEntropyLoss().to(device)

    loss_per_batch = []
    acc_per_batch = []

    model_point = torch.load(path, map_location=device)
    model = model
    model.load_state_dict(model_point["state_dict"])
    model.eval()
    with torch.no_grad():
        val_loss, val_loss_batch = 0, 0
        accuracy, accuracy_batch = 0, 0
        for i, (images, labels) in tqdm(enumerate(val_batches, 0)):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).to(device)
            loss = loss_func(predictions, labels)
            val_loss_batch = loss
            accuracy_batch = acc.accuracy_fn(y_true=labels, y_pred=predictions.argmax(dim=1))
            # Save losses & plot
            loss_per_batch.append(val_loss_batch.item())
            plot_metrics_loss_batch(modelname, loss_per_batch, False)
            acc_per_batch.append(accuracy_batch)
            plot_metrics_acc_batch(modelname, acc_per_batch, False)

            val_loss += loss
            accuracy += acc.accuracy_fn(y_true=labels, y_pred=predictions.argmax(dim=1))

        val_loss /= len(val_batches)
        accuracy /= len(val_batches)

        print("Loss: ", val_loss, "Acc: ", accuracy)

        Path_checkpoint = "../src/model/metrics/" + modelname + "test_Checkpoint.pth"

        checkpoint = {
            "Loss": val_loss,
            "Acc": accuracy,
        }
        torch.save(checkpoint, Path_checkpoint)
    return {"Loss: ": val_loss.item(), "Acc.: ": accuracy}