import torch
import numpy as np
from tqdm import tqdm
import os

from src.metrics.plot import plot_metrics_loss, plot_metrics_acc
from src.metrics import acc

def train(model, train_batches, device, path, modelname):
    torch.cuda.empty_cache()
    epochs = 10
    LEARNING_RATE = 0.001
    GRADIENT_MOMENTUM = 0.90
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=GRADIENT_MOMENTUM)
    model.train()

    loss_per_epoch = []
    acc_per_epoch = []

    for epoch in range(epochs):
        # training mode
        model.train()
        with torch.enable_grad():
            train_loss = 0
            accuracy = 0
            for i, (images, labels) in tqdm(enumerate(train_batches,0)):
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).to(device)
                loss = loss_func(predictions, labels)
                train_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accuracy += acc.accuracy_fn(y_true=labels, y_pred=predictions.argmax(dim=1))

            train_loss /= len(train_batches)
            accuracy /= len(train_batches)

        # Save losses & plot
        loss_per_epoch.append(train_loss.item())
        plot_metrics_loss(modelname, loss_per_epoch)
        acc_per_epoch.append(accuracy)
        plot_metrics_acc(modelname, acc_per_epoch)


        print( "Epoch", epoch+1,"/",epochs, "Loss: ",
                    train_loss, "Acc: ",accuracy)

        base_dir = "../src/model/metrics"  # Base directory
        Path_checkpoint = os.path.join(base_dir, f"{modelname}Checkpoint.pth")
        checkpoint = {
            "Loss": loss_per_epoch,
            "Acc": acc_per_epoch,
        }
        torch.save(checkpoint, Path_checkpoint)
    # Save the Modelstate
    torch.save({"state_dict": model.state_dict(),"Loss":train_loss.item(), "Acc.": accuracy}, path)
    return {"Loss: ":train_loss.item(), "Acc.: ": acc}