# Import Packages and custom functions

import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset

import sys

from src.metrics.plot import plot_metrics_loss, plot_metrics_acc, show_images
from src.defensemethod.genadvexamples import gen_adv


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

#filename이 안필요하넹
def AdversarialTraining(model, trainloader, device, path, modelname):
    epochs = 10 # 임시로 1
    LEARNING_RATE = 0.001
    GRADIENT_MOMENTUM = 0.90
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=GRADIENT_MOMENTUM)

    loss_per_epoch = []
    acc_per_epoch = []
    model.train()

    for epoch in range(epochs):
        train_loss_normal, train_loss_adv = 0, 0
        train_loss, train_acc_epoch, train_acc = 0, 0, 0
        correct_adv, correct_normal, correct_total = 0, 0, 0

        for i, (input, label) in tqdm(enumerate(trainloader, 0)):
            input, label = input.to(device), label.to(device)

            predictions_normal = model(input).to(device)
            loss_normal = loss_func(predictions_normal, label)
            train_loss_normal += loss_normal

            adv = gen_adv(input, label, model)
            predictions_adv = model(adv).to(device)
            loss_adv = loss_func(predictions_adv, label)
            train_loss_adv += loss_adv

            loss = loss_normal + loss_adv
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # acc
            correct_normal = torch.eq(label, predictions_normal.argmax(dim=1)).sum().item()
            correct_adv = torch.eq(label, predictions_adv.argmax(dim=1)).sum().item()
            correct_total = correct_adv + correct_normal
            train_acc = (correct_total / (len(label) + len(label))) * 100
            train_acc_epoch += train_acc

        train_loss /= (len(trainloader) * 2)
        train_acc_epoch /= (len(trainloader))
        # Save losses & plot
        loss_per_epoch.append(train_loss.item())
        plot_metrics_loss(modelname, loss_per_epoch)
        acc_per_epoch.append(train_acc_epoch)
        plot_metrics_acc(modelname, acc_per_epoch)
        # Save some samples (Normal/original and Adversarial)
        show_images(epoch, input, adv, "../data/outputs/Images")
        print("Epoch", epoch + 1, "/", epochs, "Loss: ",
              train_loss, "Acc: ", train_acc_epoch)

        Path_checkpoint = "../src/model/metrics/" + modelname + "Checkpoint.pth"

        checkpoint = {
            "Loss": loss_per_epoch,
            "Acc": acc_per_epoch,
        }
        torch.save(checkpoint, Path_checkpoint)

    torch.save({"state_dict": model.state_dict(), "Loss": train_loss.item(), "Acc.": train_acc_epoch}, path)
    return {"Loss: ": train_loss.item(), "Acc.: ": train_acc_epoch}