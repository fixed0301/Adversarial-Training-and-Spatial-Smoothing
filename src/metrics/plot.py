import os
import matplotlib.pyplot as plt


def plot_metrics_loss(modelname, losses):
    path = "../data/outputs/"
    os.makedirs(path, exist_ok=True)
    EPOCH_COUNT = range(1, len(losses) + 1)  # Anzahl der Epochen
    fig = plt.figure(figsize=(10, 5))
    plt.title("Train Loss")
    plt.plot(EPOCH_COUNT, losses, "b-", label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    name = "loss" + modelname + ".png"
    fig.savefig(path + name, dpi=fig.dpi)


def plot_metrics_acc(modelname, acc):
    path = "../data/outputs/"
    os.makedirs(path, exist_ok=True)
    EPOCH_COUNT = range(1, len(acc) + 1)  # Anzahl der Epochen
    fig = plt.figure(figsize=(10, 5))
    plt.title("Train Accuarcy")
    plt.plot(EPOCH_COUNT, acc, "r-", label="acc")
    plt.xlabel("epoch")
    plt.ylabel("accuarcy")
    plt.legend()
    name = "acc" + modelname + ".png"
    fig.savefig(path + name, dpi=fig.dpi)


def plot_metrics_acc_batch(modelname, acc, advattack):
    path = "../data/outputs/"
    os.makedirs(path, exist_ok=True)
    BATCH_COUNT = range(1, len(acc) + 1)  # Anzahl der Batches
    fig = plt.figure(figsize=(10, 5))
    if advattack == False:
        plt.title("Test Accuarcy on Orginal Data")
    else:
        plt.title("Test Accuarcy on Adversarial Data")
    plt.plot(BATCH_COUNT, acc, "r-", label="acc")
    plt.xlabel("batch")
    plt.ylabel("accuarcy")
    plt.legend()
    name = "acc" + modelname + ".png"
    fig.savefig(path + name, dpi=fig.dpi)


def plot_metrics_loss_batch(modelname, losses, advattack):
    path = "../data/outputs/"
    os.makedirs(path, exist_ok=True)
    BATCH_COUNT = range(1, len(losses) + 1)  # Anzahl der Epochen
    fig = plt.figure(figsize=(10, 5))
    if advattack == False:
        plt.title("Test Loss on Orginal Data")
    else:
        plt.title("Test Loss on Adversarial Data")
    plt.plot(BATCH_COUNT, losses, "b-", label="loss")
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.legend()
    name = "loss" + modelname + ".png"
    fig.savefig(path + name, dpi=fig.dpi)


def show_images(e, x, x_adv, save_dir):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1, i].axis("off"), axes[2, i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1, 2, 0)))
        axes[0, i].set_title("Spectogram: Original Data   ")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1, 2, 0)))
        axes[1, i].set_title("Spectogram: Adversarial Example")

    plt.axis("off")
    plt.savefig(os.path.join(save_dir, "result_{}.png".format(e)))
    plt.close('all')