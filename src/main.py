import os

# custom moduls
from utils.utils import get_device
from model.model import Mymodel
from training.train import train
from dataloader import dataloadernew
from testing.testing import test
from testing.attacks import adversarialattack
from defensemethod.adversarialtraining import AdversarialTraining
from defensemethod.spatialsmoothing import spatialsmoothingTest, spatialsmoothingTest_withoutattack
from utils import utils
from utils.config import config

# packages
import torch
import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(config.seed)


# Support function for clearing terminal output
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


# Project main
def project_main(device, mymodel):
    print("##########")
    print(
        "Abwehr von State-of-the-Art Black-Box Adversarial Attacks auf Audio Deepfake Detektionmodelle mittels Adversarial Training und Spatial Smoothing​")
    print("##########")

    print("\n")
    print("Models")
    print("0. All Models (Training & Testing)")
    print("1. ResNet18 (Baseline)")
    print("2. ResNet50 (Baseline) ")
    print("3. Adversarial Attack on ResNet18 and ResNet50")
    print("4. Spatial Smoothing - ResNet18 and ResNet50")
    print("5. Adversarial Training - ResNet18 and ResNet50")
    print("6. Adversarial Training and Spatial Smoothing - ResNet18 and  ResNet50")
    print("##########")
    print("7. Call Checkpoints (Metrics - Losses and Acc) of all models")
    print("##########")

    user_input = int(input("Model:"))
    if user_input == 0:
        cls()

        print("### Loading Dataset ### ")
        train_batches, val_batches = dataloadernew.dataset(device)

        #print("\n")
        #print("### Training model ###")
        path_model = "../src/model/hagishiro.pth"
        modelname = "_hagishiro_"
        #train(mymodel, train_batches, device, path_model, modelname)

        #print("\n")
        #print("### Testing model ###")
        #print("### Normal Testing on model ###")
        modelname_test = "_hagishiro_test_"
        #test(mymodel, val_batches, device, path_model, modelname_test)

        #print("\n")
        #rint("### Adversarial Attack ###")
        modelname_test_aa = "_hagishiro_test_aa_"
        #adversarialattack(path_model, val_batches, device, mymodel, modelname_test_aa)

        # Defense Method
        #print("\n")
        #print("### Spatial Smoothing ###")
        modelname_test_smoothing = "_hagishiro_smoothing_"
        #spatialsmoothingTest(path_model, val_batches, device, mymodel, modelname_test_smoothing)

        print("\n")
        print("### Adversarial Training ###")
        path_adv_train = "../src/model/hagishiro_adv_train22.pth"
        modelname_adv_train = "_hagishiro_adv_train22_"
        AdversarialTraining(mymodel, train_batches, device, path_adv_train, modelname_adv_train)

        print("\n")
        print("### Ad Testing ###")
        print("### Testing Adv model with original data ###")
        modelname_at_test = "_hagishiro_at_test_"
        test(mymodel, val_batches, device, path_adv_train, modelname_at_test)

        print("\n")
        print("### Testing Adv model with Adversarial Attack ###")
        modelname_test_at_aa = "_hagishiro_test_at_aa_"
        adversarialattack(path_adv_train, val_batches, device, mymodel, modelname_test_at_aa)

        # Defense Method
        print("\n")
        print("### Combine Adversarial Training with Spatial Smoothing ###")
        modelname_resne18_test_at_smoothing = "_hagishiro_at_smoothing_"
        spatialsmoothingTest(path_adv_train, val_batches, device, mymodel,
                             modelname_resne18_test_at_smoothing)
'''
    elif user_input == 7:
        cls()
        print("### !!!MAKE SURE THAT TRAINING STATE IS SAVED!!! - State of Metrics ###")

        print("Trainingscheckpoints")

        print("### ResNet18 ###")
        modelname_resnet18 = "_resnet18_"

        output = torch.load("./model/metrics/" + modelname_resnet18 + "Checkpoint.pth")
        print(output)

        print("### ResNet50 ###")
        modelname_resnet50 = "_resnet50_"
        output = torch.load("./model/metrics/" + modelname_resnet50 + "Checkpoint.pth")
        print(output)

        print("### ResNet18 with Adversarial Training ###")
        modelname_resnet18_adv_train = "_resnet18_adv_train_"
        output = torch.load("./model/metrics/" + modelname_resnet18_adv_train + "Checkpoint.pth")
        print(output)

        print("### ResNet50 with Adversarial Training ###")
        modelname_resnet50_adv_train = "_resnet50_adv_train_"
        output = torch.load("./model/metrics/" + modelname_resnet50_adv_train + "Checkpoint.pth")
        print(output)

        print("Testcheckpoints")
        print("### ResNet18 ###")
        modelname_resnet18_test = "_resnet18_test_"

        output = torch.load("./model/metrics/" + modelname_resnet18_test + "test_Checkpoint.pth", map_location='cpu')
        print(output)

        print("### ResNet50 ###")
        modelname_resnet50_test = "_resnet50_test_"

        output = torch.load("./model/metrics/" + modelname_resnet50_test + "test_Checkpoint.pth", map_location='cpu')
        print(output)

        print("### ResNet18 - Adversarial Attack ###")

        output = torch.load("./model/metrics/_resnet18_test_aa_test_with_attack_Checkpoint.pth", map_location='cpu')
        print(output)

        print("### ResNet50 - Adversarial Attack ###")
        modelname_resnet50_test_aa = "_resnet50_test_aa_"

        output = torch.load("./model/metrics/_resnet50_test_aa_test_with_attack_Checkpoint.pth", map_location='cpu')
        print(output)

        print("### ResNet18 - Spatial Smoothing with Adversarial Attack ###")
        modelname_resne18_test_smoothing = "_resnet18_smoothing_"

        output = torch.load("./model/metrics/" + modelname_resne18_test_smoothing + "_test_Checkpoint.pth",
                            map_location='cpu')
        print(output)

        print("### ResNet18 - Spatial Smoothing without Adversarial Attack ###")
        modelname_resne18_test_smoothing_withoutattack = "_resnet18_smoothing_woattack"
        output = torch.load(
            "./model/metrics/" + modelname_resne18_test_smoothing_withoutattack + "_test_Checkpoint.pth",
            map_location='cpu')
        print(output)

        print("### ResNet50 - Spatial Smoothing with Adversarial Attack ###")
        modelname_resne50_test_smoothing = "_resnet50_smoothing_"
        output = torch.load("./model/metrics/" + modelname_resne50_test_smoothing + "_test_Checkpoint.pth",
                            map_location='cpu')
        print(output)

        print("### ResNet50 - Spatial Smoothing without Adversarial Attack ###")
        modelname_resne50_test_smoothing_withoutattack = "_resnet50_smoothing_woattack"
        output = torch.load(
            "./model/metrics/" + modelname_resne50_test_smoothing_withoutattack + "_test_Checkpoint.pth",
            map_location='cpu')
        print(output)

        print("### ResNet18 with Adversarial Training with Adversarial Attack ###")
        output = torch.load("./model/metrics/_resnet18_test_at_aa_test_with_attack_Checkpoint.pth", map_location='cpu')
        print(output)

        print("### ResNet18 with Adversarial Training with original data ###")
        modelname_resnet18_at_test = "_resnet18_at_test_"
        output = torch.load("./model/metrics/" + modelname_resnet18_at_test + "test_Checkpoint.pth", map_location='cpu')
        print(output)

        print("### ResNet50 with Adversarial Training with Adversarial Attack ###")
        output = torch.load("./model/metrics/_resnet50_test_at_aa_test_with_attack_Checkpoint.pth", map_location='cpu')
        print(output)

        print("### ResNet50 with Adversarial Training with original data ###")
        modelname_resnet50_at_test = "_resnet50_at_test_"
        output = torch.load("./model/metrics/" + modelname_resnet50_at_test + "test_Checkpoint.pth", map_location='cpu')
        print(output)

        print("### ResNet18 with Adversarial Training + Spatial Smoothing with Adversarial Attack ###")
        modelname_resne18_test_at_smoothing = "_resnet18_at_smoothing_"
        output = torch.load("./model/metrics/" + modelname_resne18_test_at_smoothing + "_test_Checkpoint.pth",
                            map_location='cpu')
        print(output)

        print("### ResNet18 with Adversarial Training + Spatial Smoothing with original data ###")
        modelname_resne18_test_at_smoothing_woattack = "_resnet18_at_smoothing_woattack"
        output = torch.load("./model/metrics/" + modelname_resne18_test_at_smoothing_woattack + "_test_Checkpoint.pth",
                            map_location='cpu')
        print(output)

        print("### ResNet50 with Adversarial Training + Spatial Smoothing with Adversarial Attack ###")
        modelname_resnet50_test_at_smoothing = "_resnet50_at_smoothing_"
        output = torch.load("./model/metrics/" + modelname_resnet50_test_at_smoothing + "_test_Checkpoint.pth",
                            map_location='cpu')
        print(output)

        print("### ResNet50 with Adversarial Training + Spatial Smoothing with original data ###")

        modelname_resne50_test_at_smoothing_woattack = "_resnet50_at_smoothing_woattack"
        output = torch.load("./model/metrics/" + modelname_resne50_test_at_smoothing_woattack + "_test_Checkpoint.pth",
                            map_location='cpu')
        print(output)

'''
device = get_device()
mymodel = Mymodel().to(device)
project_main(device, mymodel)