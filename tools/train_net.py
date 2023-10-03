import os
from statistics import mean

import util
from charnet.loss import CombinedLoss
from charnet.modeling.model import CharNet
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from charnet.dataset import CustomDataset
from charnet.config import cfg
from tqdm import tqdm
import cv2 as cv
import optuna

import config
from charnet.optuna_loss import OptunaLoss


def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg.merge_from_file("configs/myconfig.yaml")
    cfg.freeze()
    # print(cfg)

    model = CharNet(img_size=config.img_size)


    # this is how I can load the default weights to train from
    if config.use_pretrained_weights:
        model.load_state_dict(torch.load(cfg.WEIGHT), strict=False)
    elif config.use_pretrained_backbone:
        model.backbone.load_state_dict(torch.load(cfg.WEIGHT), strict=False)

    model.to(device)

    # lr = trial.suggest_float("lr", 0.0004, 0.0006)
    lr = 0.0005
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_files = os.listdir("example_samples/images")
    num = len(all_files)
    train_num = int(float(num) * config.training_part)
    test_num = num - train_num
    train_files, test_files = random_split(all_files, [train_num, test_num])
    train_dataset = CustomDataset(train_files, "example_samples/images", "example_samples/labels", length=config.img_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataset = CustomDataset(test_files, "example_samples/images", "example_samples/labels", length=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    loss_fn = CombinedLoss(
        word_char=0.5,
        fg_tblro=0.45,
        aabb_theta=0.4,
        bce_dice=0.9
        # word_char=trial.suggest_float("word_char", 0.45, 0.55),
        # fg_tblro=trial.suggest_float("fg_tblro", 0.35, 0.52),
        # aabb_theta=trial.suggest_float("aabb_theta", 0.25, 0.42),
        # bce_dice=trial.suggest_float("bce_dice", 0.85, 0.96)
    )
    optuna_loss_fn = OptunaLoss()

    if config.print_batch_after_epoch:
        util.visualize_batch(model, test_loader, "before", print_unchanged=True)

    optuna_losses = []
    for epoch_idx, epoch in enumerate(range(config.num_epochs)):
        model.train()
        model.backbone.requires_grad_(False)  # Freeze backbone weights
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            # Vorwärtsdurchlauf
            data = data.to(device)
            target = [t.to(device) for t in target]
            outputs = model(data)
            loss = loss_fn(*outputs, *target)
            # print(f"Current optuna loss: {optuna_loss_fn(outputs[2], target[2])}")

            # Rückwärtsdurchlauf und Optimierung
            loss.backward()
            optimizer.step()

        # Ausgabe von Trainingsfortschritt
        # if (batch_idx + 1) % 100 == 0:
        # print(
        #     f'Training - Epoch [{epoch + 1}/{config.num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Training-Loss: {loss.item():.4f}')

        # Validierung nach jeder Epoche
        model.eval()  # Setze das Modell in den Evaluationsmodus
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data = data.to(device)
                targets = [t.to(device) for t in targets]
                # Vorwärtsdurchlauf
                outputs = model(data)
                loss = loss_fn(*outputs, *targets)
                # if epoch_idx == config.num_epochs - 1:
                #     optuna_losses.append(optuna_loss_fn(outputs[2], targets[2]))
                test_loss += loss.item()

                # Genauigkeit berechnen
                # _, predicted = torch.max(outputs.data, 1)
                # total += targets.size(0)
                # correct += (predicted == targets).sum().item()

                if config.print_batch_after_epoch and batch_idx == 0:
                    util.visualize_batch(model, test_loader, f"epoch_{epoch_idx}")

        test_loss /= len(test_loader)
        # accuracy = correct / total

        # print(f'Validation - Epoch [{epoch + 1}/{num_epochs}], Test-Loss: {test_loss:.4f}, Accuracy: {100 * accuracy:.2f}%')
        print(f'Validation - Epoch [{epoch + 1}/{config.num_epochs}], Test-Loss: {test_loss:.4f}')


    print("Training abgeschlossen!")

    # save progress
    if config.save_weights:
        util.save_weights(model, optimizer, epoch_idx)

    # return mean(optuna_losses)


if __name__ == "__main__":
    objective(None)
    # n_trials = 30
    # study = optuna.create_study()
    # study.optimize(objective, n_trials=n_trials)
    # print(f"Best value: {study.best_value} (params: {study.best_params})\n")

    # Currently best (value of 0.95218):
    # lr = 0.0005934
    # word_char = 0.51635
    # fg_tblro = 0.4239598
    # aabb_theta = 0.3260089
    # bce_dice = 0.88975822
