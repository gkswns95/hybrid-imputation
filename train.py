import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SportsDataset
from models import load_model
from models.nrtsi.utils import get_gap_lr_bs
from models.utils import get_params_str, load_pretrained_model, num_trainable_params

# from torch.utils.tensorboard import SummaryWriter


# Helper functions
def printlog(line):
    print(line)
    with open(save_path + "/log.txt", "a") as file:
        file.write(line + "\n")


def loss_str(losses: dict):
    loss_dict = [(key, losses[key]) for key in losses.keys() if key.endswith("_loss")]
    dist_dict = [(key, losses[key]) for key in losses.keys() if key.endswith("_dist")]
    acc_dict = [(key, losses[key]) for key in losses.keys() if key.endswith("accuracy")]

    ret = "\n---------------Losses---------------\n"
    for key, value in loss_dict:
        ret += " {}: {:.4f} |".format(key, np.mean(value))

    if len(dist_dict):
        ret += "\n---------------Dists---------------\n"
        for key, value in dist_dict:
            ret += " {}: {:.4f} |".format(key, np.mean(value))
    if len(acc_dict):
        ret += "\n---------------Acc---------------\n"
        for key, value in acc_dict:
            ret += " {}: {:.4f} |".format(key, np.mean(value))

    return ret[:-2]


def hyperparams_str(epoch, hp):
    ret = "\nEpoch {:d}".format(epoch)
    if hp["pretrain"]:
        ret += " (pretrain)"
    return ret


# For one epoch
def run_epoch(model: nn.DataParallel, optimizer: torch.optim.Adam, train=False, print_every=50, train_args=None):
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()

    loader = train_loader if train else valid_loader
    n_batches = len(loader)

    loss_keys = ["total"]
    pred_keys = ["pred"]
    if model.module.params["model"] == "dbhp":
        if model.module.params["n_features"] == 6:
            loss_keys += ["xy", "vel", "accel"]
        if model.module.params["physics_loss"]:
            pred_keys += ["physics_f", "physics_b"]
            if model.module.params["train_hybrid"]:
                pred_keys += ["train_hybrid"]

        loss_keys += pred_keys

    loss_dict = {f"{key}_loss": [] for key in loss_keys if key != "pred"}
    dist_dict = {f"{key}_dist": [] for key in pred_keys}

    for batch_idx, data in enumerate(loader):
        if model.module.params["model"] in ["dbhp", "brits", "naomi", "graph_imputer"]:
            if train:
                out = model(data, device=default_device)
            else:
                with torch.no_grad():
                    if model.module.params["model"] != "graph_imputer":
                        out = model(data, device=default_device)
                    else:
                        out = model.module.evaluate(data, device=default_device)

        if model.module.params["model"] == "nrtsi":
            min_gap, max_gap, cur_lr, ta = train_args
            data = data[:2]

            data.append(min_gap)
            data.append(max_gap)

            optimizer.param_groups[0]["lr"] = cur_lr
            if train:
                out = model(data, model=model, teacher_forcing=ta, device=default_device)
            else:
                with torch.no_grad():
                    out = model(data, model=model, teacher_forcing=ta, device=default_device)

        loss = out["total_loss"]

        for l_key in loss_dict:
            loss_dict[l_key] += [out[l_key].item()]

        for d_key in dist_dict:
            dist_dict[d_key] += [out[d_key].item()]

        loss_dict.update(dist_dict)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), clip)
            optimizer.step()

        if train and batch_idx % print_every == 0:
            print(f"[{batch_idx:>{len(str(n_batches))}d}/{n_batches}]  {loss_str(loss_dict)}")

    for key, value in loss_dict.items():
        loss_dict[key] = np.mean(value)

    return loss_dict


# Main starts here
parser = argparse.ArgumentParser()

parser.add_argument("-t", "--trial", type=int, required=True)
parser.add_argument("--dataset", type=str, required=True, default="soccer", help="soccer, basketball, afootball")
parser.add_argument("--model", type=str, required=True, default="dbhp")
parser.add_argument("--missing_pattern", type=str, required=False, default="camera", help="uniform, indep, camera")
parser.add_argument("--n_features", type=int, required=False, default=2, help="num features")
parser.add_argument("--normalize", action="store_true", default=False, help="normalize data")
parser.add_argument("--flip_pitch", action="store_true", default=False, help="augment data by flipping the pitch")

parser.add_argument("--n_epochs", type=int, required=False, default=200, help="num epochs")
parser.add_argument("--batch_size", type=int, required=False, default=32, help="batch size")
parser.add_argument("--start_lr", type=float, required=False, default=0.0001, help="starting learning rate")
parser.add_argument("--min_lr", type=float, required=False, default=0.0001, help="minimum learning rate")
parser.add_argument("--clip", type=float, required=False, default=10, help="gradient clipping")
parser.add_argument("--print_every_batch", type=int, required=False, default=50, help="periodically print performance")
parser.add_argument("--save_every_epoch", type=int, required=False, default=10, help="periodically save model")
parser.add_argument("--pretrain_time", type=int, required=False, default=0, help="num epochs to train macro policy")
parser.add_argument("--seed", type=int, required=False, default=128, help="PyTorch random seed")
parser.add_argument("--cuda", action="store_true", default=False, help="use GPU")
parser.add_argument("--cont", action="store_true", default=False, help="continue training the previous bast model")
parser.add_argument("--best_loss", type=float, required=False, default=0, help="best test loss")
parser.add_argument("--load_pretrained", type=int, default=0)
parser.add_argument("--freeze_pretrained", action="store_true", default=0)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    args.cuda = torch.cuda.is_available()
    default_device = "cuda:0"

    # Parameters to save
    params = {
        "trial": args.trial,
        "dataset": args.dataset,
        "n_epochs": args.n_epochs,
        "model": args.model,
        "n_features": args.n_features,
        "normalize": args.normalize,
        "flip_pitch": args.flip_pitch,
        "batch_size": args.batch_size,
        "start_lr": args.start_lr,
        "min_lr": args.min_lr,
        "seed": args.seed,
        "cuda": args.cuda,
        "best_loss": args.best_loss,
        "load_pretrained": args.load_pretrained,
        "freeze_pretrained": args.freeze_pretrained,
    }

    # Hyperparameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    clip = args.clip
    print_every = args.print_every_batch
    save_every = args.save_every_epoch
    pretrain_time = args.pretrain_time
    # Set manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load model
    model = load_model(args.model, params, parser).to(default_device)
    if args.load_pretrained:
        model = load_pretrained_model(model, params, freeze=args.load_pretrained)
    model = nn.DataParallel(model, device_ids=[0])

    # Update params with model parameters
    params = model.module.params
    params["total_params"] = num_trainable_params(model)

    # Create save path and saving parameters
    save_path = "saved/{:03d}".format(args.trial)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "/model")
    with open(f"{save_path}/params.json", "w") as f:
        json.dump(params, f, indent=4)

    # Continue a previous experiment, or start a new one
    if args.cont:
        state_dict = torch.load(
            "{}/model/{}_state_dict_best.pt".format(save_path, args.model),
            map_location=default_device,
        )
        print("{}/model/{}_state_dict_best.pt".format(save_path, args.model))
        model.module.load_state_dict(state_dict)
    else:
        title = f"{args.trial} | {args.model}"
        print_keys = ["flip_pitch", "n_features", "batch_size", "start_lr"]

        printlog(title)
        printlog(model.module.params_str)
        printlog(get_params_str(print_keys, model.module.params))
        printlog("n_params {:,}".format(params["total_params"]))
    printlog("############################################################")

    print()
    print("Generating datasets...")

    if args.dataset == "soccer":  # Soccer (Metrica) datasets
        metrica_files = ["match1.csv", "match2.csv", "match3_valid.csv"]
        metrica_paths = [f"data/metrica_traces/{f}" for f in metrica_files]
        train_paths = metrica_paths[:-1]
        valid_paths = metrica_paths[-1:]

        window_size = 200
        episode_min_len = 100
        train_stride = 5

    elif args.dataset == "basketball":  # Basketball (NBA) dataset
        nba_files = os.listdir("data/nba_traces")
        nba_paths = [f"data/nba_traces/{f}" for f in nba_files]
        nba_paths.sort()
        train_paths = nba_paths[:70]
        valid_paths = nba_paths[70:90]

        window_size = 200
        episode_min_len = 100
        train_stride = 200

    else:  # American football (NFL) dataset
        nfl_paths = ["data/nfl_traces/nfl_train.csv", "data/nfl_traces/nfl_test.csv"]
        train_paths = nfl_paths[:-1]
        valid_paths = nfl_paths[-1:]

        window_size = 50
        episode_min_len = 50
        train_stride = 5

    train_dataset = SportsDataset(
        sports=args.dataset,
        data_paths=train_paths,
        n_features=args.n_features,
        window_size=window_size,
        episode_min_len=episode_min_len,
        stride=train_stride,
        normalize=args.normalize,
        flip_pitch=args.flip_pitch,
    )
    valid_dataset = SportsDataset(
        sports=args.dataset,
        data_paths=valid_paths,
        n_features=args.n_features,
        window_size=window_size,
        episode_min_len=episode_min_len,
        stride=window_size,
        normalize=args.normalize,
        flip_pitch=False,
    )

    n_workers = len(model.device_ids) * 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1 if args.dataset != "basketball" else args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )

    # Train loop
    best_test_loss = args.best_loss

    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)

    train_data_len = f"Train data len : {len(train_dataset)}"
    test_data_len = f"Test data len : {len(valid_dataset)}"
    printlog(train_data_len)
    printlog(test_data_len)

    # writer = SummaryWriter(f"runs/{params['model']}_trial({params['trial']})")
    for e in range(n_epochs):
        epoch = e + 1

        hyperparams = {"pretrain": epoch <= pretrain_time}

        # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

        printlog(hyperparams_str(epoch, hyperparams))
        start_time = time.time()

        if args.model == "nrtsi":
            train_args, reset_best_loss, save_model = get_gap_lr_bs(args.dataset, e, args.start_lr, use_ta=1)
        else:
            train_args = None

        # Set a custom learning rate schedule
        if args.model != "NRTSI":
            if epochs_since_best == 2 and lr > args.min_lr:
                # Load previous best model
                path = "{}/model/{}_state_dict_best.pt".format(save_path, args.model)
                if args.model == "NRTSI":
                    path = "{}/model/{}_state_dict_best_gap_{}.pt".format(save_path, args.model, train_args[1])

                if epoch <= pretrain_time:
                    path = "{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model)
                    if args.model == "NRTSI":
                        path = "{}/model/{}_state_dict_best_gap_{}.pt".format(save_path, args.model, train_args[1])
                state_dict = torch.load(path)

                # Decrease learning rate
                lr = max(lr / 2, args.min_lr)
                printlog("########## lr {} ##########".format(lr))
                epochs_since_best = 0
            else:
                epochs_since_best += 1

        train_losses = run_epoch(model, optimizer, train=True, print_every=print_every, train_args=train_args)
        printlog("Train:\t" + loss_str(train_losses))

        test_losses = run_epoch(model, optimizer, train=False, train_args=train_args)
        printlog("Test:\t" + loss_str(test_losses))

        # Write learning curve on tensorboard
        # for key in train_losses.keys():
        #     key_str = "Loss" if key.endswith("loss") else "Dist"
        #     writer.add_scalar(f"{key_str}/train_{key}", train_losses[key], epoch)
        # for key in test_losses.keys():
        #     key_str = "Loss" if key.endswith("loss") else "Dist"
        #     writer.add_scalar(f"{key_str}/valid_{key}", test_losses[key], epoch)

        epoch_time = time.time() - start_time
        printlog("Time:\t {:.2f}s".format(epoch_time))

        total_test_loss = sum([value for key, value in test_losses.items() if key.endswith("loss")])
        # Best model on test set
        if args.model != "nrtsi":
            if best_test_loss == 0 or total_test_loss < best_test_loss:
                best_test_loss = total_test_loss
                epochs_since_best = 0

                path = "{}/model/{}_state_dict_best.pt".format(save_path, args.model)
                if epoch <= pretrain_time:
                    path = "{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model)

                torch.save(model.module.state_dict(), path)
                printlog("########## Best Model ###########")

            # Periodically save model
            if epoch % save_every == 0:
                path = "{}/model/{}_state_dict_{}.pt".format(save_path, args.model, epoch)
                torch.save(model.module.state_dict(), path)
                printlog("########## Saved Model ##########")

            # End of pretrain stage
            if epoch == pretrain_time:
                printlog("######### End Pretrain ##########")
                best_test_loss = 0
                epochs_since_best = 0
                lr = max(args.start_lr, args.min_lr)

                state_dict = torch.load("{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model))
                model.module.load_state_dict(state_dict)

                test_losses = run_epoch(model, optimizer, train=False)
                printlog("Test:\t" + loss_str(test_losses))

        else:  # for NRTSI model
            if reset_best_loss:
                best_test_loss = 0

            if best_test_loss == 0 or total_test_loss < best_test_loss:
                best_test_loss = total_test_loss

                path = "{}/model/{}_state_dict_best_gap_{}.pt".format(save_path, args.model, train_args[1])
                torch.save(model.module.state_dict(), path)
                printlog(f"########## Best Model (max_gap : {train_args[1]}) ###########")

    printlog("Best Test Loss: {:.4f}".format(best_test_loss))
