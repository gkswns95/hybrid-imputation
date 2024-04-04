import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SoccerDataset, NBAdataset, NFLdataset
from models import load_model
from models.baselines.nrtsi.nrtsi_utils import get_gap_lr_bs
from models.utils import (
    get_params_str,
    num_trainable_params,
    load_pretrained_model,
)

# from torch.utils.tensorboard import SummaryWriter

# Helper functions
def printlog(line):
    print(line)
    with open(save_path + "/log.txt", "a") as file:
        file.write(line + "\n")

def loss_str(losses: dict):
    loss_dict = [(key,losses[key]) for key in losses.keys() if key.endswith("_loss")]
    dist_dict = [(key,losses[key]) for key in losses.keys() if key.endswith("_dist")]
    acc_dict  = [(key,losses[key]) for key in losses.keys() if key.endswith("accuracy")]

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
def run_epoch(model: nn.DataParallel, optimizer: torch.optim.Adam, epoch: int, train=False, print_every=50, train_args=None):
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()

    loader = train_loader if train else test_loader
    n_batches = len(loader)

    loss_keys = ["total"]
    pred_keys = ["pred"]
    if model.module.params["model"] == "ours":
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
        if model.module.params["model"] in ["ours", "brits", "naomi", "graphimputer"]:
            if train:
                out = model(data, device=default_device)
            else:
                with torch.no_grad():
                    if model.module.params["model"] != "graphimputer":
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
parser.add_argument("--dataset", type=str, required=False, default="soccer", help="soccer or basketball or football")
parser.add_argument("--model", type=str, required=True, default="ours")
parser.add_argument("--target_type", type=str, required=False, default="imputation", help="imputation")
parser.add_argument("--missing_pattern", type=str, required=False, default="camera_simulate", help="all_player or player_wise or camera_simulate")
parser.add_argument("--flip_pitch", action="store_true", default=False, help="augment data by flipping the pitch") 
parser.add_argument("--n_features", type=int, required=False, default=2, help="num features")

parser.add_argument("--train_metrica", action="store_true", default=False, help="training on metrica data")
parser.add_argument("--valid_metrica", action="store_true", default=False, help="validating on metrica data")
parser.add_argument("--train_nba", action="store_true", default=False, help="training on NBA data")
parser.add_argument("--valid_nba", action="store_true", default=False, help="validating on NBA data")
parser.add_argument("--train_nfl", action="store_true", default=False, help="training on NFL data")
parser.add_argument("--valid_nfl", action="store_true", default=False, help="validating on NFL data")

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
parser.add_argument("--cont", action="store_true", default=False, help="continue training previous best model")
parser.add_argument("--best_loss", type=float, required=False, default=0, help="best test loss")
parser.add_argument("--normalize", action="store_true", default=False, help="normalize data")
parser.add_argument("--load_saved", action="store_true", default=False, help="load saved data")
parser.add_argument("--save_new", action="store_true", default=False, help="save prcessed data")
parser.add_argument("--load_pre_train", type=int, default=0, help="Load pre-trained model")
parser.add_argument("--freeze_pre_trained", action="store_true", default=0, help="Freeze pre-trained parameters")
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    args.cuda = torch.cuda.is_available()
    default_device = "cuda:0"

    # Parameters to save
    params = {
        "trial": args.trial,
        "dataset": args.dataset,
        "n_epochs": args.n_epochs,
        "dataset": args.dataset,
        "model": args.model,
        "target_type": args.target_type,
        "flip_pitch": args.flip_pitch,
        "n_features": args.n_features,
        "train_metrica": args.train_metrica,
        "valid_metrica": args.valid_metrica,
        "train_nba": args.train_nba,
        "valid_nba": args.valid_nba,
        "train_nfl" : args.train_nfl,
        "valid_nfl" : args.valid_nfl,
        "batch_size": args.batch_size,
        "start_lr": args.start_lr,
        "min_lr": args.min_lr,
        "seed": args.seed,
        "cuda": args.cuda,
        "best_loss": args.best_loss,
        "normalize": args.normalize,
        "load_saved": args.load_saved,
        "save_new": args.save_new,
        "load_pre_train": args.load_pre_train,
        "freeze_pre_trained":  args.freeze_pre_trained,
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
    if params["load_pre_train"]:
        model = load_pretrained_model(model, params, freeze=params["freeze_pre_trained"], trial_num=params["load_pre_train"])
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
        state_dict = torch.load("{}/model/{}_state_dict_best.pt".format(save_path, args.model), map_location=default_device)
        print("{}/model/{}_state_dict_best.pt".format(save_path, args.model))
        model.module.load_state_dict(state_dict)
    else:
        title = f"{args.trial} {args.target_type} | {args.model}"
        print_keys = ["flip_pitch", "n_features", "batch_size", "start_lr"]

        printlog(title)
        printlog(model.module.params_str)
        printlog(get_params_str(print_keys, model.module.params))
        printlog("n_params {:,}".format(params["total_params"]))
    printlog("############################################################")

    print()
    print("Generating datasets...")

    # Soccer datasets (Metrica)
    metrica_files = ["match1.csv", "match2.csv", "match3_valid.csv"]
    metrica_paths = [f"data/metrica_traces/{f}" for f in metrica_files]

    # Basketball dataset (NBA)
    nba_files = os.listdir("data/nba_traces")
    nba_paths = [f"data/nba_traces/{f}" for f in nba_files]
    nba_paths.sort()

    # Football datasets (NFL)
    nfl_files = os.listdir("data/nfl_traces")
    nfl_paths = ["data/nfl_traces/nfl_train.csv", "data/nfl_traces/nfl_test.csv"]
    
    assert args.train_metrica or args.train_nba or args.train_nfl
    train_paths = []

    if args.train_metrica:
        train_paths += metrica_paths[:-1]
    if args.train_nba:
        train_paths += nba_paths[:70]
    if args.train_nfl:
        train_paths += nfl_paths[:1]

    assert args.valid_metrica or args.valid_nba or args.valid_nfl
    valid_paths = []

    if args.valid_metrica:
        valid_paths += metrica_paths[-1:]
    if args.valid_nba:
        valid_paths += nba_paths[70:90]
    if args.valid_nfl:
        valid_paths += nfl_paths[1:]

    nw = len(model.device_ids) * 4
    if args.dataset == "soccer":
        dataset_class = SoccerDataset
    elif args.dataset == "basketball":
        dataset_class = NBAdataset
    elif args.dataset == "football":
        dataset_class = NFLdataset
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))
    
    train_dataset = dataset_class(
        data_paths=train_paths,
        target_type=args.target_type,
        train=True,
        load_saved=args.load_saved,
        save_new=args.save_new,
        n_features=args.n_features,
        cartesian_accel=params["cartesian_accel"],
        normalize=args.normalize,
        flip_pitch=args.flip_pitch,
        overlap=True if args.dataset == "soccer" else False,
    )
    test_dataset = dataset_class(
        data_paths=valid_paths,
        target_type=args.target_type,
        train=False,
        load_saved=args.load_saved,
        save_new=args.save_new,
        n_features=args.n_features,
        cartesian_accel=params["cartesian_accel"],
        normalize=args.normalize,
        flip_pitch=True,
        overlap=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    if args.dataset in ["soccer", "football"]:
        test_bs = 1
        # test_bs = args.batch_size
    elif args.dataset == "basketball":
        test_bs = args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=True, num_workers=nw, pin_memory=True)

    # Train loop
    best_test_loss = args.best_loss

    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)
    
    train_data_len = f"Train data len : {len(train_dataset)}"
    test_data_len = f"Test data len : {len(test_dataset)}"
    printlog(train_data_len)
    printlog(test_data_len)

    # writer = SummaryWriter(f"runs/{params['model']}_trial({params['trial']})")
    for e in range(n_epochs):
        epoch = e + 1

        hyperparams = {"pretrain": epoch <= pretrain_time}

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

        # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

        printlog(hyperparams_str(epoch, hyperparams))
        start_time = time.time()

        if args.model == "nrtsi":
            train_args, reset_best_loss, save_model = get_gap_lr_bs(args.dataset, e, args.start_lr, use_ta=1)
        else:
            train_args = None

        train_losses = run_epoch(model, optimizer, epoch, train=True, print_every=print_every, train_args=train_args)
        printlog("Train:\t" + loss_str(train_losses))
        test_losses = run_epoch(model, optimizer, epoch, train=False, train_args=train_args)
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
                path = "{}/model/{}_state_dict_{}.pt".format(
                    save_path, args.model, epoch)
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
        
        else: # for NRTSI model
            if reset_best_loss:
                best_test_loss = 0

            if best_test_loss == 0 or total_test_loss < best_test_loss:
                best_test_loss = total_test_loss

                path = "{}/model/{}_state_dict_best_gap_{}.pt".format(save_path, args.model, train_args[1])
                torch.save(model.module.state_dict(), path)
                printlog(f"########## Best Model(max_gap : {train_args[1]}) ###########")

    printlog("Best Test Loss: {:.4f}".format(best_test_loss))