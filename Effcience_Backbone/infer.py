import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import shutil
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

import model.model as module_arch
from parse_config import ConfigParser
import data_loader.data_loaders as module_data

def main(config):
    logger = config.get_logger("INFERENCE")

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        config["data_loader"]["args"]["train_image_size_"],
        batch_size=512,
        mean= config["data_loader"]["args"]["mean"],
        std= config["data_loader"]["args"]["std"],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
        is_test= True
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    try:
        with open('./class_ids.json') as f:
            class_ids = json.load(f)
    except:
        raise FileExistsError

    with torch.no_grad():
        total = 0
        for i, (data, target, path) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            outputs = F.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)
            for j in range(len(preds)):
                shutil.copy(path[j], f"./self_label/{preds[j]}")

            total += len(target)
    log = {
        "total(pics)": total
    }
    logger.info(log)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="./config-train.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)
    main(config)
