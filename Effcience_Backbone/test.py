import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import model.loss as module_loss
import model.model as module_arch
import model.metric as module_metric
from parse_config import ConfigParser
import data_loader.data_loaders as module_data


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        config["data_loader"]["args"]["train_image_size_"],
        batch_size=512,
        mean=config["data_loader"]["args"]["mean"],
        std=config["data_loader"]["args"]["std"],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
        is_test=True,
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    # checkpoint = torch.load(config.resume)
    # state_dict = checkpoint["state_dict"]
    # model.load_state_dict(state_dict)

    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    
    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # convert to ONNX

    # convert to TensorRT
    # from torch2trt import torch2trt
    # x = torch.ones((1, 3, 64, 64)).cuda()
    # model_trt = torch2trt(model, [x])
    # torch.save(model_trt.state_dict(), 'chars_clasifier_trt.pth')

    total_loss = 0.0
    total = 0
    total_pass = 0
    total_fa = 0
    underkill = 0
    overkill = 0
    missing = 0
    total_metrics = torch.zeros(len(metric_fns))

    dst_dir = "./EVALUATE"

    try:
        with open(
            "./class_ids.json"
        ) as f:
            class_ids = json.load(f)
    except:
        raise FileExistsError

    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)

    with torch.no_grad():
        for i, (data, target, path) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.softmax(output, dim=1)
            #
            # Save sample images, or do something with output here
            #
            _, preds = torch.max(output, 1)
            total += len(target)

            # Config Data
            fail_class = [0]
            pass_class = [1]
            pass_threshold = 0.4
            fail_threshold = 0.0

            # F82 cross straight
            # fail_class = [0]
            # pass_class = [1]

            for i, (x, y, z) in enumerate(zip(target, preds, path)):
                x = x.item()
                y = y.item()
                score = np.round(output[i][y].cpu(), 3)
                # pass
                if y == 1 and score <= pass_threshold:
                    y = 0
                elif y == 2 and score <= pass_threshold:
                    y = 0
                elif y == 0 and score <= fail_threshold:
                    y = 1

                # ignore dasdsad
                if y == 2 and score <= 0.86:
                    y = 0

                predicted_classname = f"{class_ids[f'{x}']}_{class_ids[f'{y}']}_{score}"
                src_image_dir = path[i]
                src_basename = os.path.basename(src_image_dir)

                if x in pass_class:
                    total_pass += 1

                if x in fail_class:
                    total_fa += 1

                if x in fail_class and y not in fail_class:
                    underkill += 1
                    dst_image_dir = os.path.join(dst_dir, "UNDERKILL")
                    os.makedirs(dst_image_dir, exist_ok=True)
                    dst_image_dir = os.path.join(
                        dst_image_dir, f"{predicted_classname}_{src_basename}"
                    )
                    shutil.copy(src_image_dir, dst_image_dir)

                elif x not in fail_class and y in fail_class:
                    overkill += 1
                    dst_image_dir = os.path.join(dst_dir, "OVERKILL")
                    os.makedirs(dst_image_dir, exist_ok=True)
                    dst_image_dir = os.path.join(
                        dst_image_dir, f"{predicted_classname}_{src_basename}"
                    )
                    shutil.copy(src_image_dir, dst_image_dir)

                elif x not in fail_class and y not in fail_class and x != y:
                    overkill += 1
                    missing += 1
                    dst_image_dir = os.path.join(dst_dir, "OVERKILL")
                    os.makedirs(dst_image_dir, exist_ok=True)
                    dst_image_dir = os.path.join(
                        dst_image_dir, f"{predicted_classname}_{src_basename}"
                    )
                    shutil.copy(src_image_dir, dst_image_dir)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

        overkill = overkill
        underkill = underkill
        missing = missing

    n_samples = len(data_loader.sampler)
    log = {
        "datadir": config["data_loader"]["args"]["data_dir"].split("/")[-1],
        "loss": total_loss / n_samples,
        "overkill(pics)": overkill,
        "overkill/Pass(%)": overkill / total_pass * 100,
        "underkill(pics)": underkill,
        "underkill/Fail(%)": underkill / total_fa * 100,
        "total_pass": total_pass,
        "total_fail": total_fa,
        "missing(pics)": missing,
        "total(pics)": total,
    }
    log.update(
        {
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_fns)
        }
    )
    logger.info(json.dumps(log, indent=4))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="./config-test.json",
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
