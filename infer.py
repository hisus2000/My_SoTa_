import torch
import argparse
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from models import *
from datasets import ImageNet
import cv2
import os
import numpy as np
import shutil
from rich.console import Console
from rich.table import Table
console = Console()
class ModelInference:
    def __init__(self, model: str, variant: str, checkpoint: str, size: int) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # dataset class labels (change to trained dataset labels) (can provide a list of labels here)
        self.labels = ImageNet.CLASSES
        # model initialization        
        self.model = eval(model)(variant, checkpoint, len(self.labels), size)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Resize((size, size)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def __call__(self, img_path: str) -> str:
        # read image
        image = io.read_image(img_path)
        
        if(image.shape[0]==1):
            image=torch.cat([image, image, image], dim=0)
        # preprocess
        image = self.preprocess(image).to(self.device)
        # model pass
        with torch.inference_mode():
            pred = self.model(image)
        # postprocess
        cls_name = self.labels[pred.argmax()]
        return cls_name

def Calculate_OK_UK(target_labels, groundtruth_label, predict_label , directory, n_pic_true, n_pic_fall):
    pic_OK=0
    pic_UK=0   
    for i, (y_true, y_pred,dir) in enumerate(zip(groundtruth_label, predict_label,directory)):
        for target_label in target_labels:
            if y_true==target_label:
                pic_OK+=1
                temp_dir=directory[i].split("/")[-1]   
                temp_dir="./result_over_kill/"+temp_dir
                shutil.copy(dir, temp_dir)
            if y_true!=target_label:
                pic_UK+=1  
                temp_dir=directory[i].split("/")[-1]   
                temp_dir="./result_under_kill/"+temp_dir
                shutil.copy(dir, temp_dir) 
    
    OK_rate= pic_OK * 1./n_pic_true
    UK_rate= pic_UK *1./ n_pic_fall

    return OK_rate,UK_rate
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data_set/test/')
    parser.add_argument('--model', type=str, default='VAN')
    parser.add_argument('--variant', type=str, default='L')
    parser.add_argument('--checkpoint', type=str, default="./output/VAN_L_ImageFolder.pt")
    parser.add_argument('--size', type=int, default=224)
    try:
        os.remove("result.txt")
        shutil.rmtree('result_over_kill')
        shutil.rmtree('result_under_kill')
    except:
        pass 

    try:
        os.mkdir("./result_over_kill/")
        os.mkdir("./result_under_kill/")
    except:
        pass  

    args = vars(parser.parse_args())

    source = args.pop('source')
    file_path = Path(source) 
    model = ModelInference(**args)

    total_predict_true=0
    total_pic=0

    target_label=["good"]
    groundtruth_label=[]
    predict_label=[]
    directory=[]
    n_pic_true=0
    n_pic_fall=0

    for filename in os.listdir(source):
        labels=filename
        folder_name = os.path.join(source, filename)
        # checking if it is a file
        if os.path.isdir(folder_name):
            for file in os.listdir(folder_name): 
                file= source+labels+"/"+file
                cls_name = model(str(file))
                # Config Label for Test
                predict=str(cls_name.capitalize()).lower()
                if(labels==predict):
                    n_pic_true+=1                                       
                else:
                    groundtruth_label.append(labels)
                    predict_label.append(predict)
                    n_pic_fall+=1
                    directory.append(str(file))
                total_pic+=1
                # print(f"{file} >>>>> {cls_name.capitalize()}")

                f = open("result.txt", "a")
                f.write(f"{file} >>>>> {cls_name.capitalize()} \n")
                f.close()

    ok_rate,uk_rate=Calculate_OK_UK(target_label, groundtruth_label, predict_label , directory, n_pic_true, n_pic_fall)
    accuracy=n_pic_true/total_pic

    # print("Accuracy is: ",accuracy)
    # print("OverKill Rate: ",ok_rate)
    # print("UnderKill Rate: ",uk_rate)
    # print("Total image for predict is: ",total_pic)
    # print("Total image for predict True: ",n_pic_true)
    # print("Total image for predict False: ",n_pic_fall)


    # print("")
    # for i in directory:
    #     print(i.split("/")[-1])

    table = Table(show_header=True, header_style="magenta")
    table.add_column("Accuracy")
    table.add_column("OverKill Rate")
    table.add_column("UnderKill Rate")
    table.add_column("Total image for predict")    
    table.add_column("Total image for predict True")
    table.add_column("Total image for predict False")
    table.add_row(f"{accuracy*100}%", f"{ok_rate}", f"{uk_rate}", f"{total_pic} pictures", f"{n_pic_true} pictures", f"{n_pic_fall} pictures")
    console.print(table)