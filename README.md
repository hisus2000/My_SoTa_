
# Introduction

This Repository combines a lot model is called SOTA of 2022. You can read more information in [here](http://github.com/sithu31296/sota-backbones).

## How to run

#### Check All Model: 

`python list_models.py`

#### Train:

`python finetune.py --cfg ./configs/finetune.yaml`

#### Evaluate

`python infer.py --source {source} --model {model} --variant {size_of_model} --checkpoint {link checkpoint} --size {size of image}`

#### Split Dataset

`splitfolders --output data_set --ratio .8 .1 .1 -- {folder_with_images}`

All Class Folder is in **Images Folder**.

##### Note:

The current model only works well on certain network architectures.