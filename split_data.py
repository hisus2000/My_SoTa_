import shutil
import os
import numpy as np
import argparse

def get_files_from_folder(path):

    files = os.listdir(path)
    return np.asarray(files)

def main(path_to_data, path_of_train, path_of_val, path_of_test, train_ratio):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))

    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)

    train_counter = np.round(data_counter_per_class * (train_ratio))  # 0 - 79 (80)
    val_counter = np.round(data_counter_per_class * ((1 - train_ratio)/2))# 80 - 89 (10)
    test_counter = np.round(data_counter_per_class * ((1 - train_ratio)/2)) # 90 - 99 (10)
  
    #creates dir
    if not os.path.exists(path_of_train):
        os.makedirs(path_of_train)
    if not os.path.exists(path_of_val):
        os.makedirs(path_of_val)
    if not os.path.exists(path_of_test):
        os.makedirs(path_of_test)            

    for i in dirs:
        if not os.path.exists(path_of_train+"/"+i+"/"):
            os.makedirs(path_of_train+"/"+i+"/")
        if not os.path.exists(path_of_val+"/"+i+"/"):
            os.makedirs(path_of_val+"/"+i+"/")
        if not os.path.exists(path_of_test+"/"+i+"/"):
            os.makedirs(path_of_test+"/"+i+"/")

    transfers files
    for i in range(len(dirs)):
        print(i)
        path_to_original = os.path.join(path_to_data, dirs[i])

        # path_to_save = os.path.join(path_to_test_data, dirs[i])
        
        path_to_train = os.path.join(path_of_train, dirs[i])
        path_to_val = os.path.join(path_of_val, dirs[i])
        path_to_test = os.path.join(path_of_test, dirs[i])

        files = get_files_from_folder(path_to_original)

        # copy data for train dir
        for j in range(int(train_counter[i])):
            dst = os.path.join(path_to_train, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)

        # copy data for val dir
        for j in range(int(val_counter[i])):
            dst = os.path.join(path_to_val, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)

        # copy data for test dir
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_test, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", default='./meo/')
  parser.add_argument("--train",  default='./data/train/')
  parser.add_argument("--val",  default='./data/val/')
  parser.add_argument("--test",  default='./data/test/')
  parser.add_argument("--train_ratio", default=0.8)
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  main(args.data_path, args.train, args.val, args.test, float(args.train_ratio))