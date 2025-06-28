from typing import TextIO
import yaml
from os import listdir
from os.path import exists as path_exists

DatasetLabelDir = {"test", "train", "val"}

def yaml_writer(stream: TextIO, yaml_data, style):
    yaml.dump(yaml_data, 
            stream, 
            indent = 2,
            default_flow_style = {"flow": True, "block": False, None: None}[style], 
            sort_keys = False)
    
    stream.write("\n")

def yaml_fixer(full_path, dataset):
    folder = f"./data/{dataset}"

    with open(f"{folder}/data.yaml", 'r') as stream: 
        yaml_data = yaml.load(stream, Loader = yaml.FullLoader)

    split_info = {
        "train": f"{full_path}/train/images",
        "val": f"{full_path}/test/images",
        "test": f"{full_path}/test/images",
    }

    object_info = {
        "kpt_shape": yaml_data["kpt_shape"],
        "flip_idx": yaml_data["flip_idx"],
    }

    class_info = {
        "nc": 1,
        "names": ['BlackSoldierFly'],
    }

    roboflow_info = {
        "roboflow": {
            "workspace": yaml_data["roboflow"]["workspace"],
            "project": yaml_data["roboflow"]["project"],
            "version": yaml_data["roboflow"]["version"],
            "license": yaml_data["roboflow"]["license"],
            "url": yaml_data["roboflow"]["url"],
        }
    
    }

    object_info = {
        "kpt_shape": yaml_data["kpt_shape"],
        "flip_idx": yaml_data["flip_idx"],
    }

    roboflow_info = {
        "roboflow": {
            "workspace": yaml_data["roboflow"]["workspace"],
            "project": yaml_data["roboflow"]["project"],
            "version": yaml_data["roboflow"]["version"],
            "license": yaml_data["roboflow"]["license"],
            "url": yaml_data["roboflow"]["url"],
        }
    }

    with open(f"{folder}/data.yaml", 'w') as stream: 
        yaml_writer(stream, split_info, "block")
        yaml_writer(stream, object_info, None)
        yaml_writer(stream, class_info, None)
        yaml_writer(stream, roboflow_info, "block")

    print(f"已修改 {folder}/data.yaml")


def class_fixer(dataset):
    fix_dirs = []
    for label_dir in DatasetLabelDir:
        label_path = f"./data/{dataset}/{label_dir}/labels"
        
        if not path_exists(label_path): continue
        fix_dirs.append(label_path)

        for file in listdir(label_path):
            with open(f"{label_path}/{file}", 'r') as label_file: 
                labels = label_file.readlines()

            with open(f"{label_path}/{file}", 'w') as label_file: 
                for label in labels:
                    label = '0' + label[1:]
                    label_file.write(label)
    
    for fix_dir in fix_dirs:
        print(f"已修改 {fix_dir}")