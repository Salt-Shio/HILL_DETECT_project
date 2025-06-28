from utils.dataset_fix_tool import yaml_fixer, class_fixer
from os import listdir, getcwd

def main():
    feature_string = "BlackSoldierFly_Lableling"
    dataset_dirs = [dataset 
                    for dataset in listdir("./data") 
                    if feature_string in dataset]

    for dataset in dataset_dirs:
        # pwd 
        full_path = f"{getcwd()}/data/{dataset}"
        yaml_fixer(full_path, dataset)
        class_fixer(dataset)
        print()

if __name__ == "__main__":
    main()
