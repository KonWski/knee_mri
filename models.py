from os import listdir
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Process paramaters for model learning')
    parser.add_argument('--view_type', type=str, help='axial/coronal/sagittal')
    parser.add_argument('--abnormality_type', type=str, help='abnormal/acl/meniscus')
    parser.add_argument('--dataset_path', type=str, help='')
    parser.add_argument('--pretrained_model_type', type=str, help='Type of model used for feature extraction AlexNet/Resnet/Inception')
    parser.add_argument('--batchsize', type=int, help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
    dataset_path = args["dataset_path"]
    files = listdir(dataset_path)
    print(files)