# When this file runs, it should take the raw data files in data/raw 
# process them into a single tensor, normalize the tensor 
# and save this intermediate representation to the data/processed folder. 
# By normalization here we refer to making sure the images have mean 0 and standard deviation 1.

# import torch
# from torch.utils.data import TensorDataset
# from torchvision import transforms

# def standardize_data():
#     """Load data, process into a single tensor, normalize and save to processed folder"""
#     # Load:
#     num_parts = 5
#     trainsets, traintargets = [], []

#     for i in range(num_parts):
#         trainset_part = torch.load(f"../../data/corruptmnist/train_images_{i}.pt")
#         traintarget_part = torch.load(f"../../data/corruptmnist/train_target_{i}.pt")
#         trainsets.append(trainset_part)
#         traintargets.append(traintarget_part)
#     testset = torch.load("../../data/corruptmnist/test_images.pt")
#     testtarget = torch.load("../../data/corruptmnist/test_target.pt")

#     trainset = TensorDataset(torch.cat(trainsets), torch.cat(traintargets))
#     testset = TensorDataset(testset, testtarget)

#     # Normalize:
#     transform = transforms.Normalize(mean=(0.5,), std=(0.5,))

#     # Apply normalization to the images within the datasets
#     trainset = TensorDataset(
#         transform(trainset.tensors[0]),  # Apply normalization to the whole tensor
#         trainset.tensors[1]
#     )
#     testset = TensorDataset(
#         transform(testset.tensors[0]),  # Apply normalization to the whole tensor
#         testset.tensors[1]
#     )
#     # Save:
#     torch.save(trainset, "../../data/processed/trainset.pt")
#     torch.save(testset, "../../data/processed/testset.pt")
    
#     return trainset, testset


import torch
from torch.utils.data import TensorDataset
from torchvision import transforms



def load_data(num_parts=5):
    """Load data and put into a single dataset"""
    trainsets, traintargets = [], []

    for i in range(num_parts):
        trainset_part = torch.load(f"../../data/corruptmnist/train_images_{i}.pt")
        traintarget_part = torch.load(f"../../data/corruptmnist/train_target_{i}.pt")
        trainsets.append(trainset_part)
        traintargets.append(traintarget_part)
    testset = torch.load("../../data/corruptmnist/test_images.pt")
    testtarget = torch.load("../../data/corruptmnist/test_target.pt")

    trainset = TensorDataset(torch.cat(trainsets), torch.cat(traintargets))
    testset = TensorDataset(testset, testtarget)

    return trainset, testset

def transform_data(dataset):
    """Normalize the dataset"""
    transform = transforms.Normalize(mean=(0.5,), std=(0.5,))
    
    transformed_trainset = TensorDataset(
        transform(dataset.tensors[0]),  # Apply normalization to the whole tensor
        dataset.tensors[1]
    )

    return transformed_trainset

def save_data(dataset, save_path):
    """Save the dataset to the specified path"""
    torch.save(dataset, save_path)

def main():
    """Main function to standardize and save the data"""
    # Load data:
    trainset, testset = load_data()

    # Transform data:
    transformed_trainset = transform_data(trainset)
    transformed_testset = transform_data(testset)

    # Save data:
    save_data(transformed_trainset, "../../data/processed/trainset.pt")
    save_data(transformed_testset, "../../data/processed/testset.pt")

if __name__ == "__main__":
    main()



