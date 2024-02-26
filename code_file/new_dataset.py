import os
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import torchvision
from dataclasses import dataclass
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
import torchvision.transforms as transforms
from diffusers import AutoencoderKL


class DatasetSVD:
    def __init__(self, data):
        if data == 'Cifar-10':
            self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        else:
            self.trainset = None

    def svd(self, image): 
        U = []
        S = []
        Vt = []
        # Iterate over each channel
        for i in range(image.shape[0]):
            u, s, vt = torch.linalg.svd(image[i], full_matrices=False)
            U.append(u)
            S.append(s)
            Vt.append(vt)
        U = torch.concat(U).reshape(3,-1,image.shape[1])  
        S = torch.concat(S).reshape(3, image.shape[1])
        Vt = torch.concat(Vt).reshape(3,-1,image.shape[2]) 
        return U, S, Vt
    
    def compress_data(self, U, S, Vt, comp_rate):
        S_k = torch.zeros_like(S)
        n = U.shape[1]
        m = Vt.shape[2]
        for i in range(3):  # Iterate over channels
            k = int(comp_rate * (n * m) / (n + 1 + m))  # Number of singular values to keep
            S_k[i, :k] = S[i, :k].clone()
        # Create Sigma_k by putting top k singular values on the diagonal
        Sigma_k = torch.zeros(3, n, m)
        for i in range(3):
            Sigma_k[i] = torch.diag(S_k[i])
        # Compressed approximation of the image
        image_channel_approx = U @ Sigma_k @ Vt
        return image_channel_approx

    def get_svd(self, comp_rate=0.1):
        compressed_images = []
        for i, (image, _) in tqdm(enumerate(self.trainset),total =int(len(self.trainset))):
            image_tensor = torch.tensor(np.array(image).transpose((2, 0, 1)), dtype=torch.float32) / 255.0
            # Apply SVD to the image
            U, S, Vt = self.svd(image_tensor)
            # Compress image data
            compressed_image = self.compress_data(U, S, Vt, comp_rate)  
            compressed_images.append(compressed_image)

        return compressed_images, self.trainset.targets


class CustomDataset(Dataset): 
  def __init__(self, images, labels, encoder=False):
    self.images = images
    self.labels = labels
    self.encoder = encoder
    self.transform  = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  def __len__(self): 
    return len(self.images)

  def __getitem__(self, idx): 
    image = self.images[idx]
    
    if self.encoder:
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
        model = AutoencoderKL.from_single_file(url)
        latent = model.encode(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long).unsqueeze(0)
        return latent, label
    
    image = self.transform(image)
    label = torch.tensor(self.labels[idx], dtype=torch.long).unsqueeze(0)
    return image, label


@dataclass # init, eq, ne등 메소드가 자동으로 생성됨. 
class BaseConfig:
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    DATASET = "Cifar-10" #  "MNIST", "Cifar-10", "Cifar-100", "Flowers"
    
    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join("Logs_Checkpoints", "Inference")
    root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")

    # Current log and checkpoint directory.
    log_dir = "version_0"
    checkpoint_dir = "version_0"


@dataclass
class TrainingConfig:
    TIMESTEPS = 1000 # Define number of diffusion timesteps
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32)  # Mnist만 gray scale
    NUM_EPOCHS = 30
    BATCH_SIZE = 128
    LR = 2e-4
    NUM_WORKERS = 0



