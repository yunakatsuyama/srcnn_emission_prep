
# srcc network for initial emission downscaling 

import time 
start_time = time.time()
import torch
from torch import nn
from torch import optim
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np

cams_data = np.load("cams_conserv_xe.npy")
eagrid_data = np.load("eagrid_dataset.npy")
bilinear_data = np.load("cams_interp_xe.npy") # omly for comparizon 
# Augu,mentation, ,,, I will do it later.
#cams_aug_file = 

class SRDataset(Dataset):
    def __init__(self, cams, eagrid, indices=None):
        if len(cams) != len(eagrid):
            raise ValueError("Length mismatch")
        self.cams = cams
        self.eagrid = eagrid
        self.indices = np.arange(len(cams)) if indices is None else indices

    def __len__(self):
        return len(self.cams)

    def __getitem__(self, idx):
        # indices is the original array index
        return self.cams[idx], self.eagrid[idx], self.indices[idx]
    


# === train/val/test divide (80/10/10） ===
def split_indices(n_samples, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    n_test = n_samples - n_train - n_val
    return indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]

train_idx, val_idx, test_idx = split_indices(len(cams_data)) # retrun index
print(f'num of data {len(cams_data)}')

print(eagrid_data)
# === Dataset Make ===
train_dataset = SRDataset(cams_data[train_idx], eagrid_data[train_idx], train_idx)
val_dataset = SRDataset(cams_data[val_idx], eagrid_data[val_idx], val_idx)
test_dataset = SRDataset(cams_data[test_idx], eagrid_data[test_idx], test_idx)

# === DataLoader ===
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # the order does not change from returned index of split_indices
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 



if torch.cuda.is_available():
    device = "cuda"
    print("GPU is used")
else:
    device = "cpu"
    print("CPU is used")




learn_rate = 1e-4
epoch = 2

class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,  
                      out_channels=64,
                      kernel_size= (9, 9),
                      stride= (1, 1),
                      padding= (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=32, 
                      kernel_size=(5, 5), 
                      stride=(1, 1), 
                      padding=(2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(in_channels = 32, 
                                        out_channels=1 , 
                                        kernel_size=(5, 5), 
                                        stride=(1, 1), 
                                        padding=(2, 2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.map(x)
        x = self.reconstruction(x)
        return x    

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self._forward_impl(x)
    
    # # Support torch.script function.
    # def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    #     out = self.features(x)
    #     out = self.map(out)
    #     out = self.reconstruction(out)

    #     return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)
    
net = SRCNN().to(device)
optimizer = optim.Adam(net.parameters(), lr=learn_rate)
criterion = nn.MSELoss()

# %%
L_array=[]
L_val_array = []

# With CPU
epoch_loss_train = 0.0
epoch_loss_val = 0.0
epoch_loss_test = 0.0
for epoch in range(epoch):

# == Train ======================= 
    net.train()
    for batch_low, batch_high , idx_batch in train_loader:   # train_high is a dataloader
        batch_low = batch_low.unsqueeze(1)  # (16, 50 ,50) -> (16, 1, 50, 50) channel = 1
        batch_high = batch_high.unsqueeze(1)
        optimizer.zero_grad()
        # send to GPU
        batch_low = batch_low.float().to(device)
        batch_high = batch_high.float().to(device)
        Z = net(batch_low)

        L = criterion(Z, batch_high)
        L.backward()
        optimizer.step()
        epoch_loss_train += L.item() * batch_low.size(0)  #multiple with batch size
    epoch_loss_train /= len(train_loader.dataset)  # get the average of all batches
    L_array.append(epoch_loss_train)
 
   # print(L.item())

    # == Validation ===================
    net.eval()
    for val_batch_low, val_batch_high , idx_batch in val_loader:
        val_batch_low = val_batch_low.unsqueeze(1)  # (16, 50 ,50) -> (16, 1, 50, 50) channel = 1
        val_batch_high = val_batch_high.unsqueeze(1)
        # send to GPU 
        val_batch_low = val_batch_low.float().to(device)
        val_batch_high = val_batch_high.float().to(device)
        Z_vali = net(val_batch_low)
        L_vali = criterion(Z_vali, val_batch_high)
        epoch_loss_val += L_vali.item() * val_batch_low.size(0)
    epoch_loss_val /= len(val_loader.dataset)
    L_val_array.append(epoch_loss_val)
        
# to compare test and bilinear interp
all_Z_test= []
all_bilinear = []  
# == Test =================
net.eval()
for  test_batch_low, test_batch_high, idx_batch in test_loader:
    test_batch_low = test_batch_low.unsqueeze(1)  # (16, 50 ,50) -> (16, 1, 50, 50) channel = 1
    test_batch_high = test_batch_high.unsqueeze(1)
    test_batch_low = test_batch_low.float().to(device)
    test_batch_high = test_batch_high.float().to(device)
    Z_test = net(test_batch_low)
    L_test = criterion(Z_test, test_batch_high)
    bilinear_batch = bilinear_data[idx_batch.numpy()]   # the same place of data with that is from test_loader
    all_Z_test.append(Z_test)
    all_bilinear.append(bilinear_batch)
    epoch_loss_test += L_test.item() * test_batch_low.size(0)
L_test_last = epoch_loss_test / len(test_dataset)

all_Z_test_np = [z.detach().cpu().numpy() for z in all_Z_test]
all_bilinear_np = [b.detach().cpu().numpy() if hasattr(b, "detach") else b for b in all_bilinear]

# all test data and the same image of biliear interp 
all_Z_test = np.concatenate(all_Z_test_np, axis = 0)
all_bilinear = np.concatenate(all_bilinear_np, axis =0)


mse_model = np.mean((all_Z_test - eagrid_data[test_idx])**2)
mse_bilinear = np.mean((all_bilinear - eagrid_data[test_idx])**2)
print("MSE model:", mse_model)
print("MSE bilinear:", mse_bilinear)
print(f'L_test {L_test_last}')    

L_filename = 'L_data/srcnn.csv'
L_out = np.column_stack((L_array, L_val_array))
np.savetxt(L_filename, L_out, delimiter=',')
# np.savetxt(Z_train_filename, Z.detach().cpu().numpy(), delimiter=',')
# np.savetxt(Z_test_filename, Z_vali.detach().cpu().numpy(), delimiter=',') 
# %%

all_Z_test = all_Z_test.cpu().numpy() if hasattr(all_Z_test, "cpu") else all_Z_test
all_bilinear = all_bilinear.cpu().numpy() if hasattr(all_bilinear, "cpu") else all_bilinear

# test index is the original 
sorted_idx = np.argsort(test_idx) 
all_Z_test_sorted = all_Z_test[sorted_idx]
all_bilinear_sorted = all_bilinear[sorted_idx]
eagrid_sorted = eagrid_data[test_idx][sorted_idx]
cams_sorted = cams_data[test_idx][sorted_idx]


np.save("result/SRCNN_output_test.npy", all_Z_test_sorted)
np.save("result/bilinear_test.npy", all_bilinear_sorted)
np.save("result/ground_truth_test.npy", eagrid_sorted)
np.save("result/input_cams_test.npy", cams_sorted)