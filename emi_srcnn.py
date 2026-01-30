
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

cams_file = "cams_corserv_xe.npy"
eagrid_file = "eagrid_dataset.npy"

# あとで...
#cams_aug_file = 

class SRDataset(Dataset):
    def __init__(self, low_file, high_file):
        cams_emission = np.load(low_file)
        eagrid_emission = np.load(high_file)
        
        if len(cams_emission) != len(eagrid_emission):
            raise ValueError(f"Length mismatch: cams_emission has {len(cams_emission)}, "
                             f"eagrid_emission has {len(eagrid_emission)}")

        self.cams_emission = cams_emission
        self.eagrid_emission = eagrid_emission

    def __len__(self):
        return len(self.cams_emission)

    def __getitem__(self, idx):
        return self.cams_emission[idx], self.eagrid_emission[idx]
    
    

train_dataset = SRDataset(img_path_train_low, img_path_train_high)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle = True)

val_dataset = SRDataset(img_path_val_low, img_path_val_high)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)


if torch.cuda.is_available():
    device = "cuda"
    print("GPU is used")
else:
    device = "cpu"
    print("CPU is used")



# %%
learn_rate = 1e-4
epoch = 2



# %%

class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,   # NO pre-processing = channel is 3 (RGB)
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
                                        out_channels=3, 
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
optimizer = optim.SGD(net.parameters(), lr=learn_rate)
criterion = nn.MSELoss()

# %%
L_array=[]
L_test_array = []

# With CPU
for epoch in range(epoch):

# == Train ======================= 
    net.train()
    for batch_low, batch_high in train_dataloader:   # train_high is a dataloader
        optimizer.zero_grad()
        # send to GPU
        batch_low = batch_low.to(device)
        batch_high = batch_high.to(device)
        Z = net(batch_low)

        L = criterion(Z, batch_high)
        L.backward()
        optimizer.step()
    L_array.append(L.item())    
   # print(L.item())

    # == Validation ===================
    net.eval()
    for val_batch_low, val_batch_high in val_dataloader:
        # send to GPU 
        val_batch_low = val_batch_low.to(device)
        val_batch_high = val_batch_high.to(device)
        Z_vali = net(val_batch_low)
        L_test = criterion(Z_vali, val_batch_high)
    L_test_array.append(L_test.item())
        
    
    

# %%
# epochs = np.arange(0, epoch, 1)
# plt.plot(epochs, L_array,label='Train Loss', c='b')
# plt.plot(epochs, L_test_array, label='Validation Loss', c='g', ls='dashed')
# plt.xlabel('epoch')
# plt.ylabel('Cross Entropy Loss')
# plt.legend()
# plt.grid()
# plt.savefig('plot/digit_sigmoid.png')
# plt.show()

#print(f'filal train L ={L_array[-1]}')
#print(f'final L test L = {L_test_array[-1]}')

L_filename = 'L_data/srcnn.csv'


L_out = np.column_stack((L_array, L_test_array))
np.savetxt(L_filename, L_out, delimiter=',')
# np.savetxt(Z_train_filename, Z.detach().cpu().numpy(), delimiter=',')
# np.savetxt(Z_test_filename, Z_vali.detach().cpu().numpy(), delimiter=',') 
# %%

img = Z[0, :,:,:]
# .permute change order of dimention 
# tensor [channel(RGB), H,W]  -> matplot [H, W, channel]
img = img.permute(1, 2, 0).detach().cpu().numpy()  # [256, 256, 3]
plt.imshow(img)
plt.axis("off")
plt.title("1st image after trained")
plt.savefig("plot/train.png")
plt.show()

img = batch_high[0,:,:,:]
img = img.permute(1, 2, 0).detach().cpu().numpy()  # [256, 256, 3]
plt.imshow(img)
plt.axis("off")
plt.title("target image")
plt.savefig("plot/target.png")
plt.show()

img = batch_low[0,:,:,:]
img = img.permute(1, 2, 0).detach().cpu().numpy()  # [256, 256, 3]
plt.imshow(img)
plt.axis("off")
plt.title("input image")
plt.savefig("plot/input.png")

end_time = time.time()
elapsed = (end_time - start_time) / 60
print(f"Total training time�: {elapsed:.2f} min")
