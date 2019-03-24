import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
import math
from tqdm import trange 

import midi
from midi2audio import FluidSynth
import utils
import numpy as np
import random

seed = int(input('Enter random seed: '))

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = 512
num_epochs = 4000
learning_rate = 0.001
hidden_size = 120
step_size = 800
gamma = 0.5
thresh = float(input('Enter density rate(float number from 0 to 1, the lower - the denser, 0.25 is optimal for most cases): '))
output_dir = input('Enter output directory (with /): ')
model_path = input('Enter path to the pretrained model: ')

fs = FluidSynth('sf2/8bitsf.SF2')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

#torch.backends.cudnn.benchmark = True

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(96 * 96, 2000)
        self.fc2 = nn.Linear(2000, 200)
        self.fc3 = nn.Linear(16 * 200, 1600)
        self.fc4 = nn.Linear(1600, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc3(x))
        x = self.bn1(self.fc4(x))
        
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(hidden_size, 1600)
        self.fc2 = nn.Linear(1600, 16 * 200)
        self.fc3 = nn.Linear(200, 2000)
        self.fc4 = nn.Linear(2000, 96 * 96)
        self.bn1 = nn.BatchNorm1d(1600)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(16)
        

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        x = x.view(-1, 16, 200)
        x = F.relu(self.bn2(x))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.sigmoid(self.fc4(x))
        
        return x.view(-1, 16, 96, 96)
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.enc = Encoder()
        self.dec = Decoder()
        
    def forward(self, x): 
        x = self.enc(x)
        x = self.dec(x)
        
        return x
    
    def encoder_forward(self, x):
        x = self.enc(x)
        
        return x
    
    def decoder_forward(self, x):
        x = self.dec(x)
        
        return x
    
model = Autoencoder().to(device)
model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
#summary(model, (16, 96, 96))

model.eval()

y_orig = np.load('samples.npy')
y_orig_tensor = torch.from_numpy(y_orig).to(device)

with torch.no_grad():        
    x_enc = np.squeeze(model.encoder_forward(y_orig_tensor.float()).cpu().numpy())

x_mean = np.mean(x_enc, axis=0)
x_stds = np.std(x_enc, axis=0)
x_cov = np.cov((x_enc - x_mean).T)
u, s, v = np.linalg.svd(x_cov)
e = np.sqrt(s)

del x_enc
del y_orig_tensor 

def make_rand_songs(write_dir, rand_vecs, thresh):
    for i in range(rand_vecs.shape[0]):
        x_rand = torch.from_numpy(rand_vecs[i : i + 1]).to(device).float()
        with torch.no_grad(): 
            y_song = model.decoder_forward(x_rand).cpu().numpy()
            midi.samples_to_midi(y_song[0], write_dir + 'rand' + str(i) + '.mid', 16, thresh)
            fs.midi_to_audio(write_dir + 'rand' + str(i) + '.mid', output_dir + 'sample' + str(i) + '.wav')

def evaluate(thresh):
    rand_vecs = np.random.normal(0.0, 1.0, (10, hidden_size))
    x_vecs = x_mean + np.dot(rand_vecs * e, v)
    
    make_rand_songs(output_dir, x_vecs, thresh)

evaluate(thresh)
