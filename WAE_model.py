
import torch
import torch.nn as nn
import torch.nn.functional as F

class WAE(nn.Module):
    def __init__(self, in_features, latent_size):
        super(WAE, self).__init__()

        self.encoder_fcl_1 = nn.Linear(in_features, 90)
        self.encoder_fcl_1_norm = nn.LayerNorm(90)
        self.encoder_fcl_2 = nn.Linear(90, 48)
        self.encoder_fcl_2_norm = nn.LayerNorm(48)
        self.latent_space = nn.Linear(48, latent_size)

        self.predict_fcl_1 = nn.Linear(latent_size, 60)
        self.predict_fcl_1_norm = nn.LayerNorm(60)
        self.predict_fcl_2 = nn.Linear(60, 30)
        self.predict_fcl_2_norm = nn.LayerNorm(30)
        self.predict_out = nn.Linear(30, 1)

        self.decoder_fcl_1 = nn.Linear(latent_size, 48)
        self.decoder_fcl_1_norm = nn.LayerNorm(48)
        self.decoder_fcl_2 = nn.Linear(48, 90)
        self.decoder_fcl_2_norm = nn.LayerNorm(90)
        self.out_fcl = nn.Linear(90, in_features)

    def encoder(self, X):
        out = self.encoder_fcl_1(X)
        out = F.relu(self.encoder_fcl_1_norm(out))
        out = self.encoder_fcl_2(out)
        out = F.relu(self.encoder_fcl_2_norm(out))
        z = self.latent_space(out)
        return z 

    def decoder(self, z):
        h = self.decoder_fcl_1(z)
        h = F.relu(self.decoder_fcl_1_norm(h))
        h = self.decoder_fcl_2(h)
        h = F.relu(self.decoder_fcl_2_norm(h))
        x_reconst = F.softmax(self.out_fcl(h), dim=1)
        return x_reconst
    
    def predict(self, z):
        p = self.predict_fcl_1(z)
        p = F.relu(self.predict_fcl_1_norm(p))
        p = self.predict_fcl_2(p)
        p = F.relu(self.predict_fcl_2_norm(p))
        pro_out = self.predict_out(p)
        return pro_out

    def forward(self, X):
        z = self.encoder(X)
        x_reconst = self.decoder(z)
        pro_out = self.predict(z)
        return x_reconst, z, pro_out
