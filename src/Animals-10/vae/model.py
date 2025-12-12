import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(BayesianVAE, self).__init__()

        # [Encoder]
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # [Decoder]
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder_conv(x)

        # [System Key] MC Dropout 강제 활성화 (training=True)
        # eval()을 호출해도 이 라인은 항상 Dropout을 수행합니다.
        x = F.dropout(x, p=0.2, training=True)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # [수정된 부분]
        # z(128) -> decoder_input -> (8192) -> decoder -> 이미지 복원
        x_recon = self.decoder_input(z)
        reconstruction = self.decoder(x_recon)
        return reconstruction, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD