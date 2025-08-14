# src/convlstm.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# ConvLSTM3DCell
# -------------------------
class ConvLSTM3DCell(nn.Module):
    """
    Cellule ConvLSTM 3D (gates classiques i,f,o,g).
    Input shape: (B, C_in, D, H, W)
    Hidden shape: (B, hidden_dim, D, H, W)
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=(3,3,3), bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = tuple(k // 2 for k in kernel_size)
        self.bias = bias

        self.conv = nn.Conv3d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device=None):
        D, H, W = image_size
        if device is None:
            device = self.conv.weight.device
        h = torch.zeros(batch_size, self.hidden_dim, D, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, D, H, W, device=device)
        return (h, c)


# -------------------------
# ConvLSTM3D multi-couches
# -------------------------
class ConvLSTM3D(nn.Module):
    """
    multi-layer ConvLSTM3D
    Input tensor shape (B, T, C, D, H, W) if batch_first=True else (T, B, C, D, H, W)
    """
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers=None,
                 batch_first=True, bias=True, return_all_layers=False):
        super().__init__()
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * len(hidden_dims)
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]
        num_layers = len(hidden_dims) if num_layers is None else num_layers
        assert len(hidden_dims) == num_layers and len(kernel_sizes) == num_layers

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_in = self.input_dim if i == 0 else self.hidden_dims[i-1]
            cell = ConvLSTM3DCell(input_dim=cur_in, hidden_dim=self.hidden_dims[i],
                                  kernel_size=self.kernel_sizes[i], bias=self.bias)
            cell_list.append(cell)
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        input_tensor: (B, T, C, D, H, W)
        returns: layer_output_list, last_state_list
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)

        B, T, C, D, H, W = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=B, image_size=(D, H, W))

        layer_output_list = []
        last_state_list = []

        cur_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(T):
                h, c = self.cell_list[layer_idx](cur_input[:, t, :, :, :, :], (h, c))
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)  # (B, T, hidden_dim, D, H, W)
            cur_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


# -------------------------
# Predictor: AutoEncoder.encode() -> ConvLSTM -> upsample -> predict mask
# -------------------------
class TumorEvolutionPredictor(nn.Module):
    """
    autoencoder: instance ayant .encode(x) -> latent (B, latent_dim)
    use_mask: si True, on concatène le masque (downsampled) aux features latents
    """
    def __init__(self, autoencoder, latent_dim=128, use_mask=True, mask_channels=1,
                 hidden_dims=[64, 32], kernel_sizes=[(3,3,3), (3,3,3)], num_layers=2,
                 grid_size=(4,4,4)):
        super().__init__()
        self.autoencoder = autoencoder
        self.latent_dim = latent_dim
        self.use_mask = use_mask
        self.mask_channels = mask_channels if use_mask else 0
        self.grid_size = grid_size  # ex (4,4,4) — doit matcher l'AdaptiveAvgPool de l'AE

        convlstm_input_dim = self.latent_dim + (self.mask_channels)
        self.convlstm = ConvLSTM3D(input_dim=convlstm_input_dim,
                                   hidden_dims=hidden_dims,
                                   kernel_sizes=kernel_sizes,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   return_all_layers=False)

        # Predictor sur la sortie du dernier LSTM -> 1 canal (probabilité)
        final_hidden = hidden_dims[-1]
        self.predictor = nn.Sequential(
            nn.Conv3d(final_hidden, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input_images, input_masks=None):
        """
        input_images: (B, T, C, D, H, W)
        input_masks:  (B, T, 1, D, H, W) or None
        returns: prediction (B,1,D,H,W)
        """
        B, T, C, D, H, W = input_images.shape
        device = input_images.device

        encoded_seq = []
        gs_d, gs_h, gs_w = self.grid_size

        for t in range(T):
            x_t = input_images[:, t].to(device)    # (B, C, D, H, W)
            # utiliser la méthode .encode() pour obtenir le vecteur latent (B, latent_dim)
            if hasattr(self.autoencoder, "encode"):
                latent = self.autoencoder.encode(x_t)    # (B, latent_dim)
            else:
                # fallback: forward() et prendre latent (si AE n'a pas encode séparé)
                _, latent = self.autoencoder(x_t)

            # reshape latent -> (B, latent_dim, gs_d, gs_h, gs_w)
            feat = latent.view(B, self.latent_dim, 1, 1, 1)
            feat = feat.expand(-1, -1, gs_d, gs_h, gs_w)  # spatialisation du vecteur latent

            if self.use_mask and input_masks is not None:
                # downsample mask to grid size (average pooling)
                mask_t = input_masks[:, t]  # (B, 1, D, H, W)
                mask_ds = F.adaptive_avg_pool3d(mask_t, output_size=self.grid_size)
                # concat features + mask
                feat = torch.cat([feat, mask_ds], dim=1)  # (B, latent_dim + mask_ch, gs_d, gs_h, gs_w)

            encoded_seq.append(feat)

        # stack temporel -> (B, T, C_feat, gs_d, gs_h, gs_w)
        temporal_features = torch.stack(encoded_seq, dim=1)

        # ConvLSTM: get last layer outputs
        lstm_out_list, last_states = self.convlstm(temporal_features)  # lstm_out_list is list (last layer only)
        lstm_out = lstm_out_list[0]  # (B, T, hidden_dim, gs_d, gs_h, gs_w)
        last_output = lstm_out[:, -1]  # (B, hidden_dim, gs_d, gs_h, gs_w)

        # Upsample vers la taille originale et prédire masque
        upsampled = F.interpolate(last_output, size=(D, H, W), mode='trilinear', align_corners=False)
        prediction = self.predictor(upsampled)  # (B,1,D,H,W)
        return prediction


# helper factory
def create_tumor_predictor(autoencoder, latent_dim=128, use_mask=True, **kwargs):
    """Crée et renvoie un TumorEvolutionPredictor configuré."""
    model = TumorEvolutionPredictor(autoencoder=autoencoder, latent_dim=latent_dim,
                                    use_mask=use_mask, **kwargs)
    return model


# Test rapide si lancé directement
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Import de l'AE local si présent
        from autoencoder import create_autoencoder
        ae = create_autoencoder(latent_dim=128).to(device)
    except Exception as e:
        print("Impossible d'importer create_autoencoder. Crée un AE avant de tester.")
        ae = None

    # Création d'un predictor de test (si ae présent)
    if ae is not None:
        pred = create_tumor_predictor(ae, latent_dim=128, use_mask=True,
                                      hidden_dims=[64,32], kernel_sizes=[(3,3,3),(3,3,3)],
                                      grid_size=(4,4,4)).to(device)

        B = 2
        # Format (B,T,C,D,H,W) -> choisis dimensions compatibles avec ton pipeline
        test_images = torch.randn(B, 2, 1, 64, 64, 64).to(device)
        test_masks = torch.randn(B, 2, 1, 64, 64, 64).to(device)
        out = pred(test_images, test_masks)
        print("Prediction shape:", out.shape)  # (B,1,D,H,W)
