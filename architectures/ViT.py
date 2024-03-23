import torch
import numpy as np
import torch.nn as nn

def patch_reshape(tensor, patch_size=4):
    # tensor shape: (batch, channels, h, w)
    batch, channels, h, w = tensor.shape

    # Create patches using unfold
    patches = tensor.unfold(2, patch_size, patch_size)
    
    patches = patches.unfold(3, patch_size, patch_size)
    # patches shape: (batch, channels, h/patch_size, w/patch_size, patch_size, patch_size)

    # Reshape patches - permute is to ensure last dimension is channels, so all channel values per patch stick together
    patches = patches.permute(0,2,3,4,5,1).contiguous().view(batch, -1, channels * patch_size * patch_size)
    # patches shape: (batch, num_tokens, model_dim)

    return patches

def zero_weights_fn(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)

##have to make custom encoder layer - to be able to not include the residual connection
class encoder_layer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, include_residual=True, zero_weights=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        if zero_weights:
            self.ff.apply(zero_weights_fn)
        self.include_residual = include_residual
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        if self.include_residual:
            x1 = self.norm1(self.mha(x,x,x)[0] + x)
            return self.norm2(self.ff(x1) + x1)
        else:
            return self.norm2(self.ff(self.norm1(self.mha(x,x,x)[0])))
            

class ViT(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, num_layers=7, include_residual=True, patch_size=4, img_shape=(32,32), num_classes=100, zero_weights=False):
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = 1+((img_shape[0]*img_shape[1])/(patch_size**2))
        self.class_embedding = nn.Embedding(1,d_model)
        self.patch_projector = nn.Linear(3*(patch_size**2), d_model)
        self.learned_pe = nn.Embedding(self.num_tokens, d_model)
        self.network = nn.TransformerEncoder(encoder_layer(d_model, nhead, dim_feedforward, include_residual=include_residual, zero_weights=zero_weights), num_layers)
        self.linear_classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        ##view in patches
        x = patch_reshape(x) #shape of (batch, num_tokens, 3*(patch_size**2))

        ##project the patches into d_model
        x = self.patch_projector(x)

        ##concat the class embedding
        x = torch.cat((self.class_embedding(torch.LongTensor([0]).to(x.device)).view(1,1,-1).repeat(x.shape[0],1,1), x))

        ##add positional encodings
        x+=self.learned_pe(torch.arange(self.num_tokens).long().to(x.device))

        ##feed through transformer encoder
        x = self.network(x)

        return self.linear_classifier(x[:,0,:])
