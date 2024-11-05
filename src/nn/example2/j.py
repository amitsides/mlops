# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, _ = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, seq_length, self.heads, self.head_dim)
        keys = keys.view(N, seq_length, self.heads, self.head_dim)
        queries = queries.view(N, seq_length, self.heads, self.head_dim)

        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        energy = torch.einsum("nqhd,nkhd->nqkh", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nqkh,nvhd->nqhd", [attention, values])
        out = out.view(N, seq_length, self.heads * self.head_dim)

        return self.fc_out(out)

# Define the Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden)
        self.fc2 = nn.Linear(ff_hidden, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# Define the Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden)

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)  # Residual connection
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)  # Residual connection
        return x

# Define the Transformer Model
class Transformer(nn.Module):
    def __init__(self, embed_size, heads, num_layers, ff_hidden):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, ff_hidden) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Hyperparameters
embed_size = 1024  # Embedding dimension
num_heads = 16     # Number of attention heads
num_layers = 16    # Number of transformer layers
ff_hidden = 2048   # Feed forward dimension

# Initialize the Transformer model
model = Transformer(embed_size, num_heads, num_layers, ff_hidden)

# Print the model architecture
print(model)
