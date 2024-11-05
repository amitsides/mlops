import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear transformations and reshape for multi-head attention
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and apply output transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)
        
        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention block
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
                 max_seq_length=512, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Token embeddings + positional encoding
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        
        # Apply transformer encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Return the mean pooled embedding as the sequence representation
        return x.mean(dim=1)

# Training utilities
class TransformerEmbeddingTrainer:
    def __init__(self, model, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
    def train_step(self, batch, labels):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = batch.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        embeddings = self.model(batch)
        
        # Calculate loss (example: using triplet loss)
        anchor, positive, negative = torch.split(embeddings, embeddings.size(0)//3)
        loss = F.triplet_margin_loss(anchor, positive, negative)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Example usage
def create_transformer_embeddings(vocab_size=30000, d_model=512):
    model = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=6,
        d_ff=d_model * 4,
        max_seq_length=512,
        dropout=0.1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98))
    trainer = TransformerEmbeddingTrainer(model, optimizer)
    
    return trainer

# Usage example:
"""
# Create model and trainer
trainer = create_transformer_embeddings()

# Training loop
for epoch in range(num_epochs):
    for batch, labels in dataloader:
        loss = trainer.train_step(batch, labels)
        
    # Generate embeddings for inference
    model.eval()
    with torch.no_grad():
        embeddings = model(input_sequence)
"""