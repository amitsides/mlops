# For simple word embeddings
vocab_size = 1000
embedding_dim = 128
model = SimpleEmbedding(vocab_size, embedding_dim)

# For Siamese network
input_dim = 256
hidden_dim = 512
embedding_dim = 128
model = SiameseNetwork(input_dim, hidden_dim, embedding_dim)

# For contrastive learning
model = ContrastiveEmbedding(input_dim, hidden_dim, embedding_dim)

# Prepare data and train
dataset = EmbeddingDataset(your_data, your_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
train_embeddings(model, dataloader)