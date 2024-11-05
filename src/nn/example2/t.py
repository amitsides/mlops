# Create model and trainer
vocab_size = 30000  # Size of your vocabulary
d_model = 512      # Embedding dimension
trainer = create_transformer_embeddings(vocab_size, d_model)

# Training loop (assuming you have a dataloader)
for epoch in range(num_epochs):
    for batch, labels in dataloader:
        loss = trainer.train_step(batch, labels)

# Generate embeddings
model.eval()
with torch.no_grad():
    embeddings = model(input_sequence)