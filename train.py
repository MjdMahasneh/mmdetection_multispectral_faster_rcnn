import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from multspectral_frcnn import get_multispectral_frcnn
from custom_dataset import CustomDataset




# Example data (replace with real data)
images = [torch.randn(7, 256, 256) for _ in range(10)]  # 7-channel random images
annotations = [
    {"boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
     "labels": torch.tensor([1], dtype=torch.int64)}
    for _ in range(10)
]

dataset = CustomDataset(images, annotations)

# DataLoader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Build the model
model = get_multispectral_frcnn()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model to training mode
model.train()

# Training loop
num_epochs = 5




if __name__ == "__main__":

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            # Move data to the same device as the model
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization step
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")




