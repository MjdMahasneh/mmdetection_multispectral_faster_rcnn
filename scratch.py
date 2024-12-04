import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim






#####################
## Custom Model
#####################

# Define a custom transform class that inherits from GeneralizedRCNNTransform and overrides the mean and std to handle 7 channels
class CustomTransform(GeneralizedRCNNTransform):

    """
    The CustomTransform class extends PyTorch's GeneralizedRCNNTransform and is responsible
    for preprocessing images before feeding them into the Faster R-CNN model.

    1. Purpose of GeneralizedRCNNTransform:
       - Prepares input images for the model by:
           * Resizing the images to a target size.
           * Normalizing pixel values using mean and standard deviation (per channel).
           * Collating a batch of images into a tensor.
       - Also applies inverse transformations to decode predictions back into the original image space.

    2. Customizing for 7 Channels:
       - The original GeneralizedRCNNTransform assumes images with 3 channels (RGB).
       - For 7-channel images, the image_mean and image_std must be overridden to match the input channels.

    3. How the Code Works:
       - Inherits behavior from GeneralizedRCNNTransform via super().__init__(*args, **kwargs).
         This retains default functionalities like resizing and collating.
       - Overrides the image_mean and image_std:
           * Default values for 3-channel images:
               self.image_mean = [0.485, 0.456, 0.406]
               self.image_std = [0.229, 0.224, 0.225]
           * Custom values for 7-channel images:
               self.image_mean = [0.0] * 7  # Mean for each of the 7 channels.
               self.image_std = [1.0] * 7   # Standard deviation for each of the 7 channels.
       - These values can be adjusted based on the dataset's statistics.

    4. Why Update Mean and Std:
       - Normalization ensures all input features are on a similar scale, stabilizing training.
       - For 7-channel images, these values must match the number of input channels.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Update the mean and std for 7 channels
        self.image_mean = [0.0] * 7  # Mean for each of the 7 channels
        self.image_std = [1.0] * 7   # Std deviation for each of the 7 channels


# Use resnet_fpn_backbone directly to handle FPN compatibility
backbone_with_fpn = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V1, trainable_layers=5)

# Modify the first layer of the backbone for 7-channel input
backbone_with_fpn.body.conv1 = torch.nn.Conv2d(
    in_channels=7,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

# Reinitialize weights for additional channels
torch.nn.init.kaiming_normal_(backbone_with_fpn.body.conv1.weight[:, 3:, :, :])

# Create a Faster R-CNN model using the modified backbone
model = FasterRCNN(backbone_with_fpn, num_classes=2)  # Change `num_classes` as per your dataset

# Replace the transform with the custom one
model.transform = CustomTransform(
    min_size=800,  # Minimum size of the image
    max_size=1333,  # Maximum size of the image
    image_mean=[0.0] * 7,  # Mean for 7 channels
    image_std=[1.0] * 7    # Std deviation for 7 channels
)




#####################
## Forward Pass (sanity check)
#####################
# Example input: Batch of 7-channel images (batch_size=2, height=256, width=256)
example_input = torch.randn(2, 7, 256, 256)

# Example dummy targets
targets = [
    {"boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32), "labels": torch.tensor([1], dtype=torch.int64)},
    {"boxes": torch.tensor([[30, 30, 120, 120]], dtype=torch.float32), "labels": torch.tensor([1], dtype=torch.int64)},
]

# Forward pass
outputs = model(example_input, targets)
print(outputs)





#####################
## Custom Dataset
#####################
class CustomDataset(Dataset):
    def __init__(self, images, annotations):
        self.images = images  # List of 7-channel images (as tensors)
        self.annotations = annotations  # List of dictionaries with 'boxes' and 'labels'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.annotations[idx]

# Example data (replace with real data)
images = [torch.randn(7, 256, 256) for _ in range(10)]  # 7-channel random images
annotations = [
    {"boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
     "labels": torch.tensor([1], dtype=torch.int64)}
    for _ in range(10)
]
dataset = CustomDataset(images, annotations)



#####################
## DataLoader
#####################
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


#####################
## Train the model
#####################

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model to training mode
model.train()

# Training loop
num_epochs = 5
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



#####################
## Evaluate the model
#####################

# Switch to evaluation mode
model.eval()

# Example input: Batch of 7-channel images (batch_size=2, height=256, width=256)
example_input = torch.randn(2, 7, 256, 256)


# Move example input to the same device as the model
example_input = example_input.to(device)

# Perform inference
with torch.no_grad():
    outputs = model(example_input)
print(outputs)
