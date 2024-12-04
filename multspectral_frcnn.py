import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform




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



def get_multispectral_frcnn():

    # Use resnet_fpn_backbone directly to handle FPN compatibility
    backbone_with_fpn = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V1,
                                            trainable_layers=5)

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
        image_std=[1.0] * 7  # Std deviation for 7 channels
    )

    return model



if __name__ == "__main__":


    ## Forward Pass (sanity check)
    # Example input: Batch of 7-channel images (batch_size=2, height=256, width=256)
    example_input = torch.randn(2, 7, 256, 256)

    # Example dummy targets
    targets = [
        {"boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32), "labels": torch.tensor([1], dtype=torch.int64)},
        {"boxes": torch.tensor([[30, 30, 120, 120]], dtype=torch.float32), "labels": torch.tensor([1], dtype=torch.int64)},
    ]

    # Forward pass
    model = get_multispectral_frcnn()
    outputs = model(example_input, targets)
    print(outputs)