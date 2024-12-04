import torch
from multspectral_frcnn import get_multispectral_frcnn



# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the model
model = get_multispectral_frcnn()

# Move the model to the device
model.to(device)

# Switch to evaluation mode
model.eval()


if __name__ == "__main__":

    # Example input: Batch of 7-channel images (batch_size=2, height=256, width=256)
    example_input = torch.randn(2, 7, 256, 256)

    # Move example input to the same device as the model
    example_input = example_input.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(example_input)
    print(outputs)
