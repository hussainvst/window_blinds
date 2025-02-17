import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import pix2pix

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load dataset
dataset = torchvision.datasets.ImageFolder(root="dataset/", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load the Pix2Pix model
model = pix2pix.Generator()
model.to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = torch.nn.MSELoss()

for epoch in range(50):  # Train for 50 epochs
    for real_A, real_B in dataloader:
        real_A, real_B = real_A.to(device), real_B.to(device)
        
        optimizer.zero_grad()
        
        fake_B = model(real_A)
        loss = criterion(fake_B, real_B)
        
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/50], Loss: {loss.item()}")
    
torch.save(model.state_dict(), "pix2pix_model.pth")
