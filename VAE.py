import torch
import torch.nn as nn
from torchvision import models

import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, backbone):
        super(VAE, self).__init__()
        if backbone == 'VGG':
            new_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.backbone = nn.Sequential(
            *list(new_model.features.children()),
            new_model.avgpool,
            nn.Flatten(),
            *list(new_model.classifier.children())[:-1],
            nn.ReLU(),
            nn.Linear(4096, 512)
        )   
        elif backbone == 'ResNet18':
            new_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(
            *list(new_model.children())[:-2],  
            new_model.avgpool,
            nn.Linear(512, 512)
        )
        elif backbone == 'ResNet34':
            new_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.backbone = nn.Sequential(
            *list(new_model.children())[:-2],  
            new_model.avgpool,
            nn.Linear(512,512)
        )
        else:
            new_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            self.backbone = nn.Sequential(
            *list(new_model.features.children()),
            new_model.avgpool,
            nn.Flatten(),
            *list(new_model.classifier.children())[:-1],
            nn.ReLU(),
            nn.Linear(4096, 512)
        )

        self.fc1 = nn.Linear(36, 512)
        self.fc2 = nn.Linear(30, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5_1 = nn.Linear(1024, 30)
        self.fc5_2 = nn.Linear(1024, 30)
        self.fc6 = nn.Linear(30, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(1024, 512)
        self.fc10 = nn.Linear(512, 512)
        self.fc11 = nn.Linear(512, 36)
        self.relu = nn.ReLU()
    
    def encode(self, scale_and_deformation, pose_class, image):
        x1 = self.relu(self.fc1(scale_and_deformation))
        x2 = self.relu(self.fc2(pose_class))
        x3 = self.relu(self.backbone(image))

        x1 = self.relu(self.fc3(x1))
        x2 = self.relu(self.fc4(torch.cat([x2,x3], dim=1)))
        
        mu = self.fc5_1(torch.cat([x1,x2], dim=1))
        logvar = self.fc5_2(torch.cat([x1,x2], dim=1))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z, pose_class, image):
        x2 = self.relu(self.fc2(pose_class))
        x3 = self.relu(self.backbone(image))

        z = self.relu(self.fc6(z))
        x = self.relu(self.fc4(torch.cat([x2,x3], dim=1)))
        
        z = self.relu(self.fc7(z))
        x = self.relu(self.fc8(x))
        
        x = self.relu(self.fc9(torch.cat([z,x], dim=1)))
        x = self.relu(self.fc10(x))
        x = self.fc11(x)

        return x
    
    def forward(self, scale_and_deformation, pose_class, image):
        mu, logvar = self.encode(scale_and_deformation, pose_class, image)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, pose_class, image), mu, logvar
    
class vae_loss(nn.Module):
    def __init__(self):
        super(vae_loss, self).__init__()
    def forward(self, target, predict, mu, logvar):
        L2 = torch.sqrt(torch.sum((target - predict) ** 2))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return L2 + KLD

random_image = torch.randn(1,3,224,224)
random_class = torch.rand(1, 30)
random_class = random_class.to(torch.float32)
random_scale = torch.randn(1, 36)
print(random_class)
criterion = vae_loss()
model = VAE('VGG')
optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer.zero_grad()
output, mu, logvar = model(random_scale, random_class, random_image)
loss = criterion(random_scale, output, mu, logvar)
print("loss!!")
print(loss)
loss.backward
optimizer.step()