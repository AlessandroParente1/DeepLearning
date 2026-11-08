import torch
from torch import nn

# Mostly grabbed from https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

print (torch.cuda.is_available())
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, input_tensor):
        input_tensor = self.flatten(input_tensor)
        logits = self.linear_relu_stack(input_tensor)
        return logits


model = SimpleNeuralNetwork().to(device)
print(model)

for name, p in model.named_parameters():
    print(name, p.shape)

input = torch.rand(1, 28, 28, device=device)
logits = model(input)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}") # no training, just checking what randomly initialised weights output

