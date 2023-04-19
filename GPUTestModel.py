import torch
import torch.nn as nn

# define the dimensions of the input and output layers
input_size = 10
output_size = 1

# define the dimensions of the hidden layer
hidden_size = 5

# define the model architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# create an instance of the model
model = SimpleNet()

# define the loss function
criterion = nn.MSELoss()

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# create a sample input
x = torch.randn(1, input_size)

# pass the input through the model and print the output
output = model(x)
print(output)