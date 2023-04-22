import signal
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# define the model architecture
class SRSRTModel(nn.Module):
    def __init__(self):
        super(SRSRTModel, self).__init__()

        # CUDA
        if not torch.cuda.is_available():
            print("Please use a device which supports CUDA")
            sys.exit(0)
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # initialisations
        self.__stop_training = False
        self.__epoch_iteration = 0
        self.__epochs = 1000

        # define the layers
        self.__input_size = 1000
        self.__hidden_size_1 = 800
        self.__hidden_size_2 = 600
        self.__hidden_size_3 = 400
        self.__output_size = 10
        self.hidden_1 = nn.Linear(self.__input_size, self.__hidden_size_1)
        self.hidden_2 = nn.Linear(self.__hidden_size_1, self.__hidden_size_2)
        self.hidden_3 = nn.Linear(self.__hidden_size_2, self.__hidden_size_3)
        self.output = nn.Linear(self.__hidden_size_3, self.__output_size)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = self.output(x)
        return x
        
    def load_model(self, path):
        if os.path.exists(f"{path}.pth"):
            self.load_state_dict(torch.load(f"{path}.pth"))
        else:
            print(f"{path}.pth doesn't exist.")
        
        # TODO read epoch_iteration from save

    def save_model(self, path):
        if not os.path.exists("models"):
            os.makedirs("models")

        torch.save(self.state_dict(), f"{path}.pth")
        # TODO save epoch_iteration from save

    def train(self, training_path):
        def signal_handler():
            print("Stopping training. Please wait...")
            self.__stop_training = True
        # run the signal handler on a new thread so the print statements dont conflict with ones which are already running while interrupted
        signal.signal(signal.SIGINT, lambda _, __: threading.Timer(0.01, signal_handler).start())

        # Create some dummy data for training
        inputs = torch.randn(100000, 1000).to('cuda')
        labels = torch.randn(100000, 10).to('cuda')

        loss_function = nn.MSELoss() # define the loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01) # define the optimizer
        while not self.__stop_training and self.__epoch_iteration < self.__epochs:
            self.__epoch_iteration += 1
            print(f"Epoch {self.__epoch_iteration}/{self.__epochs}")

            #zero the gradients
            optimizer.zero_grad()

            outputs = self(inputs)
            loss = loss_function(outputs, labels)

             # Backward pass and update weights
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")
            
            
    
    def evaluate(self, evaluation_path):
        # create a sample input
        x = torch.randn(1, self.__input_size)

        # pass the input through the model and print the output
        output = self(x)
        print(output)
