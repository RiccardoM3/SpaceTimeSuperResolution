import signal
import threading
import torch
import torch.nn as nn
import os

# define the model architecture
class SRSRTModel(nn.Module):
    def __init__(self):
        super(SRSRTModel, self).__init__()

        # initialisations
        self.__stop_training = False

        # define the layers
        self.__input_size = 10
        self.__output_size = 1
        self.__hidden_size = 5
        self.hidden = nn.Linear(self.__input_size, self.__hidden_size)
        self.output = nn.Linear(self.__hidden_size, self.__output_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x
        
    def load_model(self, path):
        if os.path.exists(f"{path}.pth"):
            self.load_state_dict(torch.load(f"{path}.pth"))
        else:
            print(f"{path}.pth doesn't exist.")

    def save_model(self, path):
        os.makedirs("models")
        torch.save(self.state_dict(), f"{path}.pth")

    def train(self, training_path):
        def signal_handler():
            print("Stopping training. Please wait...")
            self.__stop_training = True
        # run the signal handler on a new thread so the print statements dont conflict with ones which are already running while interrupted
        signal.signal(signal.SIGINT, lambda _, __: threading.Timer(0.01, signal_handler).start())

        # TODO
        criterion = nn.MSELoss() # define the loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01) # define the optimizer
        while not self.__stop_training:
            print("train")
    
    def evaluate(self, evaluation_path):
        # create a sample input
        x = torch.randn(1, self.__input_size)

        # pass the input through the model and print the output
        output = self(x)
        print(output)
