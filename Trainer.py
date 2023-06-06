import signal
import threading
import torch
import torch.nn as nn
from DataLoader import create_dataloader, create_dataset
from Loss import CharbonnierLoss

class Trainer:
    def __init__(self, model, train_path):
        self.model = model
        self.train_path = train_path

        self.__stop_training = False
        self.__epoch = 0
        self.__num_epochs = 1000

    def train(self):
        self.model.train()

        def signal_handler():
            print("Stopping training. Please wait...")
            self.__stop_training = True
        # run the signal handler on a new thread so the print statements dont conflict with ones which are already running while interrupted
        signal.signal(signal.SIGINT, lambda _, __: threading.Timer(0.01, signal_handler).start())

        train_set = create_dataset(self.train_path)
        train_loader = create_dataloader(train_set)

        loss_function = CharbonnierLoss().to('cuda') # define the loss function # TODO
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01) # define the optimizer #TODO
        while not self.__stop_training and self.__epoch < self.__num_epochs:
            self.__epoch += 1
            print(f"Epoch {self.__epoch}/{self.__num_epochs}")

            for _, train_data in enumerate(train_loader):

                inputs = train_data["LRs"].to('cuda')
                targets = train_data['HRs'].to('cuda')

                print(inputs.size())
                print(targets.size())
                
                #zero the gradients
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                
                print(f"Loss: {loss.item()}")

        self.save_training_state()

    def save_training_state():
        pass #TODO
        
