import cv2
import signal
import threading
import torch
from Loss import CharbonnierLoss
import Vimeo90K

class Trainer:
    def __init__(self, model, settings, train_list_path):
        self.model = model
        self.settings = settings
        self.train_list_path = train_list_path

        self.__stop_training = False
        self.__epoch = 0

    def train(self):
        self.model.train()

        def signal_handler():
            print("Stopping training. Please wait...")
            self.__stop_training = True
        # run the signal handler on a new thread so the print statements dont conflict with ones which are already running while interrupted
        signal.signal(signal.SIGINT, lambda _, __: threading.Timer(0.01, signal_handler).start())

        train_set = Vimeo90K.create_dataset(self.settings, self.train_list_path)
        train_loader = Vimeo90K.create_dataloader(train_set, self.settings["batch_size"])

        loss_function = CharbonnierLoss().to('cuda') # define the loss function # TODO
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01) # define the optimizer #TODO
        while not self.__stop_training and self.__epoch < self.settings['num_epochs']:
            self.__epoch += 1
            print(f"Epoch {self.__epoch}/{self.settings['num_epochs']}")

            for _, train_data in enumerate(train_loader):

                inputs = train_data["LRs"].to('cuda')   #[5, 4, 3, 64, 112]
                targets = train_data['HRs'].to('cuda')  #[5, 7, 3, 256, 448]
                
                # cv2.imshow("window", inputs[0][0].permute(1, 2, 0).cpu().numpy())
                # cv2.waitKey(0)
                # cv2.imshow("window", targets[0][0].permute(1, 2, 0).cpu().numpy())
                # cv2.waitKey(0)

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
        
