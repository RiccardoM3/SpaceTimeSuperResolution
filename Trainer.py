import os
import cv2
import signal
import threading
import torch
from Loss import CharbonnierLoss
import Vimeo90K

class Trainer:
    def __init__(self, model, model_name, settings, train_list_path):
        self.model = model
        self.model_name = model_name
        self.settings = settings
        self.train_list_path = train_list_path

        self.__stop_training = False

        self.train_set = Vimeo90K.create_dataset(self.settings, self.train_list_path)
        self.train_loader = Vimeo90K.create_dataloader(self.train_set, self.settings["batch_size"])

        self.loss_function = CharbonnierLoss().to('cuda')
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.epoch = 0
        self.iter = 0

    def train(self):
        self.model.train()

        def signal_handler():
            print("Stopping training. Please wait...")
            self.__stop_training = True
        # run the signal handler on a new thread so the print statements dont conflict with ones which are already running while interrupted
        signal.signal(signal.SIGINT, lambda _, __: threading.Timer(0.01, signal_handler).start())

        while not self.__stop_training and self.epoch < self.settings['num_epochs']:
            print(f"Epoch {self.epoch}/{self.settings['num_epochs']}")

            for _, train_data in enumerate(self.train_loader):
                if self.iter > self.settings["max_iters_per_epoch"]:
                    break

                loss = self.train_one_datapoint(train_data)
                
                current_iter = self.iter
                self.iter += 1

                if current_iter != 0 and current_iter % self.settings["training_save_interval"] == 0:
                    print(f"Iter: {current_iter}")
                    print(f"Loss: {loss.item()}")
                    self.model.save_model(self.model_name)
                    self.save_training_state()

            self.epoch += 1
            self.iter = 0

            self.model.save_model(self.model_name)
            self.save_training_state()

        self.model.save_model(self.model_name)
        self.save_training_state()

    def train_one_datapoint(self, train_data):
        inputs = train_data["LRs"].to('cuda')   #[5, 4, 3, 64, 112]
        targets = train_data['HRs'].to('cuda')  #[5, 7, 3, 256, 448]
        
        # cv2.imshow("window", inputs[0][0].permute(1, 2, 0).cpu().numpy())
        # cv2.waitKey(0)
        # cv2.imshow("window", targets[0][0].permute(1, 2, 0).cpu().numpy())
        # cv2.waitKey(0)

        #zero the gradients
        self.optimiser.zero_grad()
        outputs = self.model(inputs)

        loss = self.loss_function(outputs, targets)
        loss.backward()
        self.optimiser.step()

        return loss

    def load_training_state(self):
        training_state_path = f"training_states/{self.model_name}_state.pth"

        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            self.epoch = training_state["epoch"]
            self.iter = training_state["iter"]
            self.optimiser.load_state_dict(training_state["optimiser"])
            print("Loaded training state:")
            print(training_state)
        else:
            print(f"{training_state_path} doesn't exist.")

    def save_training_state(self):
        training_state_path = f"training_states/{self.model_name}_state.pth"
        
        training_state = {
            "epoch": self.epoch,
            "iter": self.iter,
            "optimiser": self.optimiser.state_dict()
        }

        if not os.path.exists("training_states"):
            os.makedirs("training_states")

        torch.save(training_state, training_state_path)
        print("Saved training state")


        
