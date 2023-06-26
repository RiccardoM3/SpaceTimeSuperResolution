import os
import numpy as np
import matplotlib.pyplot as plt
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

                loss = self.train_one_batch(train_data)
                
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

    def train_one_batch(self, train_data):
        inputs = train_data["LRs"].to('cuda')   #[5, 4, 3, 64, 96]
        targets = train_data['HRs'].to('cuda')  #[5, 7, 3, 256, 384]
        
        loss = 0
        for i in range(len(inputs)-1):
            #zero the gradients
            self.optimiser.zero_grad()
            
            context = inputs
            input_frames = inputs[:, i:i+2, :, :, :]
            target_frames = targets[:, 2*i:2*(i+1)+1, :, :, :]
            output_frames = self.model(context, input_frames, (i, i+1))

            # display the first output of the batch
            self.observe_sequence(input_frames[0], output_frames[0], target_frames[0])

            loss += self.loss_function(output_frames, target_frames)
            loss.backward()
            self.optimiser.step()

        # return avg loss
        return loss/(len(inputs)-1)

    def observe_sequence(self, input, output, target):
        num_outputs = output.size()[0]
        fig, axs = plt.subplots(3, num_outputs, figsize=(20,5), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(num_outputs):
            
            current_target = target[i].permute(1, 2, 0).cpu().numpy()
            current_output = output[i].permute(1, 2, 0).cpu().detach().numpy()

            if i % 2 == 0:
                current_input = input[i//2].permute(1, 2, 0).cpu().numpy()
                current_input = np.repeat(np.repeat(current_input, self.settings["scale"], axis=0), self.settings["scale"], axis=1)
                axs[0][i].imshow(current_input)
                
            axs[0][i].axis('off')
            axs[1][i].imshow(current_target)
            axs[1][i].axis('off')
            axs[2][i].imshow(current_output)
            axs[2][i].axis('off')

        plt.show()

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


        
