import numpy as np
import matplotlib.pyplot as plt
import signal
import threading
import Vimeo90K
import torch

class Evaluator:
    def __init__(self, model, model_name, settings, test_list_path):
        self.model = model
        self.model_name = model_name
        self.settings = settings
        self.test_list_path = test_list_path

        self.stop_evaluating = False

        self.test_set = Vimeo90K.create_dataset(self.settings, self.test_list_path)
        self.test_loader = Vimeo90K.create_dataloader(self.test_set, 1)


    def eval(self):
        
        def signal_handler():
            print("Stopping evalation. Please wait...")
            self.stop_evaluating = True
            plt.close()

        # run the signal handler on a new thread so the print statements dont conflict with ones which are already running while interrupted
        signal.signal(signal.SIGINT, lambda _, __: threading.Timer(0.01, signal_handler).start())

        while not self.stop_evaluating:
            for _, test_data in enumerate(self.test_loader):
                self.eval_one_sequence(test_data)

    def eval_one_sequence(self, test_data):
        inputs = test_data["LRs"].to('cuda')   #[1, 4, 3, 64, 96]
        targets = test_data['HRs'].to('cuda')  #[1, 7, 3, 256, 384]

        input_sequence = inputs[0]
        output_sequence = torch.tensor([]).to('cuda')
        target_sequence = targets[0]

        for i in range(len(input_sequence)-1):
            j = i+1

            input_frames = inputs[:, i:j+1, :, :, :]
            output_frames = self.model(inputs, input_frames, (i, j), skip_encoder=(i!=0))

            #take the first 2 out of 3 outputs for all but the last frame pair. for the last, take all 3 outputs
            if j == len(input_sequence)-1:
                output_sequence = torch.cat((output_sequence, output_frames[0]), dim=0)
            else:
                output_sequence = torch.cat((output_sequence, output_frames[0, :2]), dim=0)

        # display the first sequence of the batch
        print(input_sequence.shape, output_sequence.shape, target_sequence.shape)
        self.observe_sequence(input_sequence, output_sequence, target_sequence)

    def observe_sequence(self, input, output, target):
        num_outputs = output.size()[0]
        fig, axs = plt.subplots(3, num_outputs, figsize=(18,6), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(num_outputs):
            
            current_output = output[i].permute(1, 2, 0).cpu().detach().clone().numpy()
            current_target = target[i].permute(1, 2, 0).cpu().clone().numpy()

            if i % 2 == 0:
                current_input = input[i//2].permute(1, 2, 0).cpu().numpy()
                current_input = np.repeat(np.repeat(current_input, self.settings["scale"], axis=0), self.settings["scale"], axis=1)
                axs[0][i].imshow(current_input)
                
            axs[0][i].axis('off')
            axs[1][i].imshow(current_output)
            axs[1][i].axis('off')
            axs[2][i].imshow(current_target)
            axs[2][i].axis('off')

        plt.show()
