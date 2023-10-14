import numpy as np
import matplotlib.pyplot as plt
import signal
import threading
import torch

import Vimeo90K
from skimage.metrics import peak_signal_noise_ratio

class Evaluator:
    def __init__(self, model, model_name, settings, test_list_path, display_images):
        self.model = model
        self.model_name = model_name
        self.settings = settings
        self.test_list_path = test_list_path
        self.display_images = display_images

        self.stop_evaluating = False

        self.test_set = Vimeo90K.create_dataset(self.settings, Vimeo90K.read_image_paths_from_file(self.test_list_path))
        self.test_loader = Vimeo90K.create_dataloader(self.test_set, 1)


    def eval(self):
        
        def signal_handler():
            print("Stopping evalation. Please wait...")
            self.stop_evaluating = True
            plt.close()

        # run the signal handler on a new thread so the print statements dont conflict with ones which are already running while interrupted
        signal.signal(signal.SIGINT, lambda _, __: threading.Timer(0.01, signal_handler).start())

        PSNRS = []
        while not self.stop_evaluating:
            for _, test_data in enumerate(self.test_loader):
                PSNR = self.eval_one_sequence(test_data)
                PSNRS.append(PSNR)
            
            self.stop_evaluating = True
        
        total_avg_PSNR = sum(PSNRS)/len(PSNRS)
        print(f"Total Average PSNR: {total_avg_PSNR}")


    def eval_one_sequence(self, test_data):
        inputs = test_data["LRs"].to('cuda')   #[1, 4, 3, 64, 96]
        targets = test_data['HRs'].to('cuda')  #[1, 7, 3, 256, 384]

        batch_size = len(inputs)

        i = 2 # always operate on the last 2, non-compressed frames
        j=i+1

        input_frames = inputs[:, i:j+1, :, :, :]
        target_frames = targets[:, 2*i:2*j+1, :, :, :]
        self.model.calc_encoder(inputs)
        output_frames = self.model(input_frames, [(i, j)] * batch_size)

        input_frames = inputs[:, i:j+1, :, :, :]
        output_frames = self.model(input_frames, [(i, j)] * batch_size)
        output_frames = torch.clamp(output_frames, 0, 1)

        # calc psnrs
        PSNRs = []
        for (output_im, target_im) in zip(output_frames[0], target_frames[0]):
            output_im = output_im.permute(1, 2, 0).cpu().detach().numpy()
            target_im = target_im.permute(1, 2, 0).cpu().detach().numpy()
            PSNRs.append(peak_signal_noise_ratio(target_im, output_im))
        avg_PSNR = sum(PSNRs)/len(PSNRs)
        print(f"Current PSNR: {avg_PSNR}")

        # Note: commented out since we only interpolate the last 2 images now. the display code needs to be rewritten
        # display the first sequence of the batch
        # if self.display_images:
        #     self.observe_sequence(input_sequence, output_sequence, target_sequence)

        return avg_PSNR


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
