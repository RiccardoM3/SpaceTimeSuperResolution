import matplotlib.pyplot as plt
import signal
import threading
import time

from model import Vimeo90K

from skimage.metrics import peak_signal_noise_ratio

class EvaluatorFPS:
    def __init__(self, model, model_name, settings, test_list_path):
        self.model = model
        self.model_name = model_name
        self.settings = settings
        self.test_list_path = test_list_path

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

        last_time = time.time()

        while not self.stop_evaluating:
            fps_total = 0;
            fps_count = 0;
            for _, test_data in enumerate(self.test_loader):
                
                self.eval_one_sequence(test_data)
                
                current_time = time.time()
                time_taken = current_time - last_time
                last_time = current_time

                num_input_frames = test_data["LRs"].size(1)
                fps = num_input_frames / time_taken
                print(f"FPS: {fps}")

                fps_total += fps
                fps_count += 1

            print(f"Average FPS: {fps_total / fps_count}")
            
            self.stop_evaluating = True


    def eval_one_sequence(self, test_data):
        inputs = test_data["LRs"].to('cuda')   #[1, 4, 3, 64, 96]

        input_sequence = inputs[0]

        for i in range(len(input_sequence)-1):
            j = i+1

            input_frames = inputs[:, i:j+1, :, :, :]
            output_frames = self.model(inputs, input_frames, (i, j), skip_encoder=(i!=0))
            # the output_frames would then be shown directly to the screen

