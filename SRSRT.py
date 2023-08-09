import sys
import os
from SRSRTModel import SRSRTModel
from Trainer import Trainer
from Evaluator import Evaluator
from EvaluatorSingle import EvaluatorSingle
from LogValueObserver import LogValueObserver
import Vimeo90K
from SRSRTSettings import SRSRT_SETTINGS_DEFAULT

class SRSRT:
    def __init__(self):
        self.settings = SRSRT_SETTINGS_DEFAULT

    def run(self):
        if len(sys.argv) > 1:
            if sys.argv[1] == "prepare_data":
                if len(sys.argv) == 2:
                    if not os.path.isdir('./vimeo_septuplet/sequences'):
                        print("'./vimeo_septuplet/sequences' wasn't found. Please follow the guide in the README")
                        return
                    self.prepare_data()
                    return
            elif sys.argv[1] == "train":
                if len(sys.argv) == 4 and os.path.isfile(sys.argv[3]):
                    self.train(sys.argv[2], sys.argv[3])
                    return
            elif sys.argv[1] == "eval":
                if len(sys.argv) == 4 and os.path.isfile(sys.argv[3]):
                    self.evaluate(sys.argv[2], sys.argv[3])
                    return
            elif sys.argv[1] == "display_one":
                if (len(sys.argv) == 5 or len(sys.argv) == 6) and os.path.isdir(sys.argv[3]):
                    self.display_one(sys.argv[2], sys.argv[3], sys.argv[4], 4 if len(sys.argv) == 5 else int(sys.argv[5]))
                    return
            elif sys.argv[1] == "display":
                if len(sys.argv) == 4 and os.path.isfile(sys.argv[3]):
                    self.display(sys.argv[2], sys.argv[3])
                    return
            elif sys.argv[1] == "observe_log":
                if len(sys.argv) == 4 and os.path.isfile(sys.argv[3]):
                    self.observe_log(sys.argv[2], sys.argv[3])
                    return

        print("Please provide correct command line arguments")
        sys.exit(0)
        
    def prepare_data(self):
        Vimeo90K.prepare_data(self.settings["scale"])

    def train(self, model_name, training_path):
        model = SRSRTModel().to('cuda')
        model.load_model(model_name)

        trainer = Trainer(model, model_name, self.settings, training_path)

        trainer.load_training_state()
        trainer.train()
        trainer.save_training_state()
        
    def evaluate(self, model_name, evaluation_path):
        model = SRSRTModel().to('cuda')
        model.load_model(model_name)

        evaluator = Evaluator(model, model_name, self.settings, evaluation_path, False)
        evaluator.eval()

    def display(self, model_name, evaluation_path):
        model = SRSRTModel().to('cuda')
        model.load_model(model_name)

        evaluator = Evaluator(model, model_name, self.settings, evaluation_path, True)
        evaluator.eval()

    def display_one(self, model_name, evaluation_path, image_path, num_input_images):
        model = SRSRTModel().to('cuda')
        model.load_model(model_name)

        evaluator = EvaluatorSingle(model, model_name, self.settings, evaluation_path, image_path, num_input_images)
        evaluator.eval()

    def observe_log(self, tag, file_path):
        value_observer = LogValueObserver()
        value_observer.observe(tag, file_path)
        value_observer.show_observations()

if __name__ == "__main__":
    program = SRSRT()
    program.run()
