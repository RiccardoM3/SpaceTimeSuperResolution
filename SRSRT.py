import sys
import os
from SRSRTModel import SRSRTModel
from Trainer import Trainer
import Vimeo90K
from SRSRTSettings import SRSRT_SETTINGS_DEFAULT

class SRSRT:
    def __init__(self):
        self.settings = SRSRT_SETTINGS_DEFAULT

    def run(self):
        if len(sys.argv) == 1:
            print("Please provide command line arguments")
            sys.exit(0)

        if sys.argv[1] == "prepare_data":
            if len(sys.argv) == 2:
                if not os.path.isdir('./vimeo_septuplet/sequences'):
                    print("'./vimeo_septuplet/sequences' wasn't found. Please follow the guide in the README")
                    return
                self.prepare_data()
                return
        elif sys.argv[1] == "train":
            if len(sys.argv) == 4 and os.path.isdir(sys.argv[3]):
                self.train(sys.argv[2], sys.argv[3])
                return
        elif sys.argv[1] == "evaluate":
            if len(sys.argv) == 4 and os.path.isdir(sys.argv[3]):
                self.evaluate(sys.argv[2], sys.argv[3])
                return

        print("Please provide correct command line arguments")
        sys.exit(0)
        
    def prepare_data(self):
        Vimeo90K.prepare_data(self.settings["scale"])

    def train(self, model_name, training_path):
        model_path = f"models/{model_name}_model"
        model = SRSRTModel().to('cuda')
        trainer = Trainer(model, self.settings, training_path)

        model.load_model(model_path)
        trainer.train()
        model.save_model(model_path)
        
    def evaluate(self, model_name, evaluation_path):
        model_path = f"models/{model_name}_model"
        model = SRSRTModel().to('cuda')
        model.load_model(model_path)
        model.evaluate(evaluation_path)

if __name__ == "__main__":
    program = SRSRT()
    program.run()
