import sys
import os
from SRSRTModel import SRSRTModel

class SRSRT:
    def __init__(self):
        pass

    def perform_command_line_args(self):
        if len(sys.argv) == 1:
            print("Please provide command line arguments")
            sys.exit(0)

        if sys.argv[1] == "prepare_training":
            if len(sys.argv) == 3 and os.path.isdir(sys.argv[2]):
                self.prepare_training(sys.argv[2])
                return
        elif sys.argv[1] == "prepare_evaluation":
            if len(sys.argv) == 3 and os.path.isdir(sys.argv[2]):
                self.prepare_evaluation(sys.argv[2])
                return
        elif sys.argv[1] == "train":
            if len(sys.argv) == 4 and os.path.isdir(sys.argv[3]):
                self.train(sys.argv[2], sys.argv[3])
                return
        elif sys.argv[1] == "evaluate":
            if len(sys.argv) == 4 and os.path.isdir(sys.argv[3]):
                self.evaluate(sys.argv[2], sys.argv[3])
                return

        print("Please provide correct command line arguments and ensure your path is valid")
        sys.exit(0)
        
    def prepare_training(self, training_path):
        pass

    def prepare_evaluation(self, evaluation_path):
        pass

    def train(self, model_name, training_path):
        model_path = f"{training_path}/{model_name}_model"
        self.__model = SRSRTModel()
        self.__model.load_model(model_path)
        self.__model.train()
        self.__model.save_model(model_path)
        
    def evaluate(self, model_name, evaluation_path):
        pass

if __name__ == "__main__":
    program = SRSRT()
    program.perform_command_line_args()
