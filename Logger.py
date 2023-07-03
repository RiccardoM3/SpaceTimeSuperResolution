import os
from datetime import datetime

class Logger:
    def __init__(self, log_name):
        self.log_name = log_name

        if not os.path.exists("logs"):
            os.makedirs("logs")

        self.log_file = open(f"logs/{self.get_current_datetime()}_{self.log_name}.txt", 'a')

    def log(self, tag, log):
        log = f"{self.get_current_datetime()} [{tag}] >>> {log}"
        print(log)
        self.log_file.write(log + "\n")

    def get_current_datetime(self):
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
