import matplotlib.pyplot as plt
import pandas as pd

class LogValueObserver:
    def __init__(self):
        self.observations = []

    def observe(self, tag, log_file):
        with open(log_file, 'r') as file:
            for line in file:
                line = line.strip()
                
                if "["+tag+"]" in line:
                    value = float(line.split(">>>")[1].strip())
                    self.observations.append(value)

    def show_observations(self):
        y = pd.Series(self.observations)
        y = y.rolling(window=1000).mean().to_numpy()
        x = range(len(self.observations))  # Create x-axis values
        plt.plot(x, y)  # Plot the line graph
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Line Graph')
        plt.show()  # Display the graph
