import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


# Define sigmoid function and its derivative

def sigmoid(z):
   return 1 / (1+ np.exp(-z))

def sigmoid_derivative(z):
   return sigmoid(z) * (1- sigmoid(z))


# Initialization of parameters

w = [0,0]   # Initial weights
threshold = 0.5   # Classification Threshold
bias = 1    # Initial Bias
learning_rate = 1    # Step Size for weight updates
max_iterations = 100   # Limit on training iterations 


# Inputs and labels
x = [
    [0.72, 0.82, -1],
    [0.91, -0.69, -1],
    [0.03, 0.93, -1],
    [0.12, 0.25, -1],
    [0.96, 0.47, -1],
    [0.8, -0.75, -1],
    [0.46, 0.98, -1],
    [0.66, 0.24, -1],
    [0.72, -0.15, -1],
    [0.35, 0.01, -1],
    [-0.11, 0.1, 1],
    [0.31, -0.96, 1],
    [0.0, -0.26, 1],
    [-0.43, -0.65, 1],
    [0.57, -0.97, 1],
    [-0.72, -0.64, 1],
    [-0.25, -0.43, 1],
    [-0.12, -0.9, 1],
    [-0.58, 0.62, 1],
    [-0.77, -0.76, 1],
]

# Plotting purpose dictionary
dict_x = {
    "0.72,0.82": '-1',
    "0.91,-0.69": '-1',
    "0.03,0.93": '-1',
    "0.12,0.25": '-1',
    "0.96,0.47": '-1',
    "0.8,-0.75": '-1',
    "0.46,0.98": '-1',
    "0.66,0.24": '-1',
    "0.72,-0.15": '-1',
    "0.35,0.01": '-1',
    "-0.11,0.1": '1',
    "0.31,-0.96": '1',
    "0.0,-0.26": '1',
    "-0.43,-0.65": '1',
    "0.57,-0.97": '1',
    "-0.72,-0.64": '1',
    "-0.25,-0.43": '1',
    "-0.12,-0.9": '1',
    "-0.58,0.62": '1',
    "-0.77,-0.76": '1',
}


# Helper function

# This function stores the coordinates for each point of the graph in their respective vectors.
# Extracts x and y coordinates of data points for plotting based on their label.
# Simplifies plotting points of different classes in different colors.
def get_points_of_color(data, label):
   x_coords = [float(point.split(",")[0]) for point in data.keys() if data[point] == label]
   y_coords = [float(point.split(",")[1]) for point in data.keys() if data[point] == label]
   return x_coords, y_coords


# Turn on interactive mode
# It will allow a real time updates to plot as the perceptron learns during training by enabling the intervative
plt.ion() # ion -> interactive on, ioff -> interactive off


"""
Train the Perceptron with a Sigmoid Activation Function
-------------------------------------------------------

Overview:
    This training process uses a perceptron model with the sigmoid activation function 
    to classify data points. The training is performed for a specified maximum number 
    of iterations or until the model achieves perfect classification.

Process:
    1. **Outer Loop**:
       - Iterates over the training process up to the maximum allowed iterations.
       - After each iteration, updates and visualizes the decision boundary.

    2. **Inner Loop**:
       - For each data point in the dataset:
           a. Compute the weighted sum of inputs plus bias.
           b. Pass the result through the sigmoid activation function.
           c. Compare the predicted output with the actual label.
           d. If prediction is correct → count as a "hit".
           e. If prediction is incorrect → update the weights and bias using:
                - Error term: (expected_output - predicted_output)
                - Sigmoid derivative: predicted_output * (1 - predicted_output)

    3. **Decision Boundary Update**:
       - After each outer iteration, the decision boundary is recalculated based 
         on updated weights and bias.
       - The boundary is plotted to visualize model progress.

Key Notes:
    - The sigmoid function maps input values to the range (0, 1), allowing smooth 
      probability-based predictions.
    - Weight updates are influenced by both the error magnitude and the gradient 
      from the sigmoid derivative.
""" 
# Ensure directory exists
os.makedirs("plot", exist_ok=True)

for k in range(1, max_iterations + 1):
    hits = 0
    print(f"\n-------- ITERATION: {k} --------")

    for i in range(len(x)):
        # Weighted sum
        weighted_sum = 0
        for j in range(len(x[i]) - 1):  # exclude the label
            weighted_sum += x[i][j] * w[j]

        # Add bias
        z = bias + weighted_sum
        output = sigmoid(z)

        # Classify based on threshold
        y_pred = 1 if output > threshold else -1

        # Update weights if incorrect
        if y_pred == x[i][2]:
            hits += 1
            answer = "Correct!"
        else:
            for j in range(len(w)):
                w[j] += learning_rate * (x[i][2] - output) * sigmoid_derivative(z) * x[i][j]
            bias += learning_rate * (x[i][2] - output) * sigmoid_derivative(z)
            answer = f"Error - Updating weights to: {w}, bias to: {bias:.3f}"

        print(answer)

    # Clear plot for update
    plt.clf()
    plt.title(f'Iteration {k}')
    plt.grid(False)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Plot decision boundary
    if w[1] != 0:
        xA, xB = 1, -1
        yA = (-w[0] * xA - bias) / w[1]
        yB = (-w[0] * xB - bias) / w[1]
    else:
        xA = xB = -bias / w[0]
        yA, yB = 1, -1

    # True decision boundary (black)
    plt.plot([0.77, -0.55], [-1, 1], color='k', linestyle='-', linewidth=1)

    # Current learned decision boundary (green dashed)
    plt.plot([xA, xB], [yA, yB], color='g', linestyle='--', linewidth=2)

    # Plot points
    x_coords_neg, y_coords_neg = get_points_of_color(dict_x, '-1')
    plt.plot(x_coords_neg, y_coords_neg, 'bo')  # blue for -1

    x_coords_pos, y_coords_pos = get_points_of_color(dict_x, '1')
    plt.plot(x_coords_pos, y_coords_pos, 'ro')  # red for +1

    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plot/Iteration_{k}_{timestamp}.png"
    plt.savefig(filename)
    print(f"Saved plot: {filename}")

    plt.pause(0.3)  # allow real-time update

    # Early stopping if all points are correct
    if hits == len(x):
        print("\nTraining complete - all points classified correctly.")
        break

      