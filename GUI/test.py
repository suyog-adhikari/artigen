import numpy as np
from scipy.signal import convolve2d

# Given input and filter
input_matrix = np.array([[1, 3], [2, 4]])
filter_matrix = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])

# Parameters
padding = 1
stride = 2

# Transpose convolution operation
def transpose_convolution(input_matrix, filter_matrix, padding, stride):
    input_height, input_width = input_matrix.shape
    filter_height, filter_width = filter_matrix.shape

    # Calculate output size
    output_height = (input_height - 1) * stride - 2 * padding + filter_height
    output_width = (input_width - 1) * stride - 2 * padding + filter_width

    # Perform transpose convolution using convolve2d
    output_matrix = convolve2d(input_matrix, np.rot90(filter_matrix, 2), mode='full')

    # Adjust for padding
    output_matrix = output_matrix[padding:output_height + padding, padding:output_width + padding]

    return output_matrix

# Perform transpose convolution with the given parameters
output_matrix = transpose_convolution(input_matrix, filter_matrix, padding, stride)

# Trim to achieve a 6x6 matrix
output_matrix = output_matrix[:6, :6]

# Display the result
print("Input Matrix:")
print(input_matrix)
print("\nFilter Matrix:")
print(filter_matrix)
print("\nOutput Matrix:")
print(output_matrix)
