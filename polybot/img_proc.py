import random
from pathlib import Path
from matplotlib.image import imread, imsave


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


class Img:

    def __init__(self, path):
        """
        Do not change the constructor implementation
        """
        self.path = Path(path)
        self.data = rgb2gray(imread(path)).tolist()

    def save_img(self):
        """
        Do not change the below implementation
        """
        new_path = self.path.with_name(self.path.stem + '_filtered' + self.path.suffix)
        imsave(new_path, self.data, cmap='gray')
        return new_path

    def blur(self, blur_level=16):

        height = len(self.data)
        width = len(self.data[0])
        filter_sum = blur_level ** 2

        result = []
        for i in range(height - blur_level + 1):
            row_result = []
            for j in range(width - blur_level + 1):
                sub_matrix = [row[j:j + blur_level] for row in self.data[i:i + blur_level]]
                average = sum(sum(sub_row) for sub_row in sub_matrix) // filter_sum
                row_result.append(average)
            result.append(row_result)

        self.data = result

    def contour(self):
        for i, row in enumerate(self.data):
            res = []
            for j in range(1, len(row)):
                res.append(abs(row[j-1] - row[j]))

            self.data[i] = res

    def rotate(self):
        # TODO remove the `raise` below, and write your implementation
        original_height = len(self.data)
        original_width = len(self.data[0])

        # Create a new list to hold the rotated image data
        rotated_data = []

        # Initialize the rotated_data with empty rows
        for i in range(original_width):
            rotated_data.append([0] * original_height)

        # Fill in the rotated_data by rotating the image 90 degrees clockwise
        for i in range(original_height):
            for j in range(original_width):
                rotated_data[j][original_height - 1 - i] = self.data[i][j]

        # Update self.data with the rotated image
        self.data = rotated_data

    def salt_n_pepper(self):
        # TODO remove the `raise` below, and write your implementation
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                random_value = random.random()  # Generate a random number between 0 and 1
                if random_value < 0.2:
                    self.data[i][j] = 255  # Salt (white)
                elif random_value > 0.8:
                    self.data[i][j] = 0  # Pepper (black)
                # If between 0.2 and 0.8, the pixel remains unchanged

    def concat(self, other_img, direction='horizontal'):
        # Check if both images have the same height
        if len(self.data) != len(other_img.data):
            raise RuntimeError("Images do not have the same height and cannot be concatenated.")

        # Concatenate the images row by row manually
        concatenated_data = []
        for i in range(len(self.data)):
            concatenated_data.append(self.data[i] + other_img.data[i])

        # Update self.data with the concatenated image
        self.data = concatenated_data

    def segment(self):
        # TODO remove the `raise` below, and write your implementation
        # Iterate over each row
        for i in range(len(self.data)):
            # Iterate over each pixel in the row
            for j in range(len(self.data[i])):
                # Replace pixel value based on intensity
                if self.data[i][j] > 100:
                    self.data[i][j] = 255  # White
                else:
                    self.data[i][j] = 0  # Black
