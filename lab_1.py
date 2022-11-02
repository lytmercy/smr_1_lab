import numpy as np

# Set global variables of gray scale in ASCII
GRAYSCALE = "@%#*+=-:. "  # 10 levels of gray


def convert_array_to_ascii_art(img_array):
    """Function for transforming image value to ASCII char for creating ASCII art."""
    # Define global variable
    global GRAYSCALE
    # Set image height and width
    height, width = img_array.shape
    print(f"image shape: {width} x {height}")

    # Define ascii_image array
    ascii_image = []

    # Generate list of dimensions
    for j in range(height):
        # Add string to each row in ASCII image
        ascii_image.append("")
        for i in range(width):
            # Take one element from image of number
            image_tile = img_array[j][i]
            # Look up ascii char for making ASCII art
            gsval = GRAYSCALE[int(image_tile*9)]
            # append ascii char to string with space
            ascii_image[j] += gsval + " "

    return ascii_image


def print_image_in_console(image, what_the_image):
    """Function for printing image as ASCII art and presenting which image is (etalon or not)."""
    # Print image title (etalon or not)
    print(f"It's {what_the_image}")
    # Call function for transforming our number to ASCII art
    ascii_image = convert_array_to_ascii_art(image)
    # Print ASCII art per row
    for ascii_row in ascii_image:
        print(ascii_row)
    print("======================")


def noising_image(img_array, noise_success=0.73):
    """Function for making the noising images as ASCII art using Bernoulli noise."""
    # Take shape of image
    img_shape = img_array.shape
    # Create a binomial distribution with image shape
    bernoulli_noise = np.random.binomial(1, noise_success, img_shape)
    # Noising image
    noised_image = img_array + bernoulli_noise
    # Convert 2 to 0
    noised_image[noised_image == 2] = 0

    return noised_image


def recognition_method(target_image, etalon_images, possible_states):
    """Function for calculating recognition using the Bayesian recognition formula."""
    # Define probabilities result for each number
    prob_results = []
    # Calculate for all target_image tales & possible_states
    for i in possible_states:
        # Define result per number
        result = np.array([])
        for x_row, k_row in zip(target_image, etalon_images[i]):
            for x, k in zip(x_row, k_row):
                p = None
                one_minus_p = None
                # Calculate (x * k)ij ln(p)
                if x == k:
                    p = 0
                else:
                    p = 1 * np.log(0.1)
                # Calculate (1 * x * k)ij ln(1-p)
                if 1 == x == k:
                    one_minus_p = 0
                elif 1 != x or 1 != k:
                    one_minus_p = 1 * np.log(1 - 0.1)

                result = np.append(result, np.sum(np.array([p, one_minus_p])))
        prob_results.append(np.sum(result))

    return prob_results


def run_lab():
    """Lab-1
    Formulation of the problem:
    - Write a program to solve the problem of recognition symbols in a binary image with Bernoulli noise.
    Tasks:
    - Generate noise image;
    - Predict number in image;
    - Print all images (reference, noised and predicted).
    """
    # Loading the dataset
    Y_images = [[[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0],
                 [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0],
                 [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0],
                 [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0],
                 [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0],
                 [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]],
                ]
    # Define labels
    Y_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Convert to numpy array
    Y_images = np.array(Y_images)
    # Choose number for recognition
    chosen_number = 6
    # Choose an image from the dataset that is our chosen number
    chosen_image = Y_images[chosen_number]
    # Choose a label from the dataset that is our chosen number
    chosen_label = Y_labels[chosen_number]
    # Make results
    print("===Print Image in Console===")
    etalon_string = "Etalon Image"
    noise_string = "Noised Image"
    # ASCII image is a list of character strings
    print("Generating ASCII art...")
    print("======================")
    # View etalon image
    print_image_in_console(chosen_image, etalon_string)
    # Noise image
    noised_image = noising_image(chosen_image, noise_success=0.35)
    # View noised image
    print_image_in_console(noised_image, noise_string)
    # Made recognition of image
    result_probs = recognition_method(noised_image, Y_images, Y_labels)
    # Choose which number have the highest probabilities for program
    pred_label = np.argmax(result_probs)
    # View result
    print("++++++++++++++++++++++++++")
    print(f"Etalon image is {chosen_label}")
    print(f"Program predicted as {pred_label}")
    print("++++++++++++++++++++++++++")


if __name__ == '__main__':
    run_lab()
