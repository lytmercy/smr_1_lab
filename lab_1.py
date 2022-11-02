from PIL import Image
import numpy as np
import mnist

# Set global variables
GRAYSCALE = "@%#*+=-:. "  # 10 levels of gray


def convert_array_to_ascii_art(img_array, cols, scale):
    """"""
    global GRAYSCALE
    height, width = img_array.shape
    print(f"image shape: {width} x {height}")

    # tile_width = Width/cols
    # tile_height = tile_width/scale

    # rows = int(Height/tile_height)

    # print(f"cols: {cols}, rows: {rows}")
    # print(f"tile dims: {tile_width} x {tile_height}")

    # if cols > Width or rows > Height:
    #     print("Image too small for specified cols!")
    #     exit(0)

    ascii_image = []

    # Generate list of dimensions
    for j in range(height):
        # y1 = int(j*tile_height)
        # y2 = int((j+1)*tile_height)
        # correct last tile
        # if j == rows-1:
        #     y2 = Height
        # append an empty string
        ascii_image.append("")
        for i in range(width):
            # crop image to tile
            # x1 = int(i*tile_width)
            # x2 = int((i+1)*tile_width)
            # correct last tile
            # if i == cols-1:
            #     x2 = Width
            # crop image to extract tile
            # image_tile = img_array.crop((x1, y1, x2, y2))
            # get average luminance
            # avg = int(get_avrg_grayscale(image))
            image_tile = img_array[j][i]
            # look up ascii char
            gsval = GRAYSCALE[int(image_tile*9)]

            # append ascii char to string
            ascii_image[j] += gsval + " "

    return ascii_image


def print_image_in_console(image, cols, scale, what_the_image):
    """"""
    print(f"It's {what_the_image}")
    ascii_image = convert_array_to_ascii_art(image, cols, scale)
    for ascii_row in ascii_image:
        print(ascii_row)
    print("======================")


def noising_image(img_array, noise_success=0.73):
    """"""
    # Take shape of image
    img_shape = img_array.shape
    # Create a binomial distribution.
    bernoulli_noise = np.random.binomial(1, noise_success, img_shape)
    # Noising image
    noised_image = img_array + bernoulli_noise
    # Convert 2 to 0
    noised_image[noised_image == 2] = 0

    return noised_image


def recognition_method(target_image, etalon_images, possible_states):
    """"""
    prob_results = []
    # Calculate for all target_image tales & possible_states
    for i in possible_states:
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
    Y_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Convert to numpy array
    Y_images = np.array(Y_images)
    # Choose number for recognition
    chosen_number = 6
    #
    chosen_image = Y_images[chosen_number]
    chosen_label = Y_labels[chosen_number]
    print("===Print Image in Console===")
    # image = Image.fromarray(Y_images[6])
    etalon_string = "Etalon Image"
    noise_string = "Noised Image"
    # ASCII image is a list of character strings
    scale = 1
    cols = chosen_image.shape[1]
    print("Generating ASCII art...")
    print("======================")
    # Printing etalon image
    print_image_in_console(chosen_image, cols, scale, etalon_string)
    # Noise image
    noised_image = noising_image(chosen_image, noise_success=0.35)
    # Printing noised image
    print_image_in_console(noised_image, cols, scale, noise_string)

    # Made recognition of image
    result_probs = recognition_method(noised_image, Y_images, Y_labels)
    pred_label = np.argmax(result_probs)
    print("++++++++++++++++++++++++++")
    print(f"Etalon image is {chosen_label}")
    print(f"Program predicted as {pred_label}")
    print("++++++++++++++++++++++++++")


if __name__ == '__main__':
    run_lab()
