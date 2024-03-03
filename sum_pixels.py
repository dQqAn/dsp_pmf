from PIL import Image
import os
import datetime


def sum_images(image_paths):
    # Open all the images
    # images = [Image.open(path) for path in image_paths]
    # images = [Image.open(path).convert("L") for path in image_paths]
    images = [Image.open(path).convert("RGB") if Image.open(path).mode == "RGB" else Image.open(path).convert("L") for
              path in image_paths]

    # Get the dimensions of the first image
    # width, height = images[0].size
    min_width, min_height = min(image.size[0] for image in images), min(image.size[1] for image in images)

    # Resize images to the minimum width and height
    images = [image.resize((min_width, min_height)) for image in images]

    mode = images[0].mode
    # Create a new blank image to store the sum
    summed_image = Image.new(mode, (min_width, min_height))
    # summed_image = Image.new("RGB", (width, height), (0, 0, 0))
    # summed_image = Image.new("L", (width, height), 0)

    # Iterate over each pixel in each image and sum them
    for x in range(min_width):
        for y in range(min_height):
            pixel_sum = [0] * (3 if mode == "RGB" else 1)
            for image in images:
                pixel = image.getpixel((x, y))
                if mode == "RGB":
                    if isinstance(pixel, int):  # If grayscale image, convert to RGB
                        pixel = (pixel, pixel, pixel)
                    pixel_sum = [sum(pixels) for pixels in zip(pixel_sum, pixel)]
                else:
                    if isinstance(pixel, tuple):  # If RGB image, convert to Grayscale
                        pixel = int(0.2989 * pixel[0] + 0.5870 * pixel[1] + 0.1140 * pixel[2])
                    pixel_sum[0] += pixel
            averaged_pixel = [int(p / len(images)) for p in pixel_sum]
            summed_image.putpixel((x, y), tuple(averaged_pixel) if mode == "RGB" else averaged_pixel[0])

    return summed_image


def show_and_save(image_paths):
    summed_image = sum_images(image_paths.values())

    current_directory = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(current_directory, "outputs")
    os.makedirs(save_directory, exist_ok=True)
    # desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S-%f")
    summed_image.save(os.path.join(save_directory, f'{formatted_time}.jpg'))

    summed_image.show()

    output_image = f'outputs/{formatted_time}.jpg'
    return output_image


if __name__ == '__main__':
    image_paths = {
        # "apples": "images/apples.jpg",
        "bird": "images/bird.jpg",
        "car": "images/car.jpg",
        # "cat_1": "images/cat_1.jpg",
        # "cat_2": "images/cat_2.jpg",
        # "dog": "images/dog.jpg",
        # "flower_1": "images/flower_1.png",
        # "flower_2": "images/flower_2.png",
        # "hacker": "images/hacker.jpg",
        # "lena": "images/lena.png"
    }

    show_and_save(image_paths)
