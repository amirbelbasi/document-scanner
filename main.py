from utils import *
import numpy as np

def warp_perspective(img, transform_matrix, output_width, output_height):
    width, height, _ = img.shape
    result = np.zeros((output_width, output_height, _), dtype='int')

    for i in range(width):
        for j in range(height):
            tmp = np.dot(transform_matrix, [i, j, 1])
            x = int(tmp[0] / tmp[2])
            y = int(tmp[1] / tmp[2])
            if 0 <= x < output_width and 0 <= y < output_height:
                result[x, y] = img[i, j]

    return result

def grayscale_filter(img):
    transform_matrix = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]], dtype='float')
    return Filter(img, transform_matrix)

def crazy_filter(img):
    transform_matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype='float')
    return Filter(img, transform_matrix)

def custom_filter(img):
    transform_matrix = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype='float')
    transformed_image = Filter(img, transform_matrix)
    show_image(transformed_image, "Custom Filter")

    inverse_matrix = np.linalg.inv(transform_matrix)
    reversed_image = Filter(transformed_image, inverse_matrix)
    show_image(reversed_image, "Reversed Image", False)

def scale_img(img, scale_width, scale_height):
    width, height, _ = img.shape
    result = np.zeros((width * scale_width, height * scale_height, _), dtype='int')

    for i in range(width * scale_width):
        for j in range(height * scale_height):
            result[i, j] = img[int(i / scale_width), int(j / scale_height)]

    return result

def crop_img(img, start_row, end_row, start_column, end_column):
    return img[start_column:end_column, start_row:end_row]

if __name__ == "__main__":
    image_matrix = get_input('pic.jpg')

    width, height = 300, 400

    show_image(image_matrix, title="Input Image")

    pts1 = np.float32([[105, 215], [370, 180], [160, 645], [485, 565]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    perspective_matrix = getPerspectiveTransform(pts1, pts2)

    warped_image = warp_perspective(image_matrix, perspective_matrix, width, height)
    show_warp_perspective(warped_image)
    
    gray_scale_pic = grayscale_filter(warped_image)
    show_image(gray_scale_pic, title="Gray Scaled")

    crazy_image = crazy_filter(warped_image)
    show_image(crazy_image, title="Crazy Filter")

    custom_filter(warped_image)
    
    cropped_image = crop_img(warped_image, 50, 300, 50, 225)
    show_image(cropped_image, title="Cropped Image")

    scaled_image = scale_img(warped_image, 2, 3)
    show_image(scaled_image, title="Scaled Image")
