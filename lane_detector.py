import argparse
import cv2
import sys
import numpy as np

from halo import Halo

VIDEO_FILES = ['mov', 'mp4']


def gaussian_2d(x, sig=1):
    gauss = np.array([])
    mid = x // 2    # find the midsection of the Gaussian length
    for i in range(-mid, mid+1):
        for j in range(-mid, mid+1):
            # apply the Gaussian formula on the positive and negative x-values
            val = (1 / (sig * np.sqrt(2 * np.pi))) * \
                (1 / (np.exp(((i ** 2) + (j ** 2)) / (2 * sig ** 2))))
            gauss = np.append(gauss, val)
    # transform the list of 2D Gaussian values into an NxN filter
    gauss = np.reshape(gauss, (x, x))
    return gauss


def filter_gaussian(img):
    kernel = gaussian_2d(5)
    # cv2.filter2D(img, -1, kernel)
    result = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    return result


def sobel_edge_detection(image, filter):
    sobel_edge_x = cv2.filter2D(image, -1, filter)

    sobel_edge_y = cv2.filter2D(image, -1, np.flip(filter.T, axis=0))

    # broadcasts the pixel at the edge detection in the x and y images to find a common ground
    sobel_full = np.hypot(sobel_edge_x, sobel_edge_y)
    sobel_full = sobel_full / sobel_full.max() * 255

    theta = np.arctan2(sobel_edge_y, sobel_edge_x)
    return sobel_full.astype('uint8'), theta  # return magnitude and direction


def non_max_suppression(gradient_magnitude, gradient_direction):
    row, col = gradient_magnitude.shape
    output = np.zeros((row, col), dtype=np.int32)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    pi = 180
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < pi/8) or (pi - (pi/8) <= angle[i, j] <= pi):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif (pi/8 <= angle[i, j] < (pi/4) + (pi/8)):
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif ((pi/4) + (pi/8) <= angle[i, j] < (pi/2) + (pi/8)):
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif ((pi/2) + (pi/8) <= angle[i, j] < pi - (pi/8)):
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                output[i, j] = gradient_magnitude[i, j]
            else:
                output[i, j] = 0
    return output.astype('uint8')


def threshold(img, strong, weak, nonrelevant=100):
    output = np.zeros(img.shape)

    strong_row, strong_col = np.where(img >= strong)
    weak_row, weak_col = np.where((img <= strong) & (img >= weak))

    output[strong_row, strong_col] = 255
    output[weak_row, weak_col] = nonrelevant

    return output


def hysteresis(img, nonrelevant=100, strong=255):
    row, col = img.shape
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if img[i, j] == nonrelevant:
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                    or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def process_img(original_img):
    original_img = cv2.imread('images/dashcam.png')
    gry_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    gauss = filter_gaussian(gry_img)

    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel, theta = sobel_edge_detection(gauss, sobel_filter)

    nonmax = non_max_suppression(sobel, theta)
    double_threshold = threshold(nonmax, 20, 5)
    canny_image = hysteresis(double_threshold)
    
    canny_highway = hysteresis(double_threshold)

    filtered_canny = apply_filter(canny_highway)
    return filtered_canny

'''
TODO: edit dimensions and fine-tune based on the video img size
'''
def apply_filter(img):
    height, width = img.shape
    left_region = width * 0.15
    right_region = width * 0.85

    polygons = np.array([
        [(int(left_region), height), (int(right_region), height), (int(width/2), 245)]
    ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros(img.shape)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv2.bitwise_and(img, mask)
    return segment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect lanes in an image, video file, or webcam using canny edge detection and hough transform.')
    parser.add_argument('-i', '--input', type=str, nargs='?',
                        help='a file to process')

    args = parser.parse_args()
    if args.open:
        filename = args.open
        ext = filename.split('.')[1]
        if ext in VIDEO_FILES:
            print("Processing {} as a video file".format(filename))
            cap = cv2.VideoCapture(filename)
            frames = 1
            spinner = Halo(text='Processing frame 1', spinner='arc')
            spinner.start()
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    processed = process_img(frame)
                    cv2.imwrite('frames/frame{}.jpg'.format(frames), processed)
                    frames += 1
                    spinner.text = 'Proccessing frame {}'.format(frames)
                else:
                    spinner.stop()
                    break
        else:
            print("Processing {} as a image file".format(filename))
            original_img = cv2.imread(filename)
            processed = process_img(original_img)
            cv2.imwrite('images/processed.jpg', processed)
     
    #cv2.imshow('image', filtered_canny)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)