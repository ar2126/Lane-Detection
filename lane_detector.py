import cv2
import numpy as np

def gaussian_2d(x, sig=1):
    gauss = np.array([])
    mid = x // 2    # find the midsection of the Gaussian length
    for i in range(-mid, mid+1):
        for j in range(-mid, mid+1):
            val = (1 / (sig * np.sqrt(2 * np.pi))) * (1 / (np.exp(((i ** 2) + (j ** 2)) / (2 * sig ** 2)))) # apply the Gaussian formula on the positive and negative x-values
            gauss = np.append(gauss, val)
    gauss = np.reshape(gauss, (x, x))   # transform the list of 2D Gaussian values into an NxN filter
    return gauss

def filter_gaussian(img):
    kernel = gaussian_2d(5)
    result = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)#cv2.filter2D(img, -1, kernel)
    return result


def sobel_edge_detection(image, filter):
    sobel_edge_x = cv2.filter2D(image, -1, filter)

    sobel_edge_y = cv2.filter2D(image, -1, np.flip(filter.T, axis=0))

    sobel_full = np.hypot(sobel_edge_x, sobel_edge_y)    # broadcasts the pixel at the edge detection in the x and y images to find a common ground
    sobel_full = sobel_full / sobel_full.max() * 255

    theta = np.arctan2(sobel_edge_y, sobel_edge_x)


    return sobel_full.astype('uint8'), theta # return magnitude and direction


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

if __name__ == '__main__':
    original_img = cv2.imread('images/dashcam.png')
    gry_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    gauss = filter_gaussian(gry_img)

    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel, theta = sobel_edge_detection(gauss, sobel_filter)

    nonmax = non_max_suppression(sobel, theta)
    double_threshold = threshold(nonmax, 20, 5)
    canny_image = hysteresis(double_threshold)
    #cv2.imshow('image', canny_highway)
    #cv2.waitKey(0)
    cv2.imwrite('images/canny.jpg', canny_image)