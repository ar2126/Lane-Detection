'''
    Name: lane_detector.py

    Description: A Python file that uses OpenCV to analyze an image or video file containing dashcam footage, and
    determines what lanes exist (if any) within the media

    Authors: Aidan Rubenstein & Evan Hirsh
'''
import argparse
import cv2
import numpy as np

from halo import Halo

VIDEO_FILES = ['mov', 'mp4']

def sobel_edge_detection(image):
    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])   #Sobel filter in the x-direction
    sobel_edge_x = cv2.filter2D(image, -1, sobel_filter)    # convolve the image with the x-Sobel filter
    sobel_edge_y = cv2.filter2D(image, -1, np.flip(sobel_filter.T, axis=0)) # convolve the image with the y-Sobel filter

    sobel_full = np.hypot(sobel_edge_x, sobel_edge_y)   # use the Pythagorean Theorem to find the overall Gradient
    sobel_full = sobel_full / sobel_full.max() * 255

    theta = np.arctan2(sobel_edge_y, sobel_edge_x)  # calculates the angle theta between the x and y gradient
    return sobel_full.astype('uint8'), theta  # return magnitude and direction


def non_max_suppression(gradient_magnitude, gradient_direction):
    row, col = gradient_magnitude.shape # gets the rows/cols from the sobel operation
    output = np.zeros((row, col), dtype=np.int32)
    angle = gradient_direction * 180. / np.pi   # converts the angle from radians to degrees

    pi = 180
    for i in range(1, row - 1):
        for j in range(1, col - 1):

            # Determines what pixels to look at depending on what direction the angle is coming from in degrees
            if (0 <= angle[i, j] < pi/8) or (pi - (pi/8) <= angle[i, j] <= pi):
                m = gradient_magnitude[i, j + 1]    # the pixel before the current pixel in that direction
                n = gradient_magnitude[i, j - 1]    # the pixel after the current pixel in that direction
            elif (pi/8 <= angle[i, j] < (pi/4) + (pi/8)):
                m = gradient_magnitude[i + 1, j - 1]
                n = gradient_magnitude[i - 1, j + 1]
            elif ((pi/4) + (pi/8) <= angle[i, j] < (pi/2) + (pi/8)):
                m = gradient_magnitude[i + 1, j]
                n = gradient_magnitude[i - 1, j]
            elif ((pi/2) + (pi/8) <= angle[i, j] < pi - (pi/8)):
                m = gradient_magnitude[i - 1, j - 1]
                n = gradient_magnitude[i + 1, j + 1]

            if (gradient_magnitude[i, j] >= m) and (gradient_magnitude[i, j] >= n):
                output[i, j] = gradient_magnitude[i, j] # if current pixel is greater or equal than the pixels before and after it, then keep the same pixel intensity
            else:
                output[i, j] = 0    # otherwise, suppress it from the image
    return output.astype('uint8')


def threshold(img, strong, weak, nonrelevant=100):
    output = np.zeros(img.shape)    # create an output image with the same size as the input

    strong_row, strong_col = np.where(img >= strong)    # find the locations of pixels that are >= our strong parameter
    weak_row, weak_col = np.where((img <= strong) & (img >= weak))  # find the locations of pixels that are in-between strong and weak intensities

    output[strong_row, strong_col] = 255    # highlight the strong pixels for hysteresis
    output[weak_row, weak_col] = nonrelevant    # all of the weak pixels are now nonrelevant for hysteresis

    return output


def hysteresis(img, nonrelevant=100, strong=255):
    row, col = img.shape
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if img[i, j] == nonrelevant:
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                    or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong  # if any pixel is adjacent to a strong pixel, it also becomes strong
                else:
                    img[i, j] = 0   # otherwise, it is suppressed
    return img

def find_lanes(img, hough_lines):
    left = []   # store the coordinates of the left lane
    right = []  # store the coordinates of the right lane
    for line in hough_lines:
        x1, y1, x2, y2 = line.reshape(4)    # gives us a 1D array of values to go through
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # finds the slope and y-intercept that fits in all of the x and y coordinates
        slope = parameters[0]   # slope of the current line
        y_intercept = parameters[1] # y-intercept of the current line
        if slope < 0:
            left.append((slope, y_intercept))   # if the slope is positive, it belongs in the left lane
        else:
            right.append((slope, y_intercept))  # otherwise, a negative slope indicates the right lane

    left_avg = np.average(left, axis = 0)   # gives us a rough average slope & y-intercept for the left lane
    right_avg = np.average(right, axis = 0) # gives us a rough average slope & y-intercept for the right lane

    left_slope, left_intercept = left_avg
    l_y1 = img.shape[0] # origin (bottom) of the image
    l_y2 = int(l_y1 - 300)  # 300 pixels above the bottom of the image
    l_x1 = int((l_y1 - left_intercept) / left_slope)    # use y = mx + b equation to determine x1 location for the left lane
    l_x2 = int((l_y2 - left_intercept) / left_slope)    # use y = mx + b equation to determine x2 location for the left lane

    right_slope, right_intercept = right_avg
    r_y1 = img.shape[0]
    r_y2 = int(r_y1 - 300)
    r_x1 = int((r_y1 - right_intercept) / right_slope)  # use y = mx + b equation to determine x1 location for the right lane
    r_x2 = int((r_y2 - right_intercept) / right_slope)  # use y = mx + b equation to determine x1 location for the right lane


    coordinates = np.array([[l_x1, l_y1, l_x2, l_y2], [r_x1, r_y1, r_x2, r_y2]])    # returns a list of each line's 4 coordinates

    output = np.zeros_like(img)    # duplicate the image size to draw lines
    if coordinates is not None:
        for x1, y1, x2, y2 in coordinates:
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 5)   # draw a red line between the x1, y1 and x2, y2 coordinates to give us
    return output  # return a black image with lines drawn where the lanes will be

def process_img(original_img):

    gry_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    gauss = cv2.GaussianBlur(gry_img, (5, 5), cv2.BORDER_DEFAULT)


    sobel, theta = sobel_edge_detection(gauss)  # perform sobel edge detection

    nonmax = non_max_suppression(sobel, theta)  # use non-maximum suppression at the given theta of the overall gradient
    double_threshold = threshold(nonmax, 20, 5) # threshold the image

    canny_highway = hysteresis(double_threshold)    # use hysteresis to give the final canny edge detected image

    filtered_canny = apply_filter(canny_highway)    # apply a triangular mask to the canny edges

    hough = cv2.HoughLinesP(np.uint8(filtered_canny), 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)  # use Probabalistic Hough Transform to give us all of the lines in the image

    lanes = find_lanes(original_img, hough)    # determine the coordinates of the lanes

    output = cv2.addWeighted(original_img, 0.5, lanes, 0.5, 0)    # overlay the original color image over the final image with lane locations to write as the final product
    return output

def apply_filter(img):
    height, width = img.shape
    left_region = width * 0.15  # on average, the left lane starts at around 1/5th of the overall image width
    right_region = width * 0.85 # on average, the right lane starts at around 4/5th of the overall image width

    polygons = np.array([
        [(int(left_region), height), (int(right_region), height), (int(width/2), 245)]  # creates a triangular polygon based on the left point, right point, and center point
    ])

    mask = np.zeros(img.shape)

    cv2.fillPoly(mask, polygons, 255)   # sets all of the pixels in the triangle to white, otherwise black

    segment = cv2.bitwise_and(img, mask)    # using a bitwise-and function keeps all of the white pixels within the triangle, and ignores everything else, giving us our desired ROI
    return segment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect lanes in an image, video file, or webcam using canny edge detection and hough transform.')
    parser.add_argument('-i', '--input', type=str, nargs='?',   # adds all of the command-line arguments for the program
                        help='a file to process')

    args = parser.parse_args()
    if args.input:
        filename = args.input
        ext = filename.split('.')[1]
        # if the file being used is a video, process each frame of the video and recompile it into an mp4 result
        if ext in VIDEO_FILES:
            print("Processing {} as a video file".format(filename))
            cap = cv2.VideoCapture(filename)
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            out = cv2.VideoWriter('images/out.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
            frames = 1
            spinner = Halo(text='Processing frame 1', spinner='arc')
            spinner.start()
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    processed = process_img(frame)
                    out.write(processed)
                    frames += 1
                    spinner.text = 'Proccessing frame {}'.format(frames)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    spinner.stop()
                    break
            cap.release()   # release the video resource
            out.release()   # release the writer resource
        # if the file is an image, process only a single frame
        else:
            print("Processing {} as a image file".format(filename))
            original_img = cv2.imread(filename)
            processed = process_img(original_img)
            cv2.imwrite('images/processed.jpg', processed)