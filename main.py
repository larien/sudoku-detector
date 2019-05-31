import cv2
import numpy as np
import os

repository = f"{os.getcwd()}/images"
font = cv2.FONT_HERSHEY_TRIPLEX


def get_files(path):
    for _, _, files in os.walk(path):
        return files


def append_images(files):
    vector = []
    for file in files:
        img = cv2.imread(os.path.join(repository, file))
        vector.append(img)
    return vector


def load():
    files = get_files(repository)
    return append_images(files)


def gradient(images):
    for index, image in enumerate(images):
        analyze(str(index), image)


def erode(image):
    copy = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    kernel = np.ones((8, 8), np.uint8)
    return cv2.erode(copy, kernel, iterations=1)


def apply_filter(image):
    size = (5, 5)
    filtro = cv2.GaussianBlur(image, size, 0)
    threshold_min = 100
    threshold_max = 200
    return cv2.Canny(filtro, threshold_min, threshold_max)


def get_contours(image):
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours


def get_approx(contour):
    return cv2.approxPolyDP(
            contour,
            0.01 * cv2.arcLength(contour, True),
            True,
            )


def verify_data_size(image, contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    data = image[y: y + h, x: x + w]
    height, width = data.shape[0], data.shape[1]
    return is_offset_valid(height, width)


def is_offset_valid(height, width):
    offset_min = 20
    offset_max = 150
    minimum, maximum = False, False

    if height > offset_min and width > offset_min:
        minimum = True
    if height < offset_max and width < offset_max:
        maximum = True
    return minimum and maximum


def define_box(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def is_valid_approx(approx):
    if len(approx) >= 4 and len(approx) <= 10:
        return True
    return False


def new_filter(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold1 = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)
    # Binary adaptive threshold using 11 nearest neighbour pixels
    return cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def analyze(name, original):
    print(f"Analyzing {name}")
    # cv2.imshow(f"Original image {name}", original)

    eroded_image = erode(original)
    # cv2.imshow(f"Eroded image {name}", original)

    filtered_image = new_filter(original)
    # cv2.imshow(f"Filtered image {name}", filtered_image)

    contours = get_contours(filtered_image)
    print(contours)
    for contour in contours:
            approx = get_approx(contour)
            if is_valid_approx(approx) and verify_data_size(eroded_image, contour):
                box = define_box(contour)
                (x, y, w, h) = cv2.boundingRect(contour)
                # print(f"x: {x}, y: {y}, w: {w}, h: {h}")
                # conter = cv2.countNonZero(eroded_image[x: y, x+w: x + h])
                conter = np.sum(eroded_image[y: y + h, x: x + w] == 255)
                x1 = x+w
                x2 = x+h
                print(f"corte: {x}:{y},{x1}:{x2}")
                print(f"non-zeros: {conter}")
                if conter <= 190:
                    cv2.drawContours(original, [box], 0, (0, 0, 255), 1)    
    print(f"Treatment for {name} complete")
    cv2.imshow(f"Treated image {name}", original)


if __name__ == "__main__":
    images = load()

    gradient(images)

    cv2.waitKey()
    cv2.destroyAllWindows()
