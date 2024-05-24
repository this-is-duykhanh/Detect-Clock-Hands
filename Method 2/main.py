import cv2
import numpy as np
import math


def resize_input(img):
    """
    The purpose of the resize_input function is to alter the size while preserving the aspect ratio; its longest side is 1000 pixels.
    """
    height, width, _ = img.shape
    scale = 1000 / max(height, width)
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return img


def clock_detection(img, blurred):
    """
    The purpose of the clock_detection function is to extract the clock from the image.
    """
    radius = 0
    center_x, center_y = 0, 0

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        400,
        param1=50,
        param2=100,
        minRadius=300,
        maxRadius=500,
    )

    max_circle = None

    if circles is not None:
        for circle in circles[0, :]:
            x, y, r = circle

            if r < radius:
                continue

            max_circle = circle

        x, y, r = max_circle

        center_x = int(x)
        center_y = int(y)
        radius = int(r)

    else:
        contours, _ = cv2.findContours(
            blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        max_area = 0
        max_rect = None

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < max_area:
                continue

            max_area = area
            max_rect = contour

        if max_rect is None:
            return None

        x, y, w, h = cv2.boundingRect(max_rect)

        center_x = x + w // 2
        center_y = y + h // 2

        radius = min(w, h) // 2

    cv2.circle(img, (center_x, center_y), 20, (255, 0, 255), -1)
    return center_x, center_y, radius


def line_detection(blurred):
    """
    The purpose of the line_detection function is to identify straight lines in an image.
    """
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=90, minLineLength=30, maxLineGap=5
    )
    return lines


def distance_to_center(x1, center_x, y1, center_y):
    """
    The distance_to_center function has the function of calculating the distance between a point and the center of the clock
    """
    return np.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2)


def clock_hand_line(lines, center_x, center_y, radius):
    """
    Finding lines that are roughly parallel and close together is the purpose of the clock_hand_line  function, which groups lines into groups.
    """
    groups = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        length1 = distance_to_center(x1, center_x, y1, center_y)
        length2 = distance_to_center(x2, center_x, y2, center_y)
        max_length = np.max([length1, length2])
        min_length = np.min([length1, length2])

        if (max_length < radius) and (min_length < radius * 1 / 2):
            angle = math.atan2(y2 - y1, x2 - x1)
            angle = math.degrees(angle)

            grouped = False

            for group in groups:
                mean_angle = group["angle"]

                if (
                    abs(angle - mean_angle) < 12
                    or abs(angle - mean_angle - 180) < 12
                    or abs(angle - mean_angle + 180) < 12
                ):
                    group["lines"].append(line)

                    grouped = True
                    break

            if not grouped:
                groups.append({"lines": [line], "angle": angle})
    return groups


def hand_thickness(line1, line2):
    """
    The function to determine the distance between two parallel lines is part of the function hand thickness.
    """
    x1_1, y1_1, x2_1, y2_1 = line1[0]
    x1_2, y1_2, x2_2, y2_2 = line2[0]

    vector1 = np.array([x2_1 - x1_1, y2_1 - y1_1])

    vector_between_lines = np.array([x1_2 - x1_1, y1_2 - y1_1])

    distance = np.abs(np.cross(vector1, vector_between_lines)) / np.linalg.norm(vector1)

    return distance


def hands_detection(groups, center_x, center_y):
    """
    In order to produce a clock hand with the clock center point, the hands detection function finds the line segment's farthest endpoint from the clock center among line segments in the same group.
    """
    hands = []

    for group in groups:
        lines = group["lines"]
        num_lines = len(lines)

        max_thickness = 0
        max_length = 0

        for i in range(num_lines):
            x1, y1, x2, y2 = lines[i][0]

            length1 = distance_to_center(x1, center_x, y1, center_y)
            length2 = distance_to_center(x2, center_x, y2, center_y)

            length = np.max([length1, length2])

            if length > max_length:
                max_length = length

                if length == length1:
                    max_line = x1, y1, center_x, center_y
                else:
                    max_line = x2, y2, center_x, center_y

            for j in range(i + 1, num_lines):
                thickness = hand_thickness(lines[i], lines[j])

                if thickness > max_thickness:
                    max_thickness = thickness

        line = max_line, max_thickness, max_length

        if max_thickness > 0:
            hands.append(line)

    hands.sort(key=lambda x: x[2], reverse=True)

    hands = hands[:3]
    return hands


def clock_hand_classification(hands):
    """
    The three clock hands detected via the hands_detection function can be used to precisely determine the hour, minute, and second hands using the clock_hand_classification function.
    """
    sorted_hands_by_thickness = sorted(hands, key=lambda hands: hands[1])

    second_hand = sorted_hands_by_thickness[0]

    hands.remove(second_hand)

    sorted_hands_by_length = sorted(hands, key=lambda hands: hands[2])

    hour_hand = sorted_hands_by_length[0]
    minute_hand = sorted_hands_by_length[1]

    return hour_hand, minute_hand, second_hand


def draw_hands(img, hour_hand, minute_hand, second_hand, center_x, center_y):
    """
    The draw_hands function has the function of drawing the clock hands on the image.
    """
    x1, y1, _, _ = hour_hand[0]
    cv2.line(img, (center_x, center_y), (x1, y1), (255, 0, 0), 15)

    x1, y1, _, _ = minute_hand[0]
    cv2.line(img, (center_x, center_y), (x1, y1), (0, 255, 0), 10)

    x1, y1, _, _ = second_hand[0]
    cv2.line(img, (center_x, center_y), (x1, y1), (0, 0, 255), 5)


def get_vector(hand):
    """
    Function to determine a clock hand's direction vector
    """
    x1, y1, x2, y2 = hand[0]
    vector = [x2 - x1, y2 - y1]
    return vector


def dot_product(u, v):
    """
    Function for figuring out two vectors' dot product
    """
    return u[0] * v[0] + u[1] * v[1]


def cross_product(u, v):
    """
    The directed product of two vectors is computed by the function.
    """
    return u[0] * v[1] - u[1] * v[0]


def get_angle(hand, center_x, center_y):
    """
    Function to determine a clock hand's angle with respect to the y direction
    """
    u = get_vector(hand=hand)

    v = [center_x - center_x, center_y - (center_y - 100)]

    dot_uv = dot_product(u=u, v=v)

    length_u = math.sqrt(u[0] ** 2 + u[1] ** 2)
    length_v = math.sqrt(v[0] ** 2 + v[1] ** 2)

    cos_theta = dot_uv / (length_u * length_v)

    cos_theta = max(min(cos_theta, 1.0), -1.0)

    theta = math.acos(cos_theta)

    theta_degrees = math.degrees(theta)

    cross_uv = cross_product(u, v)
    if cross_uv > 0:
        return 360 - theta_degrees
    else:
        return theta_degrees


def get_time(hour_angle, minute_angle, second_angle):
    """
    The get_time function computes the time by utilizing the angles of the clock hands.
    """
    hour = hour_angle / 30

    minute = minute_angle / 6
    second = second_angle / 6

    if (round(hour) * 30 - hour_angle <= 6) and (
        (355 < minute_angle and minute_angle < 360) or (minute_angle < 90)
    ):
        hour = round(hour)
        if hour == 12:
            hour = 0

    if (hour_angle - hour * 30 <= 6) and (355 < minute_angle and minute_angle < 360):
        minute = 0

    if (round(minute) * 6 - minute_angle <= 6) and (second_angle < 6):
        minute = round(minute)
        if minute == 60:
            minute = 0

    if (minute_angle - minute * 30 <= 6) and (
        354 < second_angle and second_angle < 360
    ):
        second = 0

    hour = int(hour)
    minute = int(minute)
    second = int(second)

    time = f"{hour:02d}:{minute:02d}:{second:02d}"
    return time


def draw_time(img, time):
    """
    Drawing time on a clock image is the purpose of the draw_time function.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3

    hour, minute, second = map(str, time.split(":"))

    cv2.putText(img, hour, (10, 950), font, font_scale, (255, 0, 0), font_thickness)
    cv2.putText(img, ":", (90, 950), font, font_scale, (0, 0, 0), font_thickness)

    cv2.putText(img, minute, (110, 950), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(img, ":", (190, 950), font, font_scale, (0, 0, 0), font_thickness)

    cv2.putText(img, second, (210, 950), font, font_scale, (0, 0, 255), font_thickness)


def resolve(img):
    img = resize_input(img=img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.bitwise_not(img_hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    _, thresh = cv2.threshold(
        img_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # If the clock is not detected, return None
    if clock_detection(img, blurred) is None:
        return None

    center_x, center_y, radius = clock_detection(img, blurred)

    lines = line_detection(blurred)

    groups = clock_hand_line(lines, center_x, center_y, radius)

    hands = hands_detection(groups, center_x, center_y)

    # If there are not enough hands, return None
    if len(hands) < 3:
        return

    hour_hand, minute_hand, second_hand = clock_hand_classification(hands)

    draw_hands(img, hour_hand, minute_hand, second_hand, center_x, center_y)

    hour_angle = get_angle(hand=hour_hand, center_x=center_x, center_y=center_y)
    minute_angle = get_angle(hand=minute_hand, center_x=center_x, center_y=center_y)
    second_angle = get_angle(hand=second_hand, center_x=center_x, center_y=center_y)

    time = get_time(
        hour_angle=hour_angle, minute_angle=minute_angle, second_angle=second_angle
    )

    draw_time(img=img, time=time)

    return img


def main():
    for i in range(1, 31):
        filename = f"input/{i}.jpg"

        img = cv2.imread(filename=filename)

        if img is None:
            print(f"Warning: Cannot open/read file: {filename}")
            continue

        img_reresolve = resolve(img)

        if img_reresolve is not None:
            img = img_reresolve

        img = cv2.resize(src=img, dsize=(400, 400))

        if img_reresolve is None:
            img = cv2.putText(
                img=img,
                text="Cannot find enough hands.",
                org=(0, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
                thickness=2,
            )

        result_path = f"output/output_{i}.jpg"

        cv2.imwrite(filename=result_path, img=img)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
