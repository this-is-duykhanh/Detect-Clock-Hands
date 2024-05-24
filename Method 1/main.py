import cv2
import numpy as np


def find_current_time(clock_image):
    # Define hands
    hands = [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]

    essepecial_hand = False

    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)

    # location: x , y , color, border
    x_y = [
        [0, 0, blue, 3],
        [0, 0, green, 2],
        [0, 0, red, 1],
    ]

    # Resize image to 235x223
    x = 235
    y = 223

    # Resize image
    resized_image = cv2.resize(clock_image, (x, y))

    # Convert image to gray
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Convert image to binary
    _, binary_image1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary_image1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    # Define center point
    center_point = None

    # Find center point
    for contour in contours:
        area = cv2.contourArea(contour=contour)

        x, y, _, _ = cv2.boundingRect(array=contour)

        M = cv2.moments(array=contour)

        if M["m00"] != 0 and area > 1500:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center_point = (cX, cY)
            cv2.circle(resized_image, (cX, cY), 5, (0, 255, 0), -1)
            break

    # Convert image to binary
    _, binary_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)

    # Define kernel for erosion
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    dillation = cv2.erode(src=binary_image, kernel=kernel, iterations=1)
    closing = cv2.morphologyEx(src=dillation, op=cv2.MORPH_CLOSE, kernel=kernel)

    # Define array for hands
    array = [[0, 0, 0, False]]

    angle = 0
    if center_point is not None:
        while angle < 360:
            angle1 = 0

            if angle < 270:
                angle1 = angle + 90
            else:
                angle1 = angle - 270

            # Convert angle to radians
            theta = np.radians(angle)

            # Define distance
            d = [
                10,
                20,
                25,
                28,
                30,
                35,
                36,
                37,
                38,
                40,
                45,
                46,
                47,
                48,
                50,
                55,
                60,
                62,
                65,
                68,
                70,
                72,
                74,
                76,
                78,
                80,
                82,
                85,
                90,
            ]

            count_special_point = 0

            # Check if all hands are found
            if hands[0][0] != -1 and hands[1][0] != -1 and hands[2][0] != -1:
                break

            for check in range(len(d)):
                theta_x = int(np.cos(theta) * d[check])
                theta_y = int(np.sin(theta) * d[check])

                if (0 <= center_point[0] + theta_x < closing.shape[1]) and (
                    0 <= center_point[1] + theta_y < closing.shape[0]
                ):
                    if (
                        closing[center_point[1] + theta_y, center_point[0] + theta_x]
                        <= 100
                    ):
                        if np.all(
                            a=resized_image[
                                center_point[0] + theta_x, center_point[1] + theta_y
                            ]
                            != [255, 255, 255]
                        ) or np.all(
                            a=resized_image[
                                center_point[0] + theta_x, center_point[1] + theta_y
                            ]
                            != [0, 0, 0]
                        ):
                            count_special_point += 1

                        if count_special_point >= 10:
                            array[0][3] = True

                        continue

                    elif check - 1 >= 0 and d[check - 1] > array[0][2]:
                        array[0] = [
                            center_point[0] + int(np.cos(theta) * d[check - 1]),
                            center_point[1] + int(np.sin(theta) * d[check - 1]),
                            d[check - 1],
                            array[0][3],
                        ]

                    elif d[check] < array[0][2]:
                        if array[0][2] < 37 or array[0][0] == 0 or array[0][1] == 0:
                            array[0] = [0, 0, 0, False]
                            count_special_point = 0
                            break

                        if array[0][3] and not essepecial_hand:
                            if hands[2][1] != 0:
                                if hands[1][1] != 0:
                                    hands[0][0] = int(hands[1][0] / 5)
                                    hands[0][2] = hands[1][2]
                                    x_y[0][0] = x_y[1][0]
                                    x_y[0][1] = x_y[1][1]

                                    hands[1][0] = hands[2][0]
                                    hands[1][1] = hands[2][1]
                                    hands[1][2] = hands[2][2]
                                    x_y[1][0] = x_y[2][0]
                                    x_y[1][1] = x_y[2][1]

                                else:
                                    hands[1][0] = hands[2][0]
                                    hands[1][1] = hands[2][1]
                                    hands[1][2] = hands[2][2]
                                    x_y[1][0] = x_y[2][0]
                                    x_y[1][1] = x_y[2][1]

                            essepecial_hand = True
                            hands[2][0] = int((angle1 - 3) * 1 / 6)
                            hands[2][1] = array[0][2]
                            hands[2][2] = angle1 - 3
                            x_y[2][0] = array[0][0]
                            x_y[2][1] = array[0][1]
                        else:
                            if hands[2][1] == 0:
                                hands[2][2] = angle1 - 3
                                hands[2][0] = int((angle1 - 3) * 1 / 6)
                                hands[2][1] = array[0][2]
                                x_y[2][0] = array[0][0]
                                x_y[2][1] = array[0][1]

                            elif hands[2][1] < array[0][2] and not essepecial_hand:
                                hands[2][2] = angle1 - 3
                                hands[1][2] = hands[2][2]

                                hands[1][0] = hands[2][0]
                                hands[1][1] = hands[2][1]
                                x_y[2][0] = x_y[1][0]
                                x_y[2][1] = x_y[1][1]

                                hands[2][0] = int((angle1 - 3) * 1 / 6)
                                hands[2][1] = array[0][2]
                                x_y[2][0] = array[0][0]
                                x_y[2][1] = array[0][1]

                            elif hands[1][1] == 0:
                                hands[1][2] = angle1 - 3
                                hands[1][0] = int((angle1 - 3) * 1 / 6)
                                hands[1][1] = array[0][2]
                                x_y[1][0] = array[0][0]
                                x_y[1][1] = array[0][1]

                            elif hands[1][1] < array[0][2]:
                                hands[0][2] = hands[1][2]
                                hands[0][0] = int(hands[1][0] / 5)
                                x_y[0][0] = x_y[1][0]
                                x_y[0][1] = x_y[1][1]

                                hands[1][2] = angle1 - 3
                                hands[1][0] = int((angle1 - 3) * 1 / 6)
                                hands[1][1] = array[0][2]
                                x_y[1][0] = array[0][0]
                                x_y[1][1] = array[0][1]

                            elif hands[1][1] > array[0][2]:
                                hands[0][0] = int((angle1 - 3) / 30)
                                hands[0][2] = angle1 - 3
                                x_y[0][0] = array[0][0]
                                x_y[0][1] = array[0][1]

                        array[0] = [0, 0, 0, False]
                        count_special_point = 0
                    break

            if array[0] != [0, 0, 0, False]:
                angle += 3
            else:
                angle += 6

    angle_minus = [
        0,
        3,
        30,
        33,
        60,
        63,
        69,
        90,
        93,
        120,
        123,
        150,
        153,
        180,
        183,
        210,
        213,
        240,
        243,
        270,
        273,
        300,
        303,
        330,
        333,
        360,
    ]
    angle_plus = [
        0,
        27,
        30,
        57,
        60,
        87,
        90,
        117,
        120,
        17,
        150,
        177,
        180,
        207,
        210,
        237,
        270,
        267,
        270,
        297,
        300,
        327,
        330,
        357,
        360,
    ]

    if hands[1][2] > 120 and hands[0][2] in angle_minus:
        hands[0][0] -= 1

    if (hands[1][2] < 40 and angle_minus != 0) and hands[0][2] in angle_plus:
        hands[0][0] += 1

    current_time = f"{hands[0][0]}:{hands[1][0]}:{hands[2][0]}"
    for location in range(len(x_y)):
        cv2.line(
            resized_image,
            center_point,
            (x_y[location][0], x_y[location][1]),
            x_y[location][2],
            x_y[location][3],
        )
        cv2.putText(
            resized_image,
            current_time,
            (5, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    return resized_image


# not ok
# clock_image = cv2.imread("./12.jpg")
# clock_image = cv2.imread("./5.jpg")
# clock_image = cv2.imread("./6.jpg")


filenames = [
    "input/1.jpg",
    "input/2.jpg",
    "input/3.jpg",
    "input/4.jpg",
    "input/7.jpg",
    "input/9.jpg",
    "input/10.jpg",
    "input/11.jpg",
    "input/13.jpg",
    "input/14.jpg",
    "input/15.jpg",
    "input/16.jpg",
]

for filename in filenames:
    clock_image = cv2.imread(filename=filename)
    if clock_image is not None:
        base, extension = filename.split(sep="/")[-1].rsplit(sep=".", maxsplit=1)
        output_filename = f"output/{base}_output.{extension}"
        cv2.imwrite(
            filename=output_filename, img=find_current_time(clock_image=clock_image)
        )
        print(output_filename)

    else:
        print(f"Unable to load image: {filename}")
