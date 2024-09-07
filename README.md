# Retrieving The Time From The Photo Of A Clock (Digital Image Processing )

Automatic timekeeping based on image recognition has promise for several applications.
This study explores a computer vision technique to extract the current time shown on a clock face from an image. The method uses image processing techniques and OpenCV, a well-known real-time computer vision library, to do this.
The study goes into detail on the functionality of the code, covering the stages of image pre-processing, hand detection logic, and time estimation methodologies. It also
highlights the main challenges and elements to consider when implementing such a system. This research evaluates the efficiency and limitations of the code to shed light on
the feasibility and potential improvements for automatic time detection from clock images.

## Research Methodology 1

The aim of the task is to precisely ascertain the present time from a picture that
solely shows a clock. The image needs to be limited to showing the clock and nothing
else, and it needs to have a light background (dark backgrounds are not supported). All
the clock's hands should be black (RGB: 000), except for the second hand, which might
be colored differently.

### 1. Locate the clock's center:

Using contour detection techniques, locate the clock's center—where the three hands intersect—and outline the clock's face.

### 2. Preparing images:

Utilizing erosion, closing, and opening techniques, improve the input image to make the clock hands easier to see and detect. To create a binary image, turn a significant portion of the clock black and leave the background white.

### 3. Scanning for clock hands:

To scan the entire clock face, implement a loop. Determine a variable length to see if the clock hands are present at various angles. To accurately measure the length if a hand is detected, extend the length until it exceeds the hand's length. Find a line that isn't black to determine which is the second hand.

### 4. Assigning hands:

Assign the second hand's length and angle after the first hand has been located. If
you discover a second straight line whose length is longer than the second hand's, give
the minute hand the length and angle of the second hand and the second hand's length
and length. The minute hand, on the other hand, has the second straight line's length and
angle. The second hand will have the same length and angle as the third straight line and
the hour hand will have the same length and angle as the old second hand if you discover
a third straight line whose length is longer than the long second hand. We keep checking
to see if the two hands will automatically adjust in length and angle if the current hand
is longer than the minute hand. If not, nothing will change; just make sure that the length
of the hour and minute hands is equal. If so, switch the angles and lengths.

### 5. Calculating the current time:

Using trigonometric formulae, determine the current time based on the hands'
lengths and angles. Take into consideration the little angular variation between the
minute and second hands, and account for the variance between the hour and minute
hands. If there is a discrepancy between the hour mark and the minute hand's position by
three to six degrees, make the necessary corrections to the hour hand's position.

### 6. Visualization:

Draw the identified hands with distinct borders and colors to make them easier to
see on the input image: The borders of the hour, minute, and second hands are three
pixels in blue, two pixels in green, and one pixel in red, respectively.

### 7. Output:

In the designated output folder, save the output image.

## Research Methodology 2 (Better than methodology 1)

### 1. Image Preprocessing

Open the specified folder and load the input image. To simplify the computation, resize the picture.To make managing brightness and color easier, convert the image from RGB to HSV color format.For more contrast, flip the HSV colors.on create a binary image from the HSV image, apply Otsu thresholding on the V channel.Apply Gaussian blur to lower noise levels.

### 2. Center Identification of the Clock

Find the point where the three hands of the clock cross, which is the center. To ascertain the clock face's outline, using contour detection methods and the Hough Circle Transform. Determine the clock's radius and center of gravity.

### 3. Edge Detection

To find edges in the picture, use the Canny edge detector. Examine these edges to identify straight lines.

### 4. Clock Hand Line Identification

Using the thickness and proximity of the lines, identify pairs of lines that might be clock hands. Make sure the angles of these lines are similar and they lie inside the
clock radius.

### 5. Clock Hand Detection

Find the ends of the line segments that have been recognized. Draw lines from the farthest point to the clock center to create the clock hands.

### 6. Clock Hand Classification

Sort the three hands that have been found. Choose the hand with the least thickness to identify the second hand, and then take it out of the equation. Determine which of the remaining hands is the longer minute hand and which is the shorter hour hand.

### 7. Visualization of Clock Hands

Sketch every recognized hand on the input picture. Give each hand a unique color: the minute hand should be green, the hour hand should be blue, and the second hand should be red.

### 8. Rotation Angle Determination

To find the angles between the clock hands and the central point, use trigonometric formulae.

### 9. Time Calculation

Based on their lengths and angles, ascertain the hands' placements. Utilizing trigonometric formulas, determine the current time while accounting for the angular difference between the minute and hour hands. It is deemed insignificant that there is a difference between the minute and second hands. Verify the position of the minute hand to adjust for any potential errors in the detected time if the hour hand's angle deviates by 3 to 9 degrees from the predicted hour mark.

### 10. Annotate Time on Image

Using blue for the hour, green for the minute, and red for the second, display the computed time in the image's lower right corner.

### 11. Save Output Image

The annotated image should be saved in the input folder. Include the words "Cannot find enough hands" in the output image if the image processing is unable to detect enough hands. If not, make sure the output image has hand lines and extra time added while maintaining the same format as the input image

