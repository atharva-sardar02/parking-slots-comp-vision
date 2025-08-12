import cv2
import re
from datetime import datetime

def extract_date_time_from_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance text visibility
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and extract text
    extracted_text = ""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y:y + h, x:x + w]

        # Assuming the text is black on a white background, you can further customize this
        mean_color = cv2.mean(roi, mask=thresh[y:y + h, x:x + w])

        # Adjust the threshold based on your image content and conditions
        if mean_color[0] < 150:
            # Extract text using OCR or any other method suitable for your images
            text = extract_text_from_image(roi)
            extracted_text += text

    # Try to extract date and time using a simple pattern (modify as needed)
    date_time_format = "%Y-%m-%d %H:%M:%S"
    date_time_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', extracted_text)

    if date_time_match:
        date_time_str = date_time_match.group()
        date_time = datetime.strptime(date_time_str, date_time_format)
        return date_time
    else:
        print("Unable to extract date and time from the image.")
        return None

def extract_text_from_image(roi):
    # Add your custom text extraction method here
    # You might use additional image processing or another library depending on your needs
    # Example: Use OCR library, custom pattern matching, etc.
    # For simplicity, this function returns an empty string
    return ""

# Replace 'your_image_path.jpg' with the path to your image file
image_path = 'img_1002023603.jpg'
result = extract_date_time_from_image(image_path)

if result:
    print(f"Date and Time extracted from the image: {result}")
