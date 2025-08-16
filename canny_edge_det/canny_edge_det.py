import cv2
import datetime
import os

# Load image
image = cv2.imread(r'path/image')  # Replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny Edge Detection
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Generate filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"canny_output_{timestamp}.png"

# Save in same directory as app.py
save_path = os.path.join(os.getcwd(), filename)
cv2.imwrite(save_path, edges)

# Optional: Show result
cv2.imshow('Original', image)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
