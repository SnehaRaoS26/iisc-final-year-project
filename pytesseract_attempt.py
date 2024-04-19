!sudo apt install tesseract-ocr
!pip install pytesseract

import pytesseract
import cv2
from google.colab import drive
drive.mount('/content/drive')

image = cv2.imread('/content/drive/MyDrive/FinalProject/data/pyimagesearch_address.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

text = pytesseract.image_to_string(image)
print(text)
