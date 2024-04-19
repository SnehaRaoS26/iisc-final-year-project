import os
import base64 
import requests
import json
from pathlib import Path
import cv2
import numpy as np
import skimage.morphology
from scipy import stats as st
import json
from skimage.feature import peak_local_max
from skimage import img_as_float

########################################################### Get Stroke Width of Image #############################################################

######################### Version 1
def adaptive_thresholding(image):
    output_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
    return output_image


def stroke_width(image):
    dist = cv2.distanceTransform(cv2.subtract(255,image), cv2.DIST_L2, 5)
    im = img_as_float(dist)
    coordinates = peak_local_max(im, min_distance=15)
    pixel_strength = []
    for element in coordinates:
        x = element[0]
        y = element[1]
        pixel_strength.append(np.asarray(dist)[x,y])
    mean_pixel_strength = np.asarray(pixel_strength).mean()
    return mean_pixel_strength

image = cv2.imread('/Users/Z0045NQ/Desktop/img176_05.png', 0)
process_image = adaptive_thresholding(image)
stroke_width(process_image)

stroke_width(image)

######################### Version 2
import cv2

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def text_coeff(image):
  _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  coeffs = []
  for contour in contours:
    print(contour)
    x, y, w, h = cv2.boundingRect(contour)
    bounding_rectangle_area = w * h
    contour_area = cv2.contourArea(contour)
    coeff = contour_area / bounding_rectangle_area
    print(coeff)

text_coeff(cv2.imread('/Users/Z0045NQ/Desktop/img176_03.png', 0))

######################### Version 3 ---- WORKED
import cv2
import numpy as np
import skimage.morphology
from scipy import stats as st

def get_thickness(img_path):
  img = cv2.imread(img_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  distance = thresh.copy()
  distance = cv2.distanceTransform(distance, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
  binary = thresh.copy()
  binary = binary.astype(np.float32)/255
  skeleton = skimage.morphology.skeletonize(binary).astype(np.float32)
  thickness = cv2.multiply(distance, skeleton)
  # average = np.mean(thickness[skeleton!=0])
  char_mode = st.mode(thickness[skeleton!=0])
  print(char_mode)
  # thick = 2 * average
  # thick = 2 * char_mode
  # print("thickness:", thick)

get_thickness('/Users/Z0045NQ/Downloads/279098_MTech_Final_Year_Project/google_images/cropped_images/img176/img176_04.jpg')

# read input
# convert to grayscale
# use thresholding
# get distance transform
# get skeleton (medial axis)
# apply skeleton to select center line of distance 
# get average thickness for non-zero pixels
# thickness = 2*average
#########################################################################################################################################################

##################################################################### Combined Code #####################################################################

import cv2
import numpy as np
import skimage.morphology
from scipy import stats as st
import json

def crop_image_with_polygon(input_image, polygon_vertices):
    # Create a black mask with the same shape as the input image
    mask = np.zeros_like(image, dtype=np.uint8)
    # Draw the polygon on the mask
    cv2.fillPoly(mask, [np.array(polygon_vertices)], (255, 255, 255))
    # Bitwise AND operation to obtain the cropped region
    cropped_image = cv2.bitwise_and(input_image, mask)
    return cropped_image

def get_thickness(img):
  # img = cv2.imread(img_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  distance = thresh.copy()
  distance = cv2.distanceTransform(distance, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
  binary = thresh.copy()
  binary = binary.astype(np.float32)/255
  skeleton = skimage.morphology.skeletonize(binary).astype(np.float32)
  thickness = cv2.multiply(distance, skeleton)
  # average = np.mean(thickness[skeleton!=0])
  char_mode = st.mode(thickness[skeleton!=0])
  return char_mode

def get_stroke_width(image_path, api_ann_file):
  image = cv2.imread(image_path)
  api_ann = json.load(open(api_ann_file))
  stroke_width = {}
  bboxes = api_ann['responses'][0]['textAnnotations'][1:]
  num_bboxes = len(bboxes)
  for i in range(num_bboxes):
    bbox_dict = bboxes[i]['boundingPoly']['vertices']
    bbox_vertices = [tuple(bbox_dict[i].values()) for i in range(len(bbox_dict))]
    cropped_image = crop_image_with_polygon(image, bbox_vertices)
    bbox_stroke_width = get_thickness(cropped_image)[0]
    stroke_width[bboxes[i]['description']] = bbox_stroke_width
  return stroke_width

image_path = '/Users/Z0045NQ/Downloads/279098_MTech_Final_Year_Project/google_images/img/img13.jpg'
api_ann_file = '/Users/Z0045NQ/Downloads/279098_MTech_Final_Year_Project/google_images/google_ocr_result/ann_result/img13.json'
get_stroke_width(image_path, api_ann_file)

stroke_width.append({bboxes[i]['description']: bbox_stroke_width})

def distance_metric(box1, box2):
  point_a = box1['boundingPoly']['vertices'][1]
  point_b = box2['boundingPoly']['vertices'][0]
  point_c = box1['boundingPoly']['vertices'][2]
  point_d = box1['boundingPoly']['vertices'][3]
  avg_dist = (math.dist(point_a, point_b) + math.dist(point_c, point_d))/2
  return avg_dist

new_text_ann = {}
total_boxes = len(text_ann)
min_dist = 999999
boxes_merge = []
for i in range(1, 5):
  box_a = text_ann[i]
  box_b = text_ann[i+1]
  distance_a_b = distance_metric(box_a, box_b)
  if distance_a_b < min_dist:
    min_dist = distance_a_b
    boxes_merge.append(i)
    boxes_merge.append(i+1)
    
#########################################################################################################################################################

################################################################ Convert Image to base64 encoding ################################################################
import base64 

def convert_base64(image):
  with open(image, "rb") as image2string: 
    converted_string = base64.b64encode(image2string.read())
  return converted_string.decode()
print(converted_string) 


with open('encode.txt', "wb") as file: 
    file.write(converted_string)
  
#########################################################################################################################################################

################################################################ Make Post Request ################################################################

def get_google_apt_result(image):
  payload = {
    "requests":[
      {
        "image":{
          "content": f"{convert_base64(image)}"
        },
        "features":[
          {
            "type":"TEXT_DETECTION",
            "maxResults":1
          }
        ]
      }
    ]
  }
  params = {
    'key': 'AIzaSyBfKreJZAm6Zt32cNs1DoirLKe4UkK3BjI',
  }
  url_post = "https://vision.googleapis.com/v1/images:annotate"
  post_response = requests.post(url_post, json=payload, params=params)
  post_response_json = post_response.json()
  if post_response_json['responses'][0]:
    annotations = {"text_annotations": post_response_json['responses'][0]['textAnnotations'][1:]}
  else:
    annotations = None
  return annotations

base_image_path = '/Users/Z0045NQ/Downloads/279098_MTech_Final_Year_Project/google_images/img'
ann_base_path = '/Users/Z0045NQ/Downloads/279098_MTech_Final_Year_Project/google_images/google_ocr_result/ann_result'
image_files = os.listdir(base_image_path)
count = 0
for image in image_files:
  print(f'Processing image: {image}')
  image_name = image.split('.')[0]
  annotations = get_google_apt_result(f'{base_image_path}/{image}')
  with open(f'{ann_base_path}/{image_name}.json', 'w') as f:
    json.dump(annotations, f)

for image in ['img_292.jpg']:
  image_path = f'{base_image_path}/{image}'
  image_name = image.split('.')[0]
  ann_path = f'{ann_base_path}/{image_name}.json'
  if Path(ann_path).exists():
    continue
  else:
    print(f'Processing image: {image}')
    annotations = get_google_apt_result(image_path)
    if annotations is None:
      print(f'Could not detect text from image {image_name} by Vision API. Skipping the image...')
      continue
    else:
      with open(f'{ann_base_path}/{image_name}.json', 'w') as f:
        json.dump(annotations, f)

  count = count+1
  if count == 5:
    break

#########################################################################################################################################################

