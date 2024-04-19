import cv2
import json
import csv

train_images_path = r'C:\Users\sneha\Documents\MTech\FinalYearProject\data\train_images'
test_images_path = r'C:\Users\sneha\Documents\MTech\FinalYearProject\data\test_images'
train_ann_path = r'C:\Users\sneha\Documents\MTech\FinalYearProject\data\train_ann'
test_ann_path = r'C:\Users\sneha\Documents\MTech\FinalYearProject\data\test_ann'

for i in range(103):
    image = cv2.imread(f'{test_images_path }\img{409+i}.jpg')
    cv2.imwrite(f'{test_images_path}\img_{i+1}.jpg', image)
    
for i in range(293, 409):
for i in [292]:
    f = open(f'{train_ann_path}\img{i}.json')
    ann = json.load(f)
    ann_count = len(ann['text_annotations'])
    ann_list = []
    for j in range(ann_count):
        vertices_dict = ann['text_annotations'][j]['boundingPoly']['vertices']
        vertices_list = [list(vertices_dict[i].values()) for i in range(len(vertices_dict))]
        final_ann = [str(x) for xs in vertices_list for x in xs]
        label = ann['text_annotations'][j]['description']
        final_ann.append(label)
        ann_list.append(final_ann)
    with open(f'{train_ann_path}\gt_img_{i}.txt', 'w', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        for list_element in ann_list:
            writer.writerow(list_element)
  
##
for i in range(103):
    f = open(f'{test_ann_path}\img{409+i}.json')
    ann = json.load(f)
    ann_count = len(ann['text_annotations'])
    ann_list = []
    for j in range(ann_count):
        vertices_dict = ann['text_annotations'][j]['boundingPoly']['vertices']
        vertices_list = [list(vertices_dict[i].values()) for i in range(len(vertices_dict))]
        final_ann = [str(x) for xs in vertices_list for x in xs]
        label = ann['text_annotations'][j]['description']
        final_ann.append(label)
        ann_list.append(final_ann)
    with open(f'{test_ann_path}\gt_img_{i+1}.txt', 'w', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        for list_element in ann_list:
            writer.writerow(list_element)
