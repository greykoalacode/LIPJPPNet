import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

input_img_dir = "./datasets/examples/images"
output_pose_dir = "./output/pose/val"
output_parsing_dir = "./output/parsing/val"
inputimages = []
outputposeimages = []
outputparseimages = []
pose_data_array = []

for img in os.listdir(input_img_dir):
    textfilename = output_pose_dir+ "/"+ img.replace(".jpg",".txt")
    inputimagename = input_img_dir + "/" +img
    outputimagename = output_parsing_dir + "/" +img.replace(".jpg","_vis.png")
    inputimages.append(mpimg.imread(inputimagename))
    outputposeimages.append(mpimg.imread(inputimagename))
    outputparseimages.append(mpimg.imread(outputimagename))
    with open(textfilename, 'r') as f:
        firstline =  f.readlines()[0]
        linearray = list(map(int, firstline.strip().split(' ')))
        pose_data = np.array(linearray)
        pose_data = pose_data.reshape((-1,2))
        # print(pose_data)
        pose_data_array.append(pose_data)

n_images = len(inputimages)
columns = 3
n_results_to_display = 3
rows = 1
# rows=columns if columns>n_images else (n_results_to_display*n_images)/columns+1
index1=1
for i in range(0,n_images):
    fig = plt.figure(figsize=(10,200))
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    plt.subplot(1, columns, index1)
    plt.axis('off')
    plt.title('Person ')
    plt.imshow(inputimages[i])

    plt.subplot(1, columns, index1+1 )
    plt.axis('off')
    plt.title('Pose estimation')
    j=0
    plt.imshow(outputposeimages[i])
    for x,y in pose_data_array[i]: 
        plt.plot(x, y, 'co') # 'w.': color='white', marker='.'
        plt.text(x, y, str(j), color='r', fontsize=8)
        j+=1
            
    plt.subplot(1, columns, index1+2 )
    plt.axis('off')
    plt.title('Parsed Image')
    plt.imshow(outputparseimages[i])
    plt.show()
    # index1+=n_results_to_display
    # if i>rows*columns: break
# plt.show()
# print(rows, i, index1,index1+1, index1+2, index1+3,index1+4)
        # img = plt.imread(inputimagename)
        # plt.imshow(img)
        # i=0
        # for x,y in pose_data: 
        #     plt.plot(x, y, 'co') # 'w.': color='white', marker='.'
        #     plt.text(x, y, str(i), color='r', fontsize=8)
        #     i+=1
        # plt.show()
