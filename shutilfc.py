import glob

import shutil
import os
#여성
# y = 0
# path = "./face"
# x = 20
# for y in range(5):
#     i = f"age{x}"
#     x = x+10
#     a = glob.glob((os.path.join(path,i,"*","112","*.jpg")))
#     os.makedirs(f"./facedata/female/{i}", exist_ok=True)
#     #all_image = a #os.listdir(f"./face/{category}/")
#     # print("all_image>>", all_image)
#     for image in a:
#         # shutil의 인자값 = shutil.move(기존경로, 옮길 경로)
#         b=image.split("\\")[4]
#         shutil.move(f"{image}", f"./facedata/female/{i}/")


#남성
y = 0
path = "./face"
x = 20
for y in range(5):
    i = f"age{x}"
    x = x+10
    a = glob.glob((os.path.join(path,i,"*","111","*.jpg")))
    os.makedirs(f"./facedata/male/{i}", exist_ok=True)
    #all_image = a #os.listdir(f"./face/{category}/")
    # print("all_image>>", all_image)
    for image in a:
        # shutil의 인자값 = shutil.move(기존경로, 옮길 경로)
        b=image.split("\\")[4]
        shutil.move(f"{image}", f"./facedata/male/{i}/")