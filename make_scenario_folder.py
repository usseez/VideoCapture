import os

FOLDER_DIR = "/home/ubuntu/Dataset/00.datasetStructure/01.data_acquisitionStructure"
weather = ["DayFrontLight", "DayBackLight", "DaySideLight", "NightLight", "NightLowLight", "Cloudy", "Rain", "Snow", "Inside"]
camera_direction = ["Rear", "Side"]
tint = ["5", "15", "100"]
class_name = ["agood", "block", "bubble", "mud", "raindrop"]
object = ["None", "Vehicle", "Two-wheeler", "Person"]

folder_names = []


for w in weather:
    for t in tint:
        for d in camera_direction:
            for c in class_name:
                for o in object:
                    if d == "Rear":
                        if t == "100":
                            folder_name = f"{w}_{d}_{o}_{c}"
                        else:
                            folder_name = f"{w}_{t}_{d}_{o}_{c}"
                    else:
                        folder_name = f"{w}_{d}_{o}_{c}"
                    folder = FOLDER_DIR + "/" + folder_name
                    if not os.path.exists(folder):
                        os.mkdir(folder)
