import pandas as pd
from mtcnn.mtcnn import MTCNN
import cv2

# Resource used: https://github.com/ipazc/mtcnn

# Read file that lists full paths of images for training
df = pd.read_csv("data/training_images.csv")

print("Detecting faces in training images...")
for index, row in df.iterrows():
    # get image
    filename = row["name"]
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # get face from image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # mtcnn uses rgb
    face_scanner = MTCNN()
    faces = face_scanner.detect_faces(color_image)
    face = faces[0]

    # extract the bounding box from the face
    x1, y1, width, height = face['box']
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face_boundary = image[y1:y2, x1:x2]

    # new image with just the face
    new_filename = filename.replace("cohn-kanade-images", "cohn-kanade-images-just-faces")
    print("New training image: " + new_filename)
    cv2.imwrite(new_filename, face_boundary)

print("Face detection on training images complete...")
print("")

# Read file that lists full paths of images for validation
df = pd.read_csv("data/validation_images.csv")

print("Detecting faces in validation images...")
for index, row in df.iterrows():
    # get image
    filename = row["name"]
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # get face from image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    face_scanner = MTCNN()
    faces = face_scanner.detect_faces(color_image)
    face = faces[0]

    # extract the bounding box from the face
    x1, y1, width, height = face['box']
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face_boundary = image[y1:y2, x1:x2]

    # new image with just the face
    new_filename = filename.replace("cohn-kanade-images", "cohn-kanade-images-just-faces")
    print("New validation image: " + new_filename)
    cv2.imwrite(new_filename, face_boundary)

print("Face detection on validation images complete...")
print("Writing updated index file for model building...")

# update filenames for preprocessed images
# Update Training images
df = pd.read_csv("images_training_list.csv")
df["name"] = df["name"].str.replace("cohn-kanade-images", "cohn-kanade-images-just-faces")
df.to_csv("data/training_images_preprocessed.csv", index=False)

# Update Validation images
df = pd.read_csv("images_validation_list.csv")
df["name"] = df["name"].str.replace("cohn-kanade-images", "cohn-kanade-images-just-faces")
df.to_csv("data/validation_images_preprocessed.csv", index=False)
print("COMPLETE")
