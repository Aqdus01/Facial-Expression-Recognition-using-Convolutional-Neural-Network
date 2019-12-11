#  --------------------------------------- Imports -------------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from eval_plotter import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

#  --------------------------------------- Inputs -------------------------------------------------------------------
# Assess model accuracy on preprocessed validation images not used in training
image_size = (54, 72)  # facial extract
model_name = '/home/ubuntu/Project/model_wang.hdf5'
validation_images = 'data/validation_images_preprocessed.csv'

results_df = pd.read_csv(validation_images)

#  --------------------------------------- Prepare Image Generator ----------------------------------------------------

test_datagen = ImageDataGenerator(rescale=1. / 255.)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=results_df,
    directory=None,
    x_col="name",
    y_col="class",
    target_size=image_size,
    batch_size=32,
    seed=666,
    class_mode='categorical',
    shuffle=False,
    verbose=1)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

#  --------------------------------------- Load Model and Assess ----------------------------------------------------
model = tf.keras.models.load_model(model_name)
scoring = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST, verbose=0)
print("Model results: " + str(model.metrics_names[1]) + " " + str(scoring[1]))

# View Actuals vs Predicted
predictions = model.predict_generator(generator=test_generator)
predictions_class = np.argmax(predictions, axis=1)

# Setup results in a pandas data-frame
results_df["Prediction"] = predictions_class
image_map = {v: k for k, v in test_generator.class_indices.items()}
results_df["Prediction_Class"] = results_df["Prediction"].replace(image_map)

results_df["Actual"] = results_df["class"].replace(test_generator.class_indices)
results_df["Correct"] = results_df["class"] == results_df["Prediction_Class"]

#  --------------------------------------- Visualize Results ---------------------------------------------------------

# Assess the predictions using a Confusion Matrix
actuals_classes, predictions_classes = results_df["Actual"].values, results_df["Prediction"].values
classes_list = np.array(sorted(list(test_generator.class_indices.keys())))

# Refer to eval_plotter.py script
plot_confusion_matrix(actuals_classes, predictions_classes, classes=classes_list,
                      normalize=True,
                      title='Confusion Matrix')
plt.show()

print("Overall Assessment: ")
print(classification_report(results_df["class"], results_df["Prediction_Class"], np.unique(results_df["class"].values)))
