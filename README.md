This is a description of the code used for this project.  The order represents the step by step order needed to clone this project.

## Pre-processing
The following script was used to read indexed files of all the images for tuning/training the models. 
* <b>face_detection.py</b>:  Reads 2 indexed CSV files of all the data for training/testing; then processes each image to detect only the faces in the image and removes any noise.

## Modeling
The following scripts show the process used in the modeling.
  * <b>model_tuning.py</b>:  Performs hyper-parameter tuning for a CNN model.
  * <b>model_training.py</b>:  Implementation of the best CNN model found in tuning.
  
## Assessment
The following script was used to assess the quality of the best tuned CNN model.
* <b>eval_plotter.py</b>:  Implements a confusion matrix from using 'sklearn'.
* <b>model_assessment.py</b>:  Assesses the best tuned on the validated set not used in training, and visualizes the results.

## CSV Files Used in the Project
The following are the CSV files that contain links and class labels for the training and testing image data from .  
* <b>Training Set</b>:  This is the output of the face_detection.py using the original indexed list.  These files were used to create the dataframes on which the models are trained and assessed.
  * training_images.csv
  * validation_images.csv
  * training_images_preprocessed.csv
  * validation_images_preprocessed.csv
* <b>Tuning Results</b>:  This is the output of model_tuning.py.  These results were used to tune the models. Over 2000 different models were created and tested in this project.
  * tuning_results.csv
  
## Write-up
* <b>Facial Expression Recognition using CNN.pdf</b>:  Write-up for this project.

## Poster
