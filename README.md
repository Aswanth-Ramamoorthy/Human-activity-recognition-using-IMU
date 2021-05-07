# HUMAN ACTIVITY RECOGNITION USING IMU DATASET

The objective of the project is to use the raw IMU data, to predict human activities such as walking, walking upstairs, walking downstairs, sitting, standing, Lying.


# Dataset


Link: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip



# Source

The data was gotten from 30 subjects with ages between 19 and 48 years. They performed six activities listed below with their smartphones held to their waits.


The six activities performed includes:

 1. Walking
 2. Walking Upstairs
 3. Walking Downstairs
 4. Sitting
 5. Standing
 6. Laying

Each individual recorded data for the accelerometer(linear acceleration) and gyroscope (angular velocity) in the x, y, z plane using their smartphones. Observations were recorded at 50 Hz (i.e. 50 data points per second). Each participant performed the sequence of activities twice, once with the device on their left-hand side and once with the device on their right-hand side.

# Data type


The .txt dataset has been used has been pre-processed. The pre-processing steps included:

 a. Pre-processing accelerometer and gyroscope using noise filters.
 b. Splitting data into fixed windows of 2.56 seconds (128 data points) with 50% overlap.
 c. Splitting of accelerometer data into gravitational (total) and body motion components.


The data however still require some more work to load.

Based on the pre-processing, we have three signal types in the data which are; total acceleration, body acceleration, and body gyroscope. Since each has three axes of data, we will have a total of nine variables for each time step.

However, future engineering was applied to the window data to have 561 element vectors of features with X data for train and test containing the engineered data and Y containing the corresponding activity for each data.

The dataset was split into the ratio of 70:30 for train and test sets respectively based on data for subjects, e.g. 21 subjects for train and nine for the test.
