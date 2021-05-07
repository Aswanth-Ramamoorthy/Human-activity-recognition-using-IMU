# Code

Machine learning technique used : SVM
Programming Language: Python

The code is divided into three main sections;

	1) Functions definition
	2) Plot of datasets
	3) evaluating the ML model i.e SVM.

# Functions Definition

	a) Load_data function : This takes the file name as parameter and returns the loaded data as a NumPy array.

	b) Load_dataset_group : This takes a parameter of a group of data (i.e. train or test) and returns the X and Y array
   	   for each group of data.

	c) Load_dataset:  This returns X and Y for both train and test group. It calls Load_dataset twice in the function.

	d) total_count: This takes in the Y array which contains the list of HAR activities and counts the number of times 
	   each activity appears then puts each value in a 1-D NumPY array with a size of  	6 which is the number of activities.

	e) Confusion_matrix: This takes in testy(expected ouptut) and yhat(predicted output) and compare the variance and 
	   generates a covariance matrix which is a 6X6 matrix. Here in ML, it is called 	confusion matrix.

	f) Pre_rec: This function will be used to calculate both precision and recall. For recall it takes in confusion matrix 
	   and the count of testy and compares each diagonal element of the confusion 	matrix with each element in y_test_count 
	   i.e cm[I][j]/y_test_count[I] and this is multiplied by 100% to get the recall. Same is done for yhat_count  and confusion matrix to get precision.

	g) Overall_accuracy: this takes the confusion matrix and testy as parameters and returns the accuracy of the algorithm.
   	   It sums up the diagonal element of the confusion matrix, and compare if 	tallies with the size of testy.


# Plot of Datasets 

The datasets were plotted using Principal component analysis(PCA). PCA was given 2 components and train x was fit into the PCA and testy was 
transformed to give two components. The items of these components for each activity was plotted against each other to show the distribution of activities for testy.




#  Evaluating the ML model i.e SVM.

After the function definition and plots are gotten, the next step is the evaluating of model;

	a) Firstly, we fit our trainx and trainy dataset in the SVM algorithm and generate a yhat which is our predication.
	b) Next we get call the total_count function to get our y_test_count and y_pred_count.
	c) Then we get our call the confusion_matrix function takin in testy and yhat to generate our confusion matrix
	d)Next,we get our recall and precision.
	e) Lastly, we determine the accuracy of the algorithm and print out our results.



