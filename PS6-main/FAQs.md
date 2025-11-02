Report

Please use the template provided, and do not delete slides allotted for a question.

Q. In part 1, I am getting the autograder error "eigenvalues are not sorted in descending order". Why?\
A. Use eigh() instead of eig() as complex numbers confuse the autograder.

Q. Should the output image contain all overlapping face detection boxes in the last question?\
A. Either one box covering the face or multiple boxes covering all face areas are fine.

Q. Can we use 1/0 for positive/negative samples as in the literature, instead of 1/-1 as in the homework document in case of boosting?\
A. Yes. Feel free to use it if you are getting better results

Q. Can we change [sizei-1, sizej-1] to [sizei, sizej] while calculating Haar Features?\
A. If using [sizei, sizej] works better with your code then you can modify it. The change might not be needed but we are completely fine with it if you change it.

Q. What does “"Report the average accuracy over 5 iterations." mean in part 2?\
A. It means you should run the code 5 times and manually calculate the average.

Q. What does “"How do your results change when selecting different values for number of iterations?" mean?\
A. It means you should change num_iterations. However, you don't have to manually run the code 5 times and calculate the average, you can just run it, for example, once for num_iterations=[1,10, 20, 30, 40, 50] and then see how the performance changes. Does it get better with more iterations? Are there diminishing returns? Is there some number of iterations that work best and then performance drops off? What trends are you seeing?

Q. For part 1, Is the mean face supposed to look like a face?\
A. The result should look like a face that presents some smoothing.

Q. What accuracy is expected in Viola Jones?\
A. Training: 100%, Testing: 80%. You should be ok if you are able to achieve Training: > 80% Testing: > 65% .

Q. What accuracy is expected in 2?\
A. Random classifier: around 50%, Weak Classifier: above 80%, Boosting: Around 90%.

Q. How to understand weak classifier?\
A. The columns are the features and the rows the observations. The weak classifier here can indeed be seen as a single node decision tree. This weak classifier is a threshold on a single feature.
For each columns we find the threshold along with the sign that is +1/-1 (or 1/0) if x < threshold depending on the class definition) that outputs the lowest error. Then we select the column (i.e feature) that yields to the lowest error overall.

Once we found it we can define a weak classifier as:\
self.feature = column id that has the lowest error\
self.threshold = threshold value that yields to this lowest error\
self.sign = +1/-1 (or 1/0 based on the convention chosen).

Q. What is a 'top' eigenvector?\
A. Take the K largest eigenvectors.

Q. Please explain in Part 4c "build your own from the man.jpeg image by selecting subimages".\
A. The "pos" and "neg" folders contain examples of subimages. You can create more from the man.jpg. By using subimages from the same target image you should have a very "robust" detector for that specific image but not necessarily robust for random images. For that, you will need a lot more training data. This isn't like "template" matching, so the detection won't necessarily be perfect.

Q. How do you populate the score matrix inside predict in Viola Jones?\
A. The "scores" matrix contains all the features (cols) computed for all input images (rows).\
Here you have two options:\
1- You can populate the "scores" matrix using the for loop like in the train() function.\
2- Since each weak classifier operates only on one feature, you don't need to compute all the features in the "scores" matrix. So you can save time by only populating the columns of "scores" defined by clf.feature for each clf in self.classifier.


