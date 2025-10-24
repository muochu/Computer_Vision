Frequently asked questions: Please check for your questions here before asking below. We will keep updating as we see new repeated questions.

Report

Please use the template provided, and do not delete slides allotted for a question. And DO NOT forget to upload your reports Gradescope.

<!--- Q : Can we use matchTemplate as the sensor for questions 5 and 6 in experiment.py ?
A : You may use matchTemplate only as a sensor for KF. --->

Q : For Part 2.b, how should we deal with debate with heavy noise? Do we need blurring before we calculate MSE?
A : No need to blur. Just parameter tuning will get you desired result provided PF is correctly implemented.

Q : For Part 5, are we supposed to track multiple persons at the same frame? So if there are 2 persons in the same frame, we should show 2 trackers for each person in this image ?
A : Yes, you have to show multiple trackers in the same frame.

Q : Could we utilize our implementation of the KalmanFilter within MDParticleFilter to assist with dynamics?
A : Yes, you can reuse parts of the code written by you; if you think it saves you time.

Q : Could we utilize a hybrid of both Kalman and Particle filters within Part 5?
A : Using a hybrid is okay for your personal experiments, but the question is more focused on difference between the two. Hence, utilising a hybrid wouldn’t serve the purpose.

Q : For part 5, are we supposed to extract the templates from the running code so the TAs can run it, or can we extract it manually and load it in?
A : Extraction from code itself is appropriate and convenient.

Q : For Part 1.c, what part of the walking man's body is the blue circle trying to capture?
A : Any part of the body should be fine, as long as it is being tracked correctly.

Q : For part 5, are we allowed to just track the upper half of the body?
A : Yes you may if that works for you.

Q : Can we use multiple filters for part 5? One filter per tracked template, or is the point to use one filter to track 3 targets?
A : Feel free to use multiple filters.

Q : What is the tolerance for detecting the bouncing circle in part 1 b ? Is it enough for the blue circle to be completely inside the bouncing ball ?
A : Yes that should be enough.

Q : Since there is randomness in each run because of initializations in PF - can we use seeds for reproducibility?
A : With or without seeds, particles should converge with enough iterations. But if you want to improve initialization, there are two approaches you can take here. The first is to use the extra knowledge you have about the template location to initialize the particles well. The second is to only update the template if some measure of convergence is sufficiently good (e.g. particle standard deviation is < some threshold).

Q : What are we supposed to be tracking for part 2B?
A : His face has to be tracked.

Q : If a particle is very close to the frame border, then we cut a frame_cutout, it will has some part outside of the frame. How can we deal with that properly?
A : There are a few ways of looking at this 
    - Use the area of the template that would still remain inside the image. Here you would have to be careful in not modifying the original template so that it can be used with other particles.
    - Expand the image borders padding the image (frame) either replicating or reflecting the contents close to the border. This approach would not be that efficient because you would be performing this operation on the entire image. Unless you code your algorithm to just replicate the area the template will occupy leaving the rest of the array untouched. This may not be that easy since you can't just expand a range of rows and columns using the same data structure with just numpy / opencv operations.
    - Ignore the particles that would place the template (or part of it) out of bounds. In this case you will have to set the weights of such particles to 0 after not using them so that they are not resampled in the next iteration. In practice, this method works well in the context of the assignment even though you are not using all the available particles.

Q : Looking at the problem setup description for part_5 (Tracking Multiple Targets), are the following allowed :
- Taking a sample patch of each person (similar to that shown in the problem description)
- Defining between what frame number we are expected to see the person (e.g. frame 1 to 53 for the first person going off-camera)
A : Yes and yes

Q : What should we do about the bounding box when the person exits the frame?
A : You can make the tracking box disappear when the person leaves the scene.

Image Dataset Links:

[pres_debate_noisy.zip](https://d1b10bmlvqabco.cloudfront.net/attach/j6l30v0ijo95hv/iddrzpx4w0eew/j8uwb29jxq7o/pres_debate_noisy.zip) \
[pres_debate.zip](https://d1b10bmlvqabco.cloudfront.net/attach/j6l30v0ijo95hv/iddrzpx4w0eew/j8uwmhb5ty4w/pres_debate.zip) \
[pedestrians.zip](https://d1b10bmlvqabco.cloudfront.net/attach/j6l30v0ijo95hv/iddrzpx4w0eew/j8uwf9ijuc2s/pedestrians.zip) \
[follow.zip](https://d1b10bmlvqabco.cloudfront.net/attach/j6l30v0ijo95hv/iddrzpx4w0eew/j8uy2nsntsjy/follow.zip) \
 [circle.zip](https://d1b10bmlvqabco.cloudfront.net/attach/j6l30v0ijo95hv/iddrzpx4w0eew/j8uy3s12posw/circle.zip) \
[TUDCampus.zip](https://d1b10bmlvqabco.cloudfront.net/attach/j6l30v0ijo95hv/iddrzpx4w0eew/j8uy45ce30fe/TUDCampus.zip) \
[walking.zip](https://d1b10bmlvqabco.cloudfront.net/attach/j6l30v0ijo95hv/iddrzpx4w0eew/j8uy4syx5prd/walking.zip) \
[input_test.zip](https://d1b10bmlvqabco.cloudfront.net/attach/j6l30v0ijo95hv/iddrzpx4w0eew/j8z7fgo6qw5d/input_test.zip)



Banned functions :

Should not be used during filtering -
```
cv2.calcHist, cv2.compareHist, cv2.matchTemplate, np.histogram
```
Q: I am getting ```The autograder failed to execute correctly. Contact your course staff for help in debugging this issue. Make sure to include a link to this page so that they can help you most effectively.``` errors after working on part_5().\
A: Do not add any new function calls to ps5_utils.py. Autograder has it's own copy, and modifications to this file would not be known to the autograder. 
