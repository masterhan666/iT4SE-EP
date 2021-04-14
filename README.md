iT4SE-EP
=========================
iT4SE-EP was developed for identifying T4SEs by extracting evolutionary features from the position-specific scoring matrix and the position-specific frequency matrix profiles. First, four types of encoding strategies were designed to transform protein sequences into fixed-length feature vectors based on the two profiles. Then, the feature selection technique based on the random forest algorithm was utilized to reduce redundant or irrelevant features without much loss of information. Finally, the optimal features were input into a support vector machine classifier to carry out the prediction of T4SEs.

Usage
=========================
To run: $ iT4SE-EP(180D).py or iT4SE-EP(320D).py

The iT4SE-EP(180D).py and iT4SE-EP(320D).py files implement the SVM algorithm with RBF kernel and the RF feature selection method to train and evaluate the model. If you want to use different training and test data, please change the file name inside the file.
