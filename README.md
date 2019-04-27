## CS760-ML-Project

#### We want to perform Supervised Multi-Task Learning to evaluate how different learners learn from Audio Data. The multiple tasks for our learners is to learn the following features:

* Year that a Song was Released on Record
* Song Performer
* Song Genre [Most important feature to predict for us]


To do this, we have two different ways of extracting features from Audio Signals
* **Essentia Features** - Vector representations of data extracted from Audio Clips
* **Mel Spectrograms** - Image representations of data extracted from each audio clip


From these features, we can construct four categories of learners:
* **[Baseline] Single Task learners of Essentia Features** - This would be Multiclass Logistic Regression / SVM to learn Genre and Performer, and Linear regression to learn year of publication.
* **Multitask Learning of Essentia Features** - This would be using a Dense ANN to predict Genre, Performer, and Year simultaneously; with a loss function composed of a sum of losses over those 3 outputs and sharing at least one hidden layer in prediction.
* **Single Task CNNs trained on Mel Spectrograms** - These would be CNNs trained on one specific category at a time (Year, Artist, Genre), learning weights separately from each other.
* **Multitask CNN trianed on Mel Spectrogram** - This would be a CNN trained on Mel Spectrograms that shares hidden units, and has one output for Genre, one output for Artist, and one output for Year. Ideally, training using all three outputs would be informative enough to learn hidden features better than the Single Task CNNs would.

To properly evaluate these different learning models, we propose the following experiment:

* Perform Stratified Sampling to split our dataset into 80% train, 20% Test.
On our train set only, perform 10-fold Cross Validation to recover optimal hyperparameters for the above four learners.
* For the baseline model, that means selecting the highest performing linear model (logistic regression, SVM, linear regression). For the neural networks, this may mean selecting the best architecture / hyperparameters
* The ‘best’ setting in this case has the highest accuracy on the test fold
* Once we’ve selected the best parameters for the four learners, train the model with the ‘best’ setting from each learner on the entire set and evaluate them on the test set.
* After doing this, we compare the performance of the four learners on the test set and comment on them.
