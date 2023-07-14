# How to proceed

## Paper 1: Using Autoencoders
### x,y,z -> FVC -> Autoencoder -> Sequencer -> KNN+DTW
1. Create the FVC file to encode joint vectors, vels, acc, ang vels and ang accs. 
2. Then write autoencoder. Need to train, extract weights and then use it for knn.
3. Create sequencer: Has the parameter of sliding window length and sliding window steps   
4. Write the knn with DTW as its distance metric. 
Consider for FVC that you are not recieving 33 landmarks. Instead, you are getting the major ones which another file will take care of. 
## Paper 2: Using Sequencers
1. We will keep the pose tracking and landmark2Dto3D as it is. 
2. We will change what comes afterwards

We have two ways to proceed, and it would be better if we try both of them. One is to use HMM and the other is to use GRU network. We will try the HMM way first and complete the implementation. 
It involves the following
1. Use knn to classify each frame into predefined gestures
2. For feature vectors, we will use joint pair vectors, joint angles, joint velocities and joint acc.
3. We will have to do feature selection and dimension reduction otherwise our vector space will be high dimensional
4. We will then use HMMs to classify sequence of gestures into movements - HMM require preprocessing
5. We will do some data augemnetations like noise, rotation and scaling, especially if the drone is moving, thus making the human change size and rotation in succesive frames. 
6. Regularization
7. Parameter tuning of knn and HMM
8. Evaluation - Accuracy, Precision, recall(sensitivity), F1 score and Area under the ROC curve
9. We will then localize the human in the drone's frame to be able to give relative commands. 
10. Encode the behavior of robots given certain commands
