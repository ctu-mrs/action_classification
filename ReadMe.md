# How to proceed

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
