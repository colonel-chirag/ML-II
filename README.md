
# ML-II

1.> ML PROCESS FLOW W PROCESS DIAGRAM 

2.> Bias - Variance Trade-off .

3.> Overfitting vs Underfitting .

4.> KNN

5.> PCA / Dimensionality Redction .

6.> polynomial regression / Multiple linear regression 

----------------

1.> CV / CV techniques.

There are some common methods that are used for cross-validation. These methods are given below:
1.Validation Set Approach
In this we divide input data into training set and test or validation set in the validation set approach. Both the subsets are given 50% of the dataset.But it has one of the big disadvantages that we are just using a 50% dataset to train our model, so the model may miss out to capture important information of the dataset. It also tends to give the underfitted model

2.Leave-P-out cross-validation
It means, if there are total n datapoints in the original input dataset, then n-p data points will be used as the training dataset and the p data points as the validation set. This complete process is repeated for all the samples, and the average error is calculated to know the effectiveness of the model.There is a disadvantage of this technique; that is, it can be computationally difficult for the large p.

3.Leave one out cross-validation.
It means, in this approach, for each learning set, only one datapoint is reserved, and the remaining dataset is used to train the model. This process repeats for each datapoint. Hence for n samples, we get n different training set and n test set. 

4.K-Fold Cross-Validation
In this inpury dataset is divided into K groups of samples of equal sizes. For each learning set, the prediction function uses k-1 folds, and the rest of the folds are used for the test set.And the output is less biased than other methods.
The steps for k-fold cross-validation are:
Split the input dataset into K groups
For each group:
Take one group as the reserve or test data set.
Use remaining groups as the training dataset
Fit the model on the training set and evaluate the performance of the model using the test set.
Consider the below diagram:

![84f4137b-62f5-448e-8898-0930971c9267](https://user-images.githubusercontent.com/93399136/181699813-12a21801-3675-41ca-a3b6-0755cf1d9d4f.jpg)


5.Stratified k-fold cross-validation
This approach works on stratification concept, it is a process of rearranging the data to ensure that each fold or group is a good representative of the complete dataset. To deal with the bias and variance, it is one of the best approaches.

6.Holdout Method
In this method, we need to remove a subset of the training data and use it to get prediction results by training it on the rest part of the dataset.This method is the simplest cross-validation technique among all.Although this approach is simple to perform, it still faces the issue of high variance, and it also produces misleading results sometimes.

2.> Regression vs Classification.

Regression Algorithm	vs Classification Algorithm

1.In Regression, the output variable must be of continuous nature or real value.	

In Classification, the output variable must be a discrete value.

2.The task of the regression algorithm is to map the input value (x) with the continuous output variable(y).	
The task of the classification algorithm is to map the input value(x) with the discrete output variable(y).

3.Regression Algorithms are used with continuous data.
	Classification Algorithms are used with discrete data.

4.In Regression, we try to find the best fit line, which can predict the output more accurately.	
In Classification, we try to find the decision boundary, which can divide the dataset into different classes.

5.Regression algorithms can be used to solve the regression problems such as Weather Prediction, House price prediction, etc.	Classification Algorithms can be used to solve classification problems such as Identification of spam emails, Speech Recognition, Identification of cancer cells, etc.

6.The regression Algorithm can be further divided into Linear and Non-linear Regression.	The Classification algorithms can be divided into Binary Classifier and Multi-class Classifier.

3.> Curse of dimensionality .
Curse of Dimensionality refers to a set of problems that arise when working with high-dimensional data. The dimension of a dataset corresponds to the number of attributes/features that exist in a dataset. A dataset with a large number of attributes, generally of the order of a hundred or more, is referred to as high dimensional data. Some of the difficulties that come with high dimensional data manifest during analyzing or visualizing the data to identify patterns, and some manifest while training machine learning models. The difficulties related to training machine learning models due to high dimensional data are referred to as the ‘Curse of Dimensionality’.
Domains of the curse of dimensionality are listed below :

Anomaly Detection
Anomaly detection is used for finding unforeseen items or events in the dataset. In high-dimensional data anomalies often show a remarkable number of attributes which are irrelevant in nature; certain objects occur more frequently in neighbour lists than others.

Combinatorics
Whenever, there is an increase in the number of possible input combinations it fuels the complexity to increase rapidly, and the curse of dimensionality occurs.

Machine Learning
In Machine Learning, a marginal increase in dimensionality also requires a large increase in the volume in the data in order to maintain the same level of performance. The curse of dimensionality is the by-product of a phenomenon which appears with high-dimensional data.

How To Combat The CoD?
Combating COD is not such a big deal until we have dimensionality reduction. Dimensionality Reduction is the process of reducing the number of input variables in a dataset, also known as the process of converting the high-dimensional variables into lower-dimensional variables without changing their attributes of the same.

4.> AL/ML/DL

![a9adc9ea-c20d-49ef-b64e-ab0e7f83546a](https://user-images.githubusercontent.com/93399136/181699910-8e1cdf85-8835-4c14-bbbc-ff5078e14c4f.jpg)

5.> Explain machine learning in brief. Discuss application & limitations of machine learning.

Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values.
Applications:
Predictions while CommutingVideos Surveillance
Email Spam and Malware Filtering
Search Engine Result Refining
Product Recommendations
Limitations:
ML needs enough time to let the algorithms learn and develop enough to fulfill their purpose with a considerable amount of accuracy and relevancy. It also needs massive resources to function. This can mean additional requirements of computer power for you.


Machine Learning requires massive data sets to train on, and these should be inclusive/unbiased, and of good quality. There can also be times where they must wait for new data to be generated.


Another major challenge is the ability to accurately interpret results generated by the algorithms. You must also carefully choose the algorithms for your purpose.

6.> Discuss various types of machine learning algorithm.

Machine Learning algorithms are the programs that can learn the hidden patterns from the data, predict the output, and improve the performance from experiences on their own. Different algorithms can be used in machine learning for different tasks, such as simple linear regression that can be used for prediction problems like stock market prediction, and the KNN algorithm can be used for classification problems.

In this topic, we will see the overview of some popular and most commonly used machine learning
algorithms along with their use cases and categories.

Types of Machine Learning Algorithms
Machine Learning Algorithm can be broadly classified into three types:

1.Supervised Learning Algorithms.
2.Unsupervised Learning Algorithms.
3.Reinforcement Learning algorithm.

The below diagram illustrates the different ML algorithm, along with the categories:

![machine-learning-algorithms](https://user-images.githubusercontent.com/93399136/181700699-5b9e6ad6-760d-4dfc-ac81-3f61f68e3687.png)

Machine Learning Algorithms
1) Supervised Learning Algorithm
Supervised learning is a type of Machine learning in which the machine needs external supervision to learn. The supervised learning models are trained using the labeled dataset. Once the training and processing are done, the model is tested by providing a sample test data to check whether it predicts the correct output.

The goal of supervised learning is to map input data with the output data. Supervised learning is based on supervision, and it is the same as when a student learns things in the teacher's supervision. The example of supervised learning is spam filtering.

Supervised learning can be divided further into two categories of problem:

Classification
Regression
Examples of some popular supervised learning algorithms are Simple Linear regression, Decision Tree, Logistic Regression, KNN algorithm, etc. Read more..

2) Unsupervised Learning Algorithm
It is a type of machine learning in which the machine does not need any external supervision to learn from the data, hence called unsupervised learning. The unsupervised models can be trained using the unlabelled dataset that is not classified, nor categorized, and the algorithm needs to act on that data without any supervision. In unsupervised learning, the model doesn't have a predefined output, and it tries to find useful insights from the huge amount of data. These are used to solve the Association and Clustering problems. Hence further, it can be classified into two types:

Clustering
Association
Examples of some Unsupervised learning algorithms are K-means Clustering, Apriori Algorithm, Eclat, etc. Read more..

3) Reinforcement Learning
In Reinforcement learning, an agent interacts with its environment by producing actions, and learn with the help of feedback. The feedback is given to the agent in the form of rewards, such as for each good action, he gets a positive reward, and for each bad action, he gets a negative reward. There is no supervision provided to the agent. Q-Learning algorithm is used in reinforcement learning. Read more…


7.> Discuss Gradient Descent Algorithm in detail.

Gradient Descent is an optimization algorithm used for minimizing the cost function in various machine learning algorithms. It helps in finding local minimum and maximum. 

The cost function is defined as the measurement of difference or error between actual values and expected values at the current position and present in the form of a single real number. 

The best way to define the local minimum or local maximum of a function using gradient descent is as follows:

-If we move towards a negative gradient or away from the gradient of the function at the current point, it will give the local minimum of that function.

-Whenever we move towards a positive gradient or towards the gradient of the function at the current point, we will get the local maximum of that function.

![gradient-descent-in-machine-learning1](https://user-images.githubusercontent.com/93399136/181694353-091b01dd-2d86-4ccb-af33-e699d1fc2613.png)


Types of gradient Descent:

A) Batch Gradient Descent: This is a type of gradient descent which processes all the training examples for each iteration of gradient descent. But if    the number of training examples is large, then batch gradient descent is computationally very expensive. Hence if the number of training examples is      large, then batch gradient descent is not preferred. Instead, we prefer to use stochastic gradient descent or mini-batch gradient descent.

B) Stochastic Gradient Descent: This is a type of gradient descent which processes 1 training example per iteration. Hence, the parameters are being updated even after one iteration in which only a single example has been processed. Hence this is quite faster than batch gradient descent. But again, when the number of training examples is large, even then it processes only one example which can be additional overhead for the system as the number of iterations will be quite large.

C) Mini Batch gradient descent: This is a type of gradient descent which works faster than both batch gradient descent and stochastic gradient descent. Here b examples where b<m are processed per iteration. So even if the number of training examples is large, it is processed in batches of b training examples in one go. Thus, it works for larger training examples and that too with lesser number of iterations.

8.> Explain working of Decision Tree based machine learning algorithm using suitable example.

-Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

-In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.

-The decisions or the test are performed on the basis of features of the given dataset.

-It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.

-It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.

-In order to build a tree, we use the CART algorithm, which stands for Classification and Regression Tree algorithm.

-A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.

-Below diagram explains the general structure of a decision tree:


![decision-tree-classification-algorithm](https://user-images.githubusercontent.com/93399136/181695258-9f551bdd-c30c-4717-974a-950c83e50477.png)

9.> Explain Support Vector Machine (SVM) algorithm with neat &amp;amp; clean diagram.

-SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning. 

-The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

-Consider the below diagram in which there are two different categories that are classified using a decision boundary or hyperplane:

![support-vector-machine-algorithm](https://user-images.githubusercontent.com/93399136/181695969-57f8fac3-2dbd-4a08-8f5c-2f1bc74afbf6.png)

-SVM algorithm can be used for Face detection, image classification, text categorization, etc.

SVM can be of two types:

-Linear SVM: Linear SVM is used for linearly separable data, which means if a dataset can be classified into two classes by using a single straight line, then such data is termed as linearly separable data, and classifier is used called as Linear SVM classifier.

-Non-linear SVM: Non-Linear SVM is used for non-linearly separated data, which means if a dataset cannot be classified by using a straight line, then such data is termed as non-linear data and classifier used is called as Non-linear SVM classifier.

10.> Different- different value handling approaches.

There are many methods available to treat the missing data in the literature, textbook and standard courses. However, these methods consist of some drawbacks. When using the data mining process, one need to be careful to avoid bias or overestimate variability; these methods don't perform well.

Case Deletion

There are two types of case deletion methods. The first one is a list deletion (also known as complete case analysis) and second method is the pair deletion. The case deletion methods are used to delete the missing cases from our dataset on an analysis-by-analysis basis.

![missing-data-conundrum-exploration-and-imputation-techniques](https://user-images.githubusercontent.com/93399136/181777330-f624541b-02e1-4fdf-9a19-0755de589861.png)



Case Deletion
syllabus

![image](https://user-images.githubusercontent.com/59536110/180618612-c8bc0c75-33a5-43b8-adad-f9ce9bcbdf47.png)

Sessional Question Papers-

![IMG20220727114322](https://user-images.githubusercontent.com/93399136/181174539-ffba08c8-7663-4cb4-95a7-07218c979f04.jpg)
Unit - 1

1. Explain machine learning in brief. Discuss application &amp; limitations of machine learning.
2. Compare Low dimensional &amp; High dimensional Data in reference to machine learning
3. Differentiate among AI, ML &amp; DL.
4. Discuss various types of machine learning algorithm.
5. Explain, why the knowledge of linear algebra, statistic &amp; probability theory in beneficial
in machine learning development.

Unit - 2

1. Explain what bias and variance and their effect on accuracy of ML model developed. Also
discuss bias-variance trade-off in brief.
2. Differentiate between Overfitting &amp; Underfitting in machine learning.
3. Explain cross validation (CV) in brief using diagrams and discuss different-different CV
techniques used in machine learning.
4. What do you mean by performance metrics/KPI? List different-2 KPI used to evaluate ML
model performance.
5. What do you understand by fine tuning a ML model? Discuss about Grid Search &amp; Randomized search
method in brief.
6. What do you understand by ensemble learning? Also discuss concept of bagging &amp; boosting in
detail.


Unit - 3

1. Explain regression &amp; classification problem using suitable example.
2. Explain Linear regression using suitable diagram.
3. Discuss Gradient Descent Algorithm in detail.
4. Discuss Multiple linear regression (MLR) using example.
5. Explain Polynomial Regression in brief and discuss how it is different from multiple
linear regression.


Unit - 4
1. Why visualization is used in machine learning? Discuss its importance in data exploration
phase of ML.
2. What do you understand by “Curse of Dimensionality”? Does proper Feature Selection
help in avoiding this? Please discuss in brief.
3. Explain the term Dimensionality Reduction in brief. Also discuss the working of
Principal
4. Component Analysis (PCA) algorithm.
5. Explain K Nearest Neighbors (KNN) and its process in detail using suitable diagram?


Unit - 5

1. What do you understand by clustering in ML? Discuss circumstances, when it is
applicable to use for ML model development.
2. Explain working of K-Means Clustering using appropriate diagram.
3. Explain working of Decision Tree based machine learning algorithm using suitable
example.
4. Discuss process of Random Forest algorithm in detail.
5. Explain Support Vector Machine (SVM) algorithm with neat &amp;amp; clean diagram.

Solutions https://drive.google.com/file/u/0/d/1Ms5AazjGox7e4Hxx3HfJ6efqtWEj3MCm/view?usp=drive_web

