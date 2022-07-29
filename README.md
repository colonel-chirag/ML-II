
# ML-II

1.> ML PROCESS FLOW W PROCESS DIAGRAM 

2.> Bias - Variance Trade-off .

3.> Overfitting vs Underfitting .

4.> KNN

5.> PCA / Dimensionality Redction .

6.> polynomial regression / Multiple linear regression 

----------------

1.> CV / CV techniques.

2.> Regression vs Classification.

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

5.> Explain machine learning in brief. Discuss application & limitations of machine learning.

6.> Discuss various types of machine learning algorithm.

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

9.> Explain Support Vector Machine (SVM) algorithm with neat &amp;amp; clean diagram.

10.> Different- different value handling approaches.

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

