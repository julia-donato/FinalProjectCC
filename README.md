### Optimizing Disease Diagnosis on Amazon Web Services
#### By: Julia Donato

#### Introduction:

Through the use of ensemble learning techniques on a cloud platform, this research seeks to construct a reliable and effective disease diagnostic model. The model aims to address the issue of feature redundancy in large medical datasets and enhance the performance of conventional deep learning models. The improved model will be made available on a cloud platform in order to guarantee scalability, availability, and accessibility.

The motivation for this project is to research methods for improving disease diagnosis models in cloud environments. Key challenges include addressing limited feature selection methods in most disease diagnosis studies, managing feature redundancy in large medical datasets, and ensuring the scalability and availability of the optimized disease diagnosis model on a cloud platform.

Previous approaches in this field have typically concentrated on conventional deep learning models with constrained feature selection techniques and sparse cloud service utilization. With the help of ensemble learning strategies, feature selection methods, and cloud-based deployment, this project seeks to progress the field by enhancing the accuracy of disease detection.

#### Background and Related Work:

The research done for this project demonstrates the usefulness of genetic algorithms (GAs) for a variety of machine learning applications, such as feature selection, model improvement, and deep learning training. In order to find the optimal answer to a given problem, genetic algorithms are a sort of optimization algorithm that mimics the process of natural selection. Lamos-Sweeney [1] increased the performance of deep neural networks by using GAs to adjust their hyperparameters. Using genetic algorithms for feature and model selection, Abdollahi [2] created a hybrid stacked ensemble model that outperformed conventional machine learning models in predicting diabetes. Such [3] showed that genetic algorithms might train deep neural networks for reinforcement learning tasks more effectively than conventional gradient-based techniques.

Deep learning approaches have also showed potential in a number of applications, alongside genetic algorithms. An effective deep neural network-based intrusion detection system for cloud environments was proposed by Chiba et al. [4] and uses a self-adaptive evolutionary approach for feature selection. Using a convolutional neural network with transfer learning, Hasan et al. [5] created a deep learning-based method for identifying and segmenting COVID-19 and pneumonia on chest X-ray images. An ensemble of deep neural networks trained using transfer learning was utilized by Tanveer et al. [6] to classify Alzheimer's disease with great sensitivity and accuracy. In order to diagnose diseases in cloud environments, Veerasekaran and Sudhakar [7] suggested a genetic algorithm-based feature selection and neural network-based classification model.

The potential of genetic algorithms and deep learning methods for disease diagnosis, intrusion detection, and security in cloud systems is shown by these works taken as a whole. These methods provide precise and effective methods for disease early detection and diagnosis, as well as efficient security safeguards for cloud computing platforms.

#### The Data:

All data used came from the University of California Irvine’s Machine learning repository.
A variety of datasets were used for pre-model building exploration. The heart disease dataset was processed with spark in Amazon Web Services (AWS) and used to train the final model. The datasets are as follows:

 
•	[8] Heart Disease Dataset
o	Attributes: 16
o	Size: 79 KB
o	Target: ‘num’
•	[9] Cervical Cancer Dataset
o	Attributes:
o	Size: 4 KB
o	Target: ‘ca_cervix’
•	[10] Breast Cancer Dataset
o	Attributes: 32
o	Size: 20 KB
o	Target: ‘target’
•	[11] Parkinson’s Dataset
o	Attributes: 23
o	Size: 41 KB
o	Target: ‘status’ 


#### Exploration of Modeling and Feature Selection Techniques:

Genetic Algorithms were investigated in this study as a feature selection technique to choose the most pertinent features from the medical dataset. Relevant research in the topic combines deep learning and genetic algorithm feature selection for disease diagnosis.

An attempt was made to develop an adaptive genetic algorithm from scratch, however despite having a substantially longer processing time, it performed nearly as well as Scikit-Learn's GeneticSelectionCV. An adaptive genetic algorithm is used because it may constantly modify the search space based on the algorithm's performance, which can promote faster convergence and better outcomes. The processing of the data could be the cause of the adaptive genetic algorithm's failure to exceed GeneticSelectionCV.

Genetic Algorithms
F1-score of the adaptive genetic feature selection	0.985
F1-score of the GeneticSelectionCV:	0.985
Table (1) F1 Scores of Genetic Selection Algorithms

The accuracy of different machine learning models for predicting heart disease using two different feature selection techniques—Chi-squared and Genetic Algorithm—is shown in Table 2. Logistic Regression, Decision Trees, Support Vector Machines (SVM), AdaBoost, and Naive Bayes are the models utilized. In general, the Chi-squared technique outperformed the genetic algorithm by a small margin.

Accuracy Heart Data (Features in Parenthesis)
	No Selection	Chi-squared	Genetic Algorithm
Logistic Regression	 0.8312 (All)	0.8279 (10) 	0.8312 (9)
Decision Tree	0.9805 (All)	0.9578 (10)	0.9805 (9)
SVM	0.6623 (All)	0.6753 (10)	0.6688 (9)
AdaBoost	0.8896 (All)	0.8701 (10)	0 .8896 (9)
Naïve Bayes	0.8084 (All)	0.8182 (10) 	0.8182 (9)
Table 2: Accuracy Heart Data

Subsequently, the focus shifted from feature selection to comparing GeneticSelectionCV to GridSearchCV. GridSearchCV exhaustively searches through a given parameter grid and returns the combination of hyperparameters that provides the best performance based on the provided scoring metric. GridSearchCV is suitable for optimizing hyperparameters for a given model architecture and training dataset. As seen in Table 3, it was found that GridSearchCV consistently outperformed GeneticSelectionCV across the datasets.

 
 
 	Dataset
	Breast Cancer	Heart Disease	Parkinson's	Cervical Cancer
Accuracy of GeneticSelectionCV	95.61%	80.00%	87.18	80.00%
Accuracy of GridSearchCV	98.25%	78.54%	92.31	86.67%
Table 3: GeneticSelectionCV vs. GridSearchCV Accuracy

The investigation of machine learning libraries started at this point. It was found that XGBoost is useful at handling tabular data, making it a popular choice for structured data problems like classification and regression tasks. However, the majority of the relevant work employs deep learning with libraries like TensorFlow. XGBoost is renowned for both its precision and high performance.

Furthermore, XGBoost was found to be favorable for a cloud environment due to its efficiency and scalability. As seen in Tables 4-7, in comparison to TensorFlow, XGBoost outperformed on the datasets, solidifying its suitability for this project.

 
Breast Cancer Dataset
Test accuracy (TensorFlow)	97.08%
Test accuracy (XGBoost)	98.25%

Table 4: TensorFlow vs. XGBoost Breast Cancer

 
Parkinson’s Disease Dataset
Test accuracy (TensorFlow)	81.36%
Test accuracy (XGBoost)	91.53%

Table 5: TensorFlow vs. XGBoost Parkinson’s
 
 
Heart Disease Dataset
Test accuracy (TensorFlow)	82.14%
Test accuracy (XGBoost)	99.03%

Table 6: TensorFlow vs. XGBoost Heart Disease

 
Cervical Cancer Dataset
Test accuracy (TensorFlow)	81.36%
Test accuracy (XGBoost)	91.53%

Table 7: TensorFlow vs. XGBoost Cervical Cancer
 
#### Proposed: Architecture
 
Figure 1: Project Architecture

Utilizing a variety of AWS services and machine learning methods, a heart disease prediction model was created and deployed on Amazon SageMaker in this project. Figure 1 depicts the project's overall architecture, which is as follows:


1. Data is submitted by a user via a Flask interface.
2. As a proxy between the user interface and the Lambda function, Amazon API Gateway receives the data and processes it.
3. Data is received from API Gateway via the Lambda function, which then prepares it for the model and delivers it to Amazon SageMaker for model inference.
4. The Lambda function receives the model inference result and returns it to API Gateway.
5. The data used for model inference is kept in an S3 bucket, and API Gateway transmits the outcome back to the user interface.

After taking into account aspects like VPC isolation, encryption, access control, and built-in algorithms, it was decided to adopt Amazon SageMaker over Amazon EC2. SageMaker provides a secure and convenient environment for deploying machine learning models.

Apache Spark was used within SageMaker to preprocess the data, providing benefits in scalability and speed, which are essential for handling huge medical datasets.

The model was created using an XGBoost classifier after principal component analysis (PCA) was used to reduce the model's dimensionality and pick its features. The performance of the XGBoost model was enhanced by using PCA to minimize the dimensionality of the dataset while preserving the greatest amount of original data.

The number of components, learning rate, maximum depth, subsample utilized for boosting, and Gamma are just a few of the hyperparameters that were modified. The ability to tune hyperparameters within SageMaker streamlined this procedure.

Pandas, NumPy, Seaborn, and Matplotlib were used for data manipulation and visualization; train_test_split from the sklearn.model_selection module split the dataset; boto3 was used to communicate with AWS services; and the SageMaker Python SDK was used to work with Amazon SageMaker, including building, training, and deploying machine learning models.

Results:

 
 	Precision	Accuracy	Recall
XGBoost without tuning	80.2%	82.4%	77.7%
XGBoost with gridsearch	82.4%	82.4%	79.3%
Tuned XGBoost with PCA 	94.9%	95.5%	95.1%
Table 8: Final Model Results

In this section, the performance of the optimized model, Tuned XGBoost with PCA, is compared to that of XGBoost with grid search in terms of precision, recall, and accuracy. The results in Table 8 show significant improvements in all three metrics, highlighting the effectiveness of the proposed approach.

Precision:
When compared to the XGBoost model with grid search, the precision of the Tuned XGBoost with PCA model was 15.3% higher. This suggests that the optimized model is less likely to generate false positives when identifying cases of heart disease.

Recall:
When recall was compared, the Tuned XGBoost with PCA model outperformed the XGBoost model with grid search by 15.9%. This enhancement means that the improved model is more effective at locating all true positive cases and minimizing false negatives.

Accuracy:
The accuracy of the Tuned XGBoost with PCA model significantly outperformed the XGBoost model with grid search by a factor of 20.4%. The improved accuracy shows that the updated model performs better overall at correctly classifying cases of heart disease.

In every metric that was tested, the Tuned XGBoost with PCA model outperformed the XGBoost model with grid search, demonstrating its potency in predicting heart disease. The suggested method, which combines PCA for dimensionality reduction and XGBoost as the classifier, is confirmed to be an effective and accurate method for the detection of heart disease by the considerable gains in precision, recall, and accuracy.

#### Conclusion and Future Work:

In conclusion, by utilizing Amazon SageMaker's security, scalability, and built-in capabilities, this project was able to design and deploy a model for predicting heart disease. The usage of AWS services enabled a seamless and user-friendly experience, while the combination of PCA and XGBoost offered an effective and accurate approach for forecasting heart disease.

There are several recommendations for future work:

1. Explore cost-saving options on AWS, such as employing reserved instances or spot instances, optimizing instance utilization, and cutting data storage expenses.

2. In order to confirm that the optimized model remains accurate and efficient, deploy it in a production setting and track its performance over time.

3. To further improve the XGBoost model's performance, make minor adjustments. For example, use bagging techniques to decrease overfitting and boost model robustness.

4. Discuss the moral ramifications of using machine learning to diagnose diseases, taking into account issues with data privacy, potential biases in the training data, and the effects of false positives and false negatives. When scaling the model, it is essential to guarantee accuracy and security.
The optimized disease diagnosis model can be further enhanced and tailored to solve practical issues by taking into account various potential future paths, ultimately leading to better disease diagnosis and improved patient outcomes. 
 


References
[1] Lamos-Sweeney, J. (2012). Deep learning using genetic algorithms. In Proceedings of the 14th annual conference companion on Genetic and evolutionary computation (pp. 1425-1432). IEEE.
[2] Abdollahi, J. (2019). Hybrid stacked ensemble combined with genetic algorithms for Prediction of Diabetes. Health Information Science and Systems, 7(1), 1-13. IEEE.
[3] Such, F. P. (2017). Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning. In Proceedings of the Genetic and Evolutionary Computation Conference (pp. 497-504). IEEE.
[4] Chiba, Z., Abghour, N., Moussaid, K., Omri, A. E., & Rida, M. (2019). A Clever Approach to Develop an Efficient Deep Neural Network Based IDS for Cloud Environments Using a Self-Adaptive Genetic Algorithm. In 2019 International Conference on Advanced Communication Technologies and Networking (CommNet) (pp. 1-9). IEEE. https://doi.org/10.1109/COMMNET.2019.8742390
[5] Hasan, M. J., Alom, M. S., & Ali, M. S. (2021). Deep Learning based Detection and Segmentation of COVID-19 & Pneumonia on Chest X-ray Image. In 2021 International Conference on Information and Communication Technology for Sustainable Development (ICICT4SD) (pp. 210-214). IEEE. https://doi.org/10.1109/ICICT4SD50815.2021.9396878
[6] Tanveer, M., Rashid, A. H., Ganaie, M. A., Reza, M., Razzak, I., & Hua, K.-L. (2022). Classification of Alzheimer’s Disease Using Ensemble of Deep Neural Networks Trained Through Transfer Learning. IEEE Journal of Biomedical and Health Informatics, 26(4), 1453-1463. https://doi.org/10.1109/JBHI.2021.3083274
[7] Veerasekaran, K., & Sudhakar, P. (2019). An optimal feature selection based classification model for disease diagnosis in cloud environment. In 2019 International Conference on Smart Systems and Inventive Technology (ICSSIT) (pp. 163-167). IEEE. https://doi.org/10.1109/ICSSIT46314.2019.8987874
[8] R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, and V. Froelicher, "International application of a new probability algorithm for the diagnosis of coronary artery disease," The American Journal of Cardiology, vol. 64, no. 5, pp. 304-310, Aug. 1989. https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
[9] DR. Sobar, R. Machmud, and A. Wijaya, "Behavior Determinant Based Cervical Cancer Early Detection with Machine Learning Algorithm," Advanced Science Letters, vol. 22, no. 10, pp. 3120-3123, Oct. 2016. https://archive.ics.uci.edu/ml/datasets/Cervical%20Cancer%20Behavior%20Risk
[10] W. H. Wolberg, W. N. Street, and O. L. Mangasarian, "Breast cancer Wisconsin (diagnostic) data set," UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
[11] M. A. Little, P. E. McSharry, E. J. Hunter, and L. O. Ramig, "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease," IEEE Transactions on Biomedical Engineering, vol. 56, no. 4, pp. 1015-1022, Apr. 2009. https://archive.ics.uci.edu/ml/datasets/parkinsons
![image](https://github.com/julia-donato/FinalProjectCC/assets/121905325/a589c6fd-5dc6-417c-9856-58aac0af0bfb)
