# AI-ML_project_295611

**Introduction:**
This project has been developed as part of AI ML class at Luiss Guido Carli. Authors are: Andrea Scarso (Team Captain, ID: 295611); Simona Miss Sultana (ID: 305441), Giovanni Cammalleri (ID: 303291)

The main goal of this project is to create a Classification model in order to predict scholar's Guild Memberships based on available data.
This process is made up of three main parts:
- EDA (Exploratory Data Analysis)
- Data Preprocessing
- Model configuration

**Methods:**

*Environment and Libraries*
The project was developed using Python 3.9.21 with several key libraries for data manipulation, visualization, preprocessing, and modeling. The libraries included:
-	Pandas and NumPy for data handling and analysis.
-	Seaborn and Matplotlib.pyplot for data visualization and correlation analysis.
-	Missingno for visualizing the distribution of missing values.
-	Scikit-learn for preprocessing, model evaluation, and performance scoring.
These libraries were instrumental in conducting the Exploratory Data Analysis (EDA) and preparing the data for modeling.
*Preprocessing*
1.	Manual Encoding:
    a.	Binary categorical columns were manually encoded, where "present" and "absent" were transformed into 1 and 0, respectively.

2.	Imputation:
    a.	Missing values in other columns were imputed using Scikit-learn’s SimpleImputer, with the mean of each column used as the fill value. Due to dataset size limitations, this simpler approach was prioritized over more complex techniques like KNN imputation.

    b.	Rows with missing values in the target variable were removed.

3.	Scaling:
    a.	To account for diverse data distributions, numerical columns were scaled using four methods from Scikit-learn’s preprocessing library: 
        i.	Standard Scaler: Standardizes features by removing the mean and scaling to unit variance.
        ii.	Min-Max Scaler: Scales features to a range of [0, 1].
        iii.	Robust Scaler: Scales data using statistics that are robust to outliers.
        iv.	MaxAbs Scaler: Scales features by dividing by the maximum absolute value.

4. Split train-validation-test:
The remaining dataset underwent stratified splitting into training, validation, and test sets, using Scikit-learn’s model selection tools, with the following ratios: 
- 60% training set
- 20% validation set
- 20% test set

*Model configuration*
Five machine learning models were evaluated based on four key metrics: Accuracy, Precision, Recall, and F1 Score:
 -	Random Forest: Provided a strong balance across all metrics, with an accuracy of 0.8461 and an F1 score of 0.7989.
 -	Gradient Boosting: Achieved the highest F1 score (0.8080) but had marginal gains over Random Forest.
 -	Logistic Regression, SVM, and KNN: Performed slightly lower across metrics, with KNN having the lowest scores.
Decision: Random Forest was selected due to its consistent performance across all metrics, computational efficiency, robustness against overfitting, and interpretability.

Hyperparameter Tuning
A grid search using Scikit-learn’s GridSearchCV was conducted to optimize the following parameters: 

o	n_estimators
o	max_depth
o	min_samples_split
o	min_samples_leaf
o	max_features

The best parameters identified were:
•	max_depth: None
•	max_features: sqrt
•	min_samples_leaf: 1
•	min_samples_split: 5
•	n_estimators: 200

Extensive hyperparameter tuning yielded no significant accuracy improvements. The final model was run with these parameters on the test set, and results were evaluated accordingly.

**Experimental design:**

1st experiment - Model comparison
The main purpose of comparing the different models was to analyze the models used for classification problems so that we could choose the one that best fits our needs. To choose the model we used the following as evaluation parameters: Accuracy, Precision, Recall, and F1 Score.
Based on the results we obtained, we reached the conclusion that the top-performing models are Gradient Boosting and Random Forest. Gradient Boosting achieved the highest scores across most metrics and Random Forest performed just as well, whilst offering the following advantages at the same time:
- Computational efficiency (faster training and tuning compared to Gradient Boosting).
- Robustness to overfitting.
- Easier interpretability, especially with feature importance analysis.
Logistic Regression also demonstrated a solid performance but fell slightly behind the top-performing models. SVM underperformed in terms of Recall and F1 Score, while KNN delivered the lowest scores across all metrics. 

2nd experiment – Hyper parameter tuning
The main purpose of hyperparameter tuning is to identify the parameters that will give us the best results for the given dataset. To avoid overfitting, we compared the model performance on the validation set. We utilized the GridSearchCV method, tuning the following parameters: 
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features
The grid search confirmed that Random Forest performs extremely well and is a flexible and robust solution. While the tuning process did refine the model slightly, the lack of significant performance gains proves that the original parameter set was already suitable for the problem. This further proves the stability and adaptability of Random Forest. 


**Results:**

The classification model shows commendable overall accuracy, achieving 84.67% with a ROC-AUC score of 0.8058, which reflects its strong capability in distinguishing between the three classes. However, upon deeper examination, discrepancies emerge in its performance across categories:

1. Class Performance Breakdown
- No_Guild Class
This category is reliably predicted, with a precision of 0.86, recall of 0.99, and an F1-score of 0.92. These metrics underline the model’s proficiency in identifying instances within this group.
- Master_Guild Class
Here, the model struggles, achieving moderate precision (0.55) but a notably low recall (0.11). This imbalance indicates a high prevalence of false negatives, signaling difficulty in correctly identifying samples from this class.
- Apprentice_Guild Class
The model performs poorly for this minority class, with precision, recall, and F1-scores all at 0.00. This failure points to issues such as class imbalance or insufficiently impactful features.

2. Feature Importance Insights

Key features driving the model’s decisions include:
- Mystical_Index (6.28%), Spell_Mastering_Days (6.13%), and Celestial_Alignment (6.12%), which play a pivotal role in distinguishing classes.
Conversely, features like Healer_consultation_Presence (0.76%) and Dexterity_check_Presence (0.71%) contribute minimally, suggesting their limited utility.

3. Misclassification Patterns

An examination of the confusion matrix reveals:
- The Master_Guild class is frequently misclassified as No_Guild, with 5,625 instances mislabeled.
- The Apprentice_Guild class is entirely unclassified, with all instances incorrectly assigned to other categories.

Obtained Results:
Accuracy: 0.8467369158980875
Precision: 0.7977096371562327
Recall: 0.8467369158980875
F1-Score: 0.7982318081996438

![alt text](https://github.com/andreascarso/AI-ML_project_295611/blob/main/Classification_report.png?raw=true)
![alt text](https://github.com/andreascarso/AI-ML_project_295611/blob/main/confusion_matrix.png?raw=true)
![alt text](https://github.com/andreascarso/AI-ML_project_295611/blob/main/ROC_curve.png?raw=true)

**Conclusions:**

Takeaway Points
This project used a step-by-step approach to solve a classification problem by utilizing machine learning techniques. The Random Forest model was found to be the most suitable solution due to its consistent performance across essential parameters, computational efficiency, robustness against overfitting, and interpretability.  Hyperparameter tuning further proved the stability and adaptability of Random Forest, as the original parameter set was already close to optimal for this dataset. This underscores the value of Random Forest as a reliable choice for classification tasks, particularly in datasets with diverse feature distributions and moderate sizes.

Unanswered Questions and Future Work
While this work successfully identified an optimal model for the dataset, how well the chosen model generalizes to new, unseen datasets in the same domain remains an open question. Moreover, the impact of alternative imputation methods for missing data, such as KNN imputation or advanced deep learning-based approaches, was not explored and could further improve model performance. For future work, exploring advanced feature engineering, combining Gradient Boosting and Random Forest, and testing on similar datasets could enhance generalizability. Incorporating domain-specific knowledge into preprocessing may further improve model robustness and precision.

Thanks by all the team!