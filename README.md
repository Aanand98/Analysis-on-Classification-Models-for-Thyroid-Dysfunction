# Analysis-on-Classification-Models-for-Thyroid-Dysfunction

Team Members:
  1. Mrudula Krishna Prasad https://github.com/Mrudula916
  2. Meghana Ramesh https://github.com/MeghanaYRamesh
  3. Aanand Dhandapani(me)

# Introduction:
  Thyroid is a gland present near the windpipe (trachea). It releases hormones such as T3 and
  T4, which control multiple vital functions of the body, such as metabolism, body temperature.
  The dysfunction of the thyroid gland and improper release of the T3 and T4 hormones causes
  thyroid disease. The pituitary gland handles this malfunction by releasing the Thyroid
  Stimulating Hormone (TSH).There are two types of thyroid conditions. Hypothyroidism is a
  condition wherein TSH is released in insufficient amounts by the thyroid gland.
  Hyperthyroidism is a condition wherein TSH is produced in excess amounts by the thyroid
  gland.

  Globally, thyroid dysfunction is one of the most common diseases in people. 5 in every 100
  people are estimated to have it. This disease cannot be cured but can be treated before it
  becomes severe. If left untreated or uncontrolled, it eventually causes goitre in patients. Thus
  the aim of this project is to classify if a patient is suffering from thyroid or not, and if they do
  test positive, then ultimately predict type.

  To begin with, training and testing will be performed using 5 different classification models.
  These models will be compared for best results based on the metrics like accuracy, precision
  and recall.
    
# Dataset Description:

Link: https://www.kaggle.com/rithikkotha/thyroid-dataset/metadata
    
  The dataset has been taken from Kaggle. It has 30 attributes/features with more than 3000
  observations. There are both categorical and continuous attributes. Categorical features
  include pregnant, thyroid surgery, I131 Treatment etc. Continuous attributes include T3, T4,
  TSH, Age etc. Since the dataset deals with classification problems, the attribute binaryClass
  describes whether a person is testing positive or negative.

  Initial Data Analysis:
    Initial data analysis includes data cleaning, pre-processing the data and feature selection.
    Data pre-processing includes tidying the dataset. Tidy dataset has the following
    characteristics:
        ● Each Variable must have its own column
        ● Each observation must have its own row
        ● Each value must have its own cell
  Pre-processing also includes removing observations that have null values, label encoding the
  categorical features.
  Feature selection includes visualisation of the features using techniques such as heat maps, to
  identify the correlation between the features and their contribution towards the classification.
  The total training data samples will be 80% of the total data, and the remaining 20% will be
  used for testing.
# Methodology:
  The following machine learning algorithms will be compared on the thyroid dataset and the
  model with the best result will be chosen for the classification.
  
  1. Logistic regression: This algorithm takes the likelihood of the features such as
  pregnancy, T3 levels, T4 levels, etc and classifies if the patient is suffering from a
  thyroid dysfunction or not.

  2. Decision tree: This algorithm is based on finding the consequences of a decision.
  Thus, it will choose the maximum number of features that can eventually lead to the
  most accurate classification of the disease, unlike other algorithms which are biased
  towards certain features.

  3. Random forest: This algorithm handles missing values in the data used for
  classification. Due to this, the accuracy over a large dataset can be maintained. For
  instance, in the thyroid dataset, the missing values of the T3, T4 levels can be handled
  easily for a better and accurate classification.

  4. Naive Bayes Classification: This model works well when there is less training data,
  and independence between the features. For instance, age or sex might be independent
  features.

  5. Support Vector Machine (SVM): Since the SVM classifier defines a soft margin for
  the hyperplane, even if the prediction does not attribute to a particular class, the soft
  margin hyperplane will handle this situation efficiently.

