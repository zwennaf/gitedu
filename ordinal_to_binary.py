# Based on A Simple Approach to Ordinal Classification by Eibe Frank and Mark Hall http://old-www.cms.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_is_fitted, clone


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf=LogisticRegression()):
        self.clf = clf
        self.clfs = []

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        self.clfs = []
        for i in range(len(self.unique_class) - 1):
            # Binary classification for each threshold
            binary_y = (y > self.unique_class[i]).astype(int)
            clf = clone(self.clf)
            clf.fit(X, binary_y)
            self.clfs.append(clf)
        return self

    def predict_proba(self, X):
        # Check if classifier has been fitted
        check_is_fitted(self)
        # Validate input array
        X = check_array(X, accept_sparse=True)
        # Initialize probability matrix
        probas = np.zeros((X.shape[0], len(self.unique_class)))
        # Get probabilities from each binary classifier
        for idx, clf in enumerate(self.clfs):
            probas[:, idx + 1] = clf.predict_proba(X)[:, 1]
        # Convert binary probabilities to cumulative ordinal probabilities
        probas[:, 0] = 1 - probas[:, 1]
        for idx in range(1, len(self.unique_class) - 1):
            probas[:, idx] = probas[:, idx] - probas[:, idx + 1]
        return probas

    def predict(self, X):
        # Predict the class with the highest probability
        return self.unique_class[np.argmax(self.predict_proba(X), axis=1)]


# Personal and absolute path to access the datafiles.
dir_path = "C:/Users/fenna/PycharmProjects/fennaz/Data/"
absolute_path = dir_path + "RawData/"

df = pd.read_csv(absolute_path + "RF_plus_plus.csv")
y = df[:, -1]
X = df.drop(columns="LS_SCORE_")
y = np.sort(y)  # Ensure y is ordered

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the ordinal classifier
ordinal_clf = OrdinalClassifier()
ordinal_clf.fit(X_train, y_train)

# Predict probabilities and classes
probas = ordinal_clf.predict_proba(X_test)
predictions = ordinal_clf.predict(X_test)

# Print the predicted probabilities for the first instance
print("Predicted probabilities:", probas[0])
print("Predicted class:", predictions[0])


""" when comparing techniques in report, possible evaluations for when dealing with ordinal data, there are several methods you can consider for analysis:

1. Analyze Ordinal Variables as Nominal:
   - Ordinal variables are fundamentally categorical, but sometimes you can treat them as nominal by ignoring the order of their categories. This approach treats all categories equally and doesn't consider the rank order².
   - For example, if you have an ordinal variable like "education level" (e.g., primary school, high school, bachelor's degree, master's degree, PhD), you can treat it as nominal and use appropriate categorical analysis techniques.

2. Analyze Ordinal Variables as Numeric:
   - While ordinal data lacks equal intervals between categories, you can sometimes treat it as numeric for simplicity. This approach assumes that the differences between adjacent categories are equal, even though this might not be strictly true.
   - Be cautious when using this method, as it oversimplifies the underlying nature of ordinal data.

3. Non-Parametric Tests:
   - Non-parametric tests are suitable for ordinal data because they don't assume a specific distribution. Examples include the Wilcoxon rank-sum test, Kruskal-Wallis test, and Spearman's rank correlation.
   - These tests focus on the order of observations rather than their exact values.

4. Ordinal Logistic & Probit Regression:
   - These regression models are specifically designed for ordinal response variables. They account for the ordered nature of the categories and estimate the effects of predictors on the odds of moving from one category to another.
   - Ordinal logistic regression assumes proportional odds, while ordinal probit regression uses a different link function.

5. Rank Transformations:
   - Rank transformations involve converting ordinal data into ranks (e.g., 1st, 2nd, 3rd) and then analyzing the transformed ranks as if they were continuous.
   - This method can be useful when you want to apply parametric statistical techniques (e.g., ANOVA) to ordinal data.

Sources
(1) Five Ways to Analyze Ordinal Variables (Some Better than Others). https://www.theanalysisfactor.com/ways-analyze-ordinal-variables/.
(2) Ordinal Data - Definition, Uses, and How To Analyze. https://corporatefinanceinstitute.com/resources/data-science/ordinal-data/.

Certainly! The techniques mentioned earlier can indeed be used for classification in machine learning. Let's explore how each method can be applied:

1. **Ordinal Variables as Nominal**:
   - While treating ordinal variables as nominal doesn't directly lead to classification, it's a preprocessing step. You can convert ordinal features into dummy variables (one-hot encoding) and then use them as input features for classification algorithms.
   - For example, if you have an ordinal feature like "education level" (with categories: primary school, high school, bachelor's degree, etc.), you can create binary features (e.g., "is_primary_school," "is_high_school," etc.) and use them in classification models.

2. **Ordinal Variables as Numeric**:
   - Treating ordinal variables as numeric allows you to use regression-based or distance-based classifiers.
   - For instance, you can use k-nearest neighbors (KNN) with a distance metric that considers the ordinal nature of the data. However, be cautious about the assumptions you're making.

3. **Non-Parametric Tests**:
   - Non-parametric tests are not classification algorithms per se, but they help analyze relationships between variables.
   - For instance, if you want to compare groups based on an ordinal variable (e.g., life satisfaction), you can use the Kruskal-Wallis test to determine if there are significant differences among the groups.

4. **Ordinal Logistic & Probit Regression**:
   - These models are specifically designed for ordinal response variables.
   - In the context of classification, you can use ordinal logistic regression to predict the ordinal category (e.g., low, medium, high) based on predictor variables.

5. **Rank Transformations**:
   - While not a direct classification method, rank transformations can be useful for feature engineering.
   - After rank-transforming ordinal features, you can use any classification algorithm that works with continuous data.

Remember that the choice of method depends on the problem, data, and context. If you have labeled data with ordinal outcomes (e.g., SWLS scores), you can train and evaluate classification models using these techniques. Keep in mind that interpreting the results in terms of life satisfaction categories is essential for practical applications. If you need further assistance or have specific data, feel free to ask! 😊"""
