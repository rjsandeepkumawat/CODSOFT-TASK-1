# README: Titanic Survival Prediction

## Author: Sandeep Kumawat

## Batch: April 2024 (A48)

## Domain: Data Science

## Aim

The objective of this endeavor is to construct a model capable of forecasting whether a Titanic passenger survived or perished, leveraging specified features.

## Dataset

The data utilized in this project is extracted from a CSV file named "archive.zip". It comprises details concerning Titanic passengers, encompassing their survival status, class (Pclass), gender (Gender), and age (Age).
## Libraries Used
- pandas
- seaborn
- numpy
- matplotlib.pyplot
- sklearn.linear_model.LogisticRegression
- sklearn.model_selection.train_test_split
- sklearn.preprocessing.LabelEncoder

## Data Exploration and Preprocessing

1. The dataset was loaded into a pandas DataFrame, and its shape along with a preview of the first 10 rows were displayed using `df.shape` and `df.head(10)` respectively.

2. Descriptive statistics for the numerical columns were generated using `df.describe()` to provide an overview of the data, including any missing values.

3. Visualization of the count of passengers who survived versus those who did not was achieved through `sns.countplot(x=df['Survived'])`.

4. Further visualization was conducted to examine the count of survivals concerning the passenger class (Pclass) using `sns.countplot(x=df['Survived'], hue=df['Pclass'])`.

5. A similar visualization approach was employed to explore the count of survivals concerning gender, utilizing `sns.countplot(x=df['Sex'], hue=df['Survived'])`.

6. To ascertain the survival rate by gender, calculations were performed and presented via `df.groupby('Sex')[['Survived']].mean()`.

7. The 'Sex' column was transformed from categorical to numerical values using LabelEncoder from `sklearn.preprocessing`.

8. Following the encoding of the 'Sex' column, non-essential columns like 'Age' were removed from the DataFrame.

## Model Training

1. The feature matrix `X` and target vector `Y` were created using relevant columns from the DataFrame.
2. The dataset was split into training and testing sets using `train_test_split` from `sklearn.model_selection`.
3. A logistic regression model was initialized and trained on the training data using `LogisticRegression` from `sklearn.linear_model`.

## Model Prediction

1. The model was used to predict the survival status of passengers in the test set.
2. The predicted results were printed using `log.predict(X_test)`.
3. The actual target values in the test set were printed using `Y_test`.
4. A sample prediction was made using `log.predict([[2, 1]])` with Pclass=2 and Sex=Male (1).
