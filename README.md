# Social Media Wellbeing Data

## Project Overview
This project analyzes the age distribution of users from a social media dataset. The key steps involve cleaning the age data, categorizing users into specific age groups, visualizing the distribution of users across these groups using a bar charts, and prediction models.

## Data Source
The dataset captures information on social media usage and the dominant emotional state of users based on their activities. The dataset is ideal for exploring the relationship between social media usage patterns and emotional well-being.

Dataset: (https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being/data)

Year: 2024

Number of Respondents: 1000

## Preliminary Questions
1. Does frequency of use matter? How does engagement with the platform matter?
2. Do likes make you happy?
3. Does gender matter?
4. Does age matter?
5. Does platform matter?

## Script Breakdown & Predictve Models

### Script Breakdown - Importing, Cleaning, Encoding, and Scaling dataset
This script demonstrates a step-by-step process of importing, cleaning, encoding, and scaling a dataset, followed by splitting the data into training and testing sets. The processed data will later be used for training machine learning models to predict the Dominant_Emotion in the dataset. The script employs multiple machine learning tools, including Logistic Regression, Random Forest, Decision Trees, K-Nearest Neighbors, and SVM. 

#### Key Steps
1. Importing Required Libraries: The script imports several libraries needed for data manipulation, model training, and evaluation. This includes pandas for data handling, scikit-learn for model preparation and training, and matplotlib for visualizations.
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  from sklearn import preprocessing
  from sklearn.preprocessing import OneHotEncoder, StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn import tree
  import matplotlib.pyplot as plt
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.svm import SVC
  from sklearn.pipeline import make_pipeline
2. Data Import and Initial Exploration: The wellbeing data is imported from a CSV file named 'train_update.csv', and the first few records of the dataset are viewed to understand its structure.
3. Data Cleaning
  * Irrelevant Column Removal: The User_ID column is removed as it is not relevant to the analysis.
  * Handling Missing Values: Any rows with missing values are dropped.
  * Column Type Changes: The Age column is converted to integer type for consistent analysis.
  * The cleaned data is saved into a new CSV file called CleanWellbeingData.csv for later use.
4. Feature Selection: The Daily_Usage_Time (minutes) column is dropped from the dataset based on its feature importance analysis that is done later in the script.
5. Creating Features (X) and Target (y): The script defines the features (X) by dropping the Dominant_Emotion column from the dataset. The Dominant_Emotion column becomes the target variable (y).
6. Splitting the Data: The data is split into training and testing sets using train_test_split. The split is done with a random_state of 13 to ensure reproducibility.
7. Encoding Non-Numeric Data: Non-numeric categorical variables are encoded using OneHotEncoder. This transformation ensures that machine learning algorithms can process the data correctly.
  * handle_unknown='ignore' ensures that the model can handle categories in the test set that were not present in the training set.
  * drop='first' is used to avoid the dummy variable trap by dropping one category from each feature.
8. Scaling the Data: Scaling ensures that all features contribute equally to the model's training process. The StandardScaler is used to standardize the features by removing the mean and scaling to unit variance.

#### Output
After executing the script: 
1. A cleaned dataset (CleanWellbeingData.csv) saved.
2. The X_train, X_test, y_train, and y_test variables will be prepared and scaled, ready for model training.
3. With the data now preprocessed, you can proceed to train various machine learning models.

## Predictive Models

### Logistics Regression
This script uses a logistic regression model to make predictions on test data and evaluates the performance of the model by calculating the accuracy score. The logistic regression model is trained on encoded training data, and the predictions are generated using the test data.

#### Key Componenta
1. Logistic Regression Model Initialization: The logistic regression model is created using the LogisticRegression() class from sklearn.
2. Model Training: The model is trained (or fit) using the encoded training data X_train_encoded and corresponding labels y_train. This allows the model to learn patterns from the training dataset.
3. Make Predictions: After training the model, it makes predictions using the encoded test dataset X_test_encoded. These predictions are saved in the testing_predictions variable.
4. Evaluate the Model: The accuracy of the logistic regression model is evaluated by comparing the predicted labels (testing_predictions) with the true labels (y_test). The accuracy_score function from sklearn.metrics is used to calculate and return the accuracy score.

#### Output

1. Predictions: The script will output the model's predictions for the test dataset (testing_predictions).
2. Accuracy Score: The accuracy score is a metric that represents the percentage of correct predictions made by the model out of the total predictions. It is a value between 0 and 1, with 1 being the highest accuracy (perfect prediction).

### Random Forest
This script demonstrates how to train a Random Forest Classifier, make predictions on test data, calculate the model's accuracy, and identify the most important features contributing to the model.

#### Key Componenta
1. Train Random Forest Classifier: The Random Forest Classifier is initialized and trained using the scaled training data X_train_scaled and the corresponding labels y_train.
  * random_state=42: Ensures reproducibility by setting a seed for random number generation.
  * n_estimators=75: Specifies the number of decision trees in the forest (75 in this case).
2. Make Predictions: After training the model, predictions are generated on the scaled test data X_test_scaled.
3. Evaluate the Model: The performance of the model is evaluated by comparing the predicted labels (predictions) with the true labels (y_test). The accuracy_score function is used to calculate the accuracy of the model.
4. Feature Importance: Random Forest provides a built-in method to evaluate the importance of each feature used in the model. The feature_importances_ attribute is accessed to get the importance values for each feature. The most important features are then sorted in descending order.

#### Output
1. Predictions: The script outputs the predicted values (predictions) for the test dataset.
2. Accuracy Score: The accuracy of the model is calculated and displayed as a value between 0 and 1, with 1 representing perfect accuracy.

### Decision Tree
This script implements a Decision Tree Classifier for predicting labels from a dataset. It trains the model on the scaled training data, makes predictions on the test data, and evaluates the accuracy of the model using the accuracy score metric.

#### Key Componenta
1. Create Decision Tree Classifier: A DecisionTreeClassifier object is created from the tree module of scikit-learn. The model is trained using the scaled training data (X_train_scaled) and corresponding labels (y_train).
2. Make Predictions: After the model is trained, it is used to predict the labels for the test dataset (X_test_scaled). The predicted values are stored in tree_predictions.
3. Evaluate Model Accuracy: The accuracy of the Decision Tree model is calculated by comparing the predicted values (tree_predictions) with the actual labels (y_test). The accuracy_score function from sklearn.metrics is used for this purpose.
4. Accuracy Score: The accuracy score indicates the percentage of correct predictions out of the total predictions, represented as a value between 0 and 1, with 1 being the highest accuracy (perfect prediction).

#### Output
1. Predictions: The predicted values for the test dataset (tree_predictions) can be reviewed or printed.
2. Accuracy Score: The accuracy score shows how well the Decision Tree model performed on the test dataset.

### KNN
This script implements a K-Nearest Neighbors (KNN) model for classification. It trains the KNN model using different values of k (the number of neighbors), evaluates its performance on both training and test data, and visualizes the accuracy scores for different values of k. The best k value is selected, and the model is refitted with this optimal value.

#### Key Componenta
1. K-Nearest Neighbors (KNN) Model Evaluation: The script evaluates the KNN model using different values of k (the number of nearest neighbors) ranging from 1 to 19, with a step size of 2. For each value of k, it calculates the accuracy scores for both the training and test datasets and stores them in lists (train_scores and test_scores).
  * n_neighbors: Specifies the number of neighbors to consider for each prediction.
  * Accuracy scores for training and testing data are printed for each value of k.
2. Plotting the Accuracy Scores: The script plots the accuracy scores for both training and test data against the different values of k. This helps to visualize the performance of the model as k changes, and allows for easy identification of overfitting or underfitting.
  * train_scores: A list of training accuracy scores for each value of k.
  * test_scores: A list of test accuracy scores for each value of k.
  The plot shows how model performance changes as the number of neighbors (k) increases.
3. Selecting the Best k: Based on the results from the previous step, the script selects the best value for k (in this case, k=5) and refits the KNN classifier using that value.
4. Evaluating the Final Model: After refitting the KNN classifier using the chosen value of k, the script evaluates the accuracy of the model on the test data.

#### Output
1. Train/Test Scores for Different k Values: For each k (from 1 to 19, in steps of 2), the training and test accuracy scores are printed.
2. Accuracy Plot: The plot shows the accuracy scores for both the training and testing datasets across different values of k.
3. Final Test Accuracy: The final test accuracy score for the selected k value is printed.

### SVM
This script implements a Support Vector Machine (SVM) classifier using a pipeline that includes data scaling and model training. The model is trained on scaled data, predictions are made on test data, and the model's performance is evaluated using the accuracy score metric.

#### Key Componenta
1. Pipeline Creation with Data Scaling and SVM Training: A pipeline is created using StandardScaler for feature scaling and SVC for the Support Vector Machine classifier. The SVM is set with a linear kernel for classification tasks.
  * StandardScaler(): This scales the features by removing the mean and scaling to unit variance. This step is essential as SVM is sensitive to the scale of the data.
  * SVC(kernel='linear'): The SVM classifier with a linear kernel is used. This kernel is suited for linearly separable data.
  * random_state=42: Sets a fixed random state to ensure reproducibility of results.
2. Training the SVM Model: The pipeline is fitted on the scaled training data (X_train_scaled and y_train), effectively training the SVM model.
3. Making Predictions: The trained SVM model is used to make predictions on the test data (X_test_scaled). These predictions can be compared to the actual labels (y_test).
4. Evaluating the Model: The performance of the SVM model is evaluated using the accuracy score metric, which compares the predicted values (svm_predictions) with the actual test labels (y_test).
  * accuracy_score(): This function calculates the proportion of correct predictions out of the total number of predictions, returning a score between 0 and 1, where 1 represents perfect accuracy.
5. The SVM (Support Vector Machine) is a robust and powerful classifier that works by finding the optimal hyperplane that separates data points of different classes in the feature space. The linear kernel assumes that the data is linearly separable, making it efficient for binary or multi-class classification tasks.

#### Output
1. Predictions: The predicted values for the test dataset (svm_predictions) can be reviewed or printed.
2. Accuracy Score: The accuracy score shows the percentage of correct predictions.

## Script Breakdown for Graphs
### Age Demographic Breakdown
This Python script is designed to analyze and visualize the distribution of age demographics in the dataset. It reads the data from a CSV file, cleans the age column, categorizes users into age groups, and visualizes the age distribution using a bar chart. This script is useful for understanding the demographic breakdown of users based on their age.

1. Importing Libraries: The script imports the necessary libraries for data manipulation and visualization: 
import pandas as pd
import matplotlib.pyplot as plt
2. Loading the Dataset: The dataset containing information about users, including their age, is loaded from a CSV file. You need to ensure that the data_path points to the correct file location.
3. Cleaning the Age Column: The script converts the 'Age' column to numeric values, coercing any errors (e.g., non-numeric values) to NaN. Afterward, any rows with missing or invalid age values are dropped.
4. Creating Age Groups: To simplify the analysis, users are grouped into tighter age ranges. The pd.cut() function is used to categorize users into predefined age groups (ex., 19-21, 22-24, etc.).
5. Counting Users in Each Age Group: The script counts how many users fall into each age group using value_counts(), and the results are printed to the console.
6. Visualizing Age Distribution: A bar chart is generated to visualize the distribution of users across different age groups. The chart includes:
* A title: 'Age Demographic Breakdown'
* Labeled axes (age group and number of users)
* Color customization (skyblue)
* Gridlines for better readability
* Rotated x-axis labels for clarity
7. Output: Printed Console Output: A table showing the number of users in each age group. Visualization: A bar chart displaying the age demographic breakdown.

### Gender Breakdown
This Python script is designed to analyze the usage of various social media platforms by gender in the dataset. The script reads the data from a CSV file, processes it to count the number of users per gender on each platform, and generates a bar chart to visualize the platform gednder usage distribution.

1. Importing Necessary Libraries: The script uses pandas for data handling and matplotlib for plotting charts.
2. Loading the Dataset: The dataset containing information about social media usage is loaded from a CSV file located at '/Users/rygofoss/Desktop/AI_SocialMediaEmotion Project2/train_update.csv'. This file path should be updated based on your local or cloud environment. After loading the dataset, the first few rows are displayed to verify that the data has been loaded correctly.
3. Counting the Number of Users by Social Media Platform: The script counts how many users are on each social media platform by using the value_counts() function on the Platform column. The breakdown of users by platform is printed to the console.
4. Visualizing Platform Usage: The script generates a bar chart to visually represent the number of users for each social media platform. The bar chart is customized with:
* A title: 'Breakdown of Social Media Platforms Used'
* Labeled axes for clarity
* A color scheme (skyblue)
* Rotated platform names for better readability
* Gridlines for improved visual interpretation
5. Output: Printed Console Output: A table showing the number of users for each social media platform. Visualization: A bar chart displaying the breakdown of social media platforms used by the users in the dataset.

### Social Media Platform Breakdown
This Python script is used to analyze and visualize the distribution of users across different social media platforms in the dataset. It loads the dataset, counts the number of users per platform, and generates a bar chart to represent the data visually.

1. Importing Libraries: The script imports the necessary libraries to handle the dataset (pandas) and generate plots (matplotlib).
2. Loading the Dataset: The dataset is loaded from a CSV file that contains information about social media platforms and their users. The file path must be updated in the data_path variable to match your local environment.
3. Counting Users by Platform: The script counts the number of users for each social media platform using the value_counts() method. This method calculates the frequency of each unique value in the 'Platform' column.
4. Visualizing the Breakdown: A bar chart is created to visualize the number of users for each platform. The following elements are customized:
* A title: 'Breakdown of Social Media Platforms Used'
* Labeled axes (Platform and Number of Users)
* Custom colors (skyblue)
* Rotated x-axis labels to improve readability
* Gridlines to enhance the chart’s clarity
5. Output: Console Output: A table displaying the number of users per platform. Visualization: A bar chart that visually represents the platform breakdown.

### Social Media Breakdown by Age Group
This python script is used to to analyze and visualizes the distribution of users' ages and the social media platforms they use. The primary goal is to categorize users into age groups and visualize the distribution of users across different social media platforms.

1. Import Necessary Libraries: The code imports the pandas library for data manipulation and the matplotlib.pyplot library for data visualization.
2. Load the Dataset: The dataset is loaded into a pandas DataFrame from the specified file path.
3. Display Dataset Structure: The first few rows of the DataFrame are printed to understand the dataset's structure.
4. Data Cleaning: The Age column is converted to a numeric type, coercing any errors to NaN. Rows with missing values in the Age column are dropped.
5. Age Grouping: The ages are categorized into tighter ranges defined by age_bins. Corresponding labels are assigned to each age range.
6. Counting Users: The code groups the DataFrame by Platform and Age_Group and counts the number of users in each category.
7. Display Counts: The counts of users by platform and age group are printed.
8. Data Visualization: A stacked bar chart is created to visualize the breakdown of users across different social media platforms by age groups.

### Social Media Breakdown by Gender
This python script is used to to analyze and visualizes the distribution of users' genders and the social media platforms they use. The primary goal is to categorize users by gender and visualize the distribution across different social media platforms.

1. Import Necessary Libraries: The code imports the pandas library for data manipulation and the matplotlib.pyplot library for data visualization.
2. Load the Dataset: The dataset is loaded into a pandas DataFrame from the specified file path.
3. Display Dataset Structure: The first few rows of the DataFrame are printed to ensure the dataset is loaded correctly.
4. Data Filtering: The code selects relevant columns to focus on the gender breakdown.
5. Counting Users: The DataFrame is grouped by Platform and Gender, and the number of users in each category is counted. The counts are unstacked to create a matrix format with platforms as rows and genders as columns, filling missing values with zeros.
6. Display Counts: The counts of users by platform and gender are printed.
7. Data Visualization: A stacked bar chart is created to visualize the breakdown of users across different social media platforms by gender.

#### Dominant Emotion of each Social Media Platform
This python script is used to to analyze and visualizes the distribution of users' dominant emotions and the social media platforms they use. The primary goal is to categorize users by their dominant emotions and visualize the distribution across different social media platforms.

1. Import Necessary Libraries: The code imports the pandas library for data manipulation and the matplotlib.pyplot library for data visualization.
2. Load the Dataset: The dataset is loaded into a pandas DataFrame from the specified file path.
3. Display Dataset Structure: The first few rows of the DataFrame are printed to ensure the dataset is loaded correctly.
4. Counting Users: The DataFrame is grouped by Platform and Dominant_Emotion, and the number of users in each category is counted. The counts are unstacked to create a matrix format with platforms as rows and emotions as columns, filling missing values with zeros.
5. Display Counts: The counts of users by platform and dominant emotion are printed.
6. Data Visualization: A stacked bar chart is created to visualize the emotional impact of social media usage across different platforms.

## References
Dataset: (https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being/data)

Chart Examples: https://www.kaggle.com/code/megha1703/emotion-prediction-by-social-media-behavior

SVM Example: https://scikit-learn.org/stable/modules/svm.html

## License

MIT License

## Contact

For any questions or suggestions, feel free to contact:
Shyla Tatum, Rygo Foss, Lou Danjar, Natasia McLean

GitHub: https://github.com/ShylaTatum/Project-2.git

