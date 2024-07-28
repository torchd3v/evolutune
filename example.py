from evolutune.tuner import GeneticTuner
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

hyperparameter_space = {
    'criterion': ['gini', 'entropy'],  # Splitting criterion
    'splitter': ['best', 'random'],  # Strategy to choose the split at each node
    'max_depth': [None, 5, 10, 15, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider for the best split
    'max_leaf_nodes': [None, 10, 20, 30],  # Grow a tree with max_leaf_nodes in best-first fashion
    'min_impurity_decrease': [0.0, 0.1, 0.2], # A node will be split if this split induces a decrease of the impurity greater than or equal to this value
    'ccp_alpha': [0.0, 0.1, 0.2]  # Complexity parameter used for Minimal Cost-Complexity Pruning
}

# Load Titanic dataset from URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url)

# Drop unnecessary columns (e.g., Name, Ticket, Cabin) and handle missing values
titanic_df = titanic_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
titanic_df = titanic_df.dropna()

# Convert categorical variables to numerical using one-hot encoding
titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], drop_first=True)

# Define features (X) and target variable (y)
X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Initialize the Cross Validator
skf = StratifiedKFold(n_splits=3)

# Initialize the Genetic Tuner
tuner = GeneticTuner(dt_model,
                     param_grid=hyperparameter_space,
                     scoring="accuracy",
                     n_jobs=-1,
                     cv=skf)

# Fitting the tuner instance for parameters
tuner.fit(train_set=[X_train, y_train],
          eval_set=[X_test, y_test],
          direction="maximize")

print(tuner.best_params_)
print(tuner.best_score_)
