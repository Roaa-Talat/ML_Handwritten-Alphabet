# %% [markdown]
# ### Section IS S1 & S2  
# 
# #### Team Members  
# 
# | Name                                | ID       |
# |-------------------------------------|----------|
# | Salma Mamdoh Sabry                  | 20210162 |
# | Roaa Talat Mohamed                  | 20210138 |
# | Youssef Ehab Mohamed                | 20210466 |
# | Zeyad Ehab Maamoun                  | 20211043 |
# | Youssef Mohamed Salah Eldin Anwar   | 20210483 |
# 

# %% [markdown]
# # About the Dataset: A-Z Handwritten Alphabets
# 
# This dataset provides handwritten representations of English alphabets (A-Z), captured in grayscale images stored in a structured `.csv` format. It is designed to support machine learning projects, particularly in training models for handwritten character recognition. The dataset includes over **370,000 samples** of alphabets.
# 
# ---
# 
# ## Key Information
# 
# ### Dataset Overview:
# - **Size**: 370,000+ images of handwritten English alphabets.
# - **Format**: Each image is represented as a row in a `.csv` file, with pixel values and corresponding labels.
# - **Pixel Data**: Grayscale intensity values ranging from 0 (black) to 255 (white).
# - **Labels**: Indicate the corresponding alphabet (A-Z).
# 
# ### Image Details:
# - **Resolution**: Each image is resized to `28x28` pixels.
# - **Center-Fitting**: Alphabets are centered in a `20x20` pixel bounding box.
# - **Noisy Samples**: Some noisy images may be present in the dataset.
# 
# ---
# 
# ## Column Descriptions
# 
# 1. **Pixel Values**: 784 columns (28x28 pixels) representing the grayscale intensity of each pixel.
# 2. **Label**: Indicates the alphabet associated with the image, represented numerically (e.g., 0 = A, 1 = B, ..., 25 = Z).

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
plt.rcParams["figure.figsize"] = (10, 6)  
import warnings
warnings.filterwarnings('ignore') 

# %% [markdown]
# ## Preprocessing

# %%
df = pd.read_csv('/kaggle/input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv')
df.head()

# %%
df.tail()

# %%
print(df.head(1))

# %%
print("Dataset Information: \n")
df.info()

# %%
print("Dataset Shape: \n")
df.shape

# %%
# Rename columns (first column is the label, the rest are features)
df.rename(columns={'0':'label'}, inplace=True)
print(df.columns)

# %%
df.head()

# %% [markdown]
# ### Identify the number of unique classes

# %%
# the first column contains the labels, let's check for unique values
unique_classes = df['label'].nunique()

print(f"Number of unique classes: {unique_classes}")

# %% [markdown]
# ### show their distribution.

# %%
# Show the distribution of labels (count of each class)
class_counts = df['label'].value_counts().sort_index()

# Print the sorted distribution
print("\nClass distribution (sorted):")
print(class_counts)

# %%
# Plot class distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title("Class Distribution")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(range(26), [chr(i + 65) for i in range(26)])  # Convert numeric labels to alphabets
plt.show()

# %%
# Visualize samples for each class
plt.figure(figsize=(15, 10))
for label in range(26):
    sample = df[df['label'] == label].iloc[0, 1:].values.reshape(28, 28)
    plt.subplot(4, 7, label + 1)
    plt.imshow(sample, cmap='viridis')
    plt.axis('off')
    plt.title(chr(label + 65))
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Spliting Data Into Train and Test

# %%
# Split features and labels
X = df.iloc[:, 1:]
y = df['label']

# %%
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# %%
df.describe()

# %%
# Map labels to alphabets
y_mapped = y.map(lambda x: chr(x + 65))

# %%
# Show the distribution of labels (count of each class)
class_counts = df['label'].value_counts().sort_index()

# Print the sorted distribution with corresponding characters
print("\nClass distribution (sorted with corresponding characters):")
for label, count in class_counts.items():
    print(f"{chr(label + 65)}: {count}")

# %% [markdown]
# ### Normalize each image

# %%
X_train = X_train.astype('float32') / 255.0  # Normalize to 0-1 range
X_test = X_test.astype('float32') / 255.0

# %%
# 8. Normalization Validation: Verify that the images are properly normalized
print(f"Minimum value in normalized train data: {np.min(X_train)}")
print(f"Maximum value in normalized train data: {np.max(X_train)}")


# %%
# 8. Normalization Validation: Verify that the images are properly normalized
print(f"Minimum value in normalized test data: {np.min(X_test)}")
print(f"Maximum value in normalized test data: {np.max(X_test)}")

# %%
X_train = np.array(X_train)
X_test = np.array(X_test)

# 5. Reshape the flattened vectors back to 28x28 images 
X_train_reshaped = X_train.reshape(-1, 28, 28)  # Reshape to original image dimensions
X_test_reshaped = X_test.reshape(-1, 28, 28)

# %%
# Function to plot images with their labels
def plot_images(images, labels, num_images=10):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)  # Create a grid of 2 rows and 5 columns
        plt.imshow(images[i], cmap='gray')  # Display image
        plt.title(f"Label: {labels[i]}")  # Display corresponding label
        plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()


labels_alphabet = [chr(label + 65) for label in y_test]  # Convert encoded labels to A-Z letters

# Plot the first 10 images with their labels
plot_images(X_test_reshaped, labels_alphabet, num_images=10)


# %% [markdown]
# ### Experiment 1 -- SVM models 

# %%
# Installing Kaggle API
!pip install --upgrade --quiet kaggle


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score
)

# Settin up Kaggle API credentials
from google.colab import files

print("Please upload your kaggle.json file.")
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d sachinpatel21/az-handwritten-alphabets-in-csv-format

!unzip -o az-handwritten-alphabets-in-csv-format.zip

# Verify the dataset file size
dataset_path = 'A_Z Handwritten Data.csv'
file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
print(f'Dataset file size: {file_size_mb:.2f} MB')

print("Loading the dataset...")
data = pd.read_csv(dataset_path, header=None)
print('Dataset shape:', data.shape)

# Assign column names
column_names = ['label'] + [f'pixel_{i}' for i in range(784)]
data.columns = column_names

# Identify unique classes and show their distribution
unique_classes = data['label'].unique()
unique_classes.sort()
print('Unique classes in the dataset:', unique_classes)
print('Number of unique classes:', len(unique_classes))

# Show the distribution of classes
class_distribution = data['label'].value_counts().sort_index()
plt.figure(figsize=(12,6))
sns.barplot(x=class_distribution.index, y=class_distribution.values, palette='viridis')
plt.title('Distribution of Classes in the Dataset')
plt.xlabel('Class Label')
plt.ylabel('Number of Samples')
plt.show()

# Sample 10% of data from each class
portion = 0.10
data_sampled = data.groupby('label', group_keys=False).apply(
    lambda x: x.sample(frac=portion, random_state=42)
).reset_index(drop=True)

# Verify the distribution of classes in the sampled data
sampled_class_distribution = data_sampled['label'].value_counts().sort_index()
print('\nSampled dataset class distribution:')
print(sampled_class_distribution)

plt.figure(figsize=(12,6))
sns.barplot(x=sampled_class_distribution.index, y=sampled_class_distribution.values, palette='coolwarm')
plt.title('Distribution of Classes in the Sampled Dataset')
plt.xlabel('Class Label')
plt.ylabel('Number of Samples')
plt.show()

# Separate features and labels
X = data_sampled.drop('label', axis=1).values
y = data_sampled['label'].values

X_normalized = X / 255.0

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.20, random_state=42, stratify=y
)

# Train SVM models
print('\nTraining SVM with Linear Kernel...')
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)

print('Training SVM with RBF Kernel...')
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)

# Test and evaluate the linear SVM
print('\nEvaluating SVM with Linear Kernel...')
y_pred_linear = svm_linear.predict(X_test)

# Compute Accuracy
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print('Accuracy (Linear Kernel): {:.2f}%'.format(accuracy_linear * 100))

# Compute F1 Score
f1_linear = f1_score(y_test, y_pred_linear, average='weighted')
print('Average F1 Score (Linear Kernel): {:.2f}'.format(f1_linear))

# Classification Report
report_linear = classification_report(y_test, y_pred_linear)
print('\nClassification Report for SVM with Linear Kernel:\n')
print(report_linear)

# Confusion Matrix
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)

# Visualize Confusion Matrix
plt.figure(figsize=(12,10))
sns.heatmap(conf_matrix_linear, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM Linear Kernel')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Test and evaluate the RBF SVM
print('Evaluating SVM with RBF Kernel...')
y_pred_rbf = svm_rbf.predict(X_test)

# Compute Accuracy
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print('Accuracy (RBF Kernel): {:.2f}%'.format(accuracy_rbf * 100))

# Compute F1 Score
f1_rbf = f1_score(y_test, y_pred_rbf, average='weighted')
print('Average F1 Score (RBF Kernel): {:.2f}'.format(f1_rbf))

# Classification Report
report_rbf = classification_report(y_test, y_pred_rbf)
print('\nClassification Report for SVM with RBF Kernel:\n')
print(report_rbf)

# Confusion Matrix
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)

# Visualize Confusion Matrix
plt.figure(figsize=(12,10))
sns.heatmap(conf_matrix_rbf, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - SVM RBF Kernel')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# %% [markdown]
# #### Split Data into Training , Test And Validation , one Hot Encoding For Labels 

# %%
from sklearn.model_selection import train_test_split

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training data shape:", X_train_split.shape)
print("Validation data shape:", X_val.shape)
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# %% [markdown]
# ## **Experiment 2 -- Logistic Regression From Scratch**

# %% [markdown]
# #### Sigmoid Function Explanation
# 
# The sigmoid function maps any input \( z \) to a value between \( 0 \) and \( 1 \). It is used to model the probability that a given input belongs to a particular class. The formula is:
# 
# 
# $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
# 
# 
# Where:
# 
# - \( z \) is a linear combination of inputs and weights:
# $z = X \cdot \theta\$
# - \( e \) is the base of the natural logarithm.

# %%
# 1. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# %% [markdown]
# #### Cost Function 
# 
# The cost function for logistic regression measures how well the model's predictions match the true labels.
# 
# The formula is:
# 
# 
# $$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(x^{(i)})) \right]$$
# 
# where
# 
# - $h_{\theta}(x) = \sigma(\theta^T x)$ is the predicted probability.
# - $ m $ is the number of training examples.
# - $ y^{(i)} $ is the true label (0 or 1) for the \( i \)-th example.
# - $ x^{(i)} $ is the input features for the \( i \)-th example.
# 
# The cost penalizes incorrect predictions, particularly those with high confidence.

# %%
# 2. Cost function for logistic regression
def compute_cost(X, y, theta):
    m = len(y)  
    h = sigmoid(np.dot(X, theta))  
    cost = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    return cost

# %% [markdown]
# #### Gradient Descent
# 
# Gradient descent is an optimization algorithm used to minimize the cost function by iteratively updating the parameters $\theta$.
# 
# At each iteration:
# 
# $$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$
# 
# 
# Where:
# 
# - $\alpha$ is the learning rate, controlling the step size.
# - $\frac{\partial J(\theta)} {\partial \theta_j}$ is the gradient of the cost function with respect to $\theta_j$.
# 
# The gradient is given by:
# 
# 
# $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left[ h_{\theta}(x^{(i)}) - y^{(i)} \right] x_j^{(i)}$$
# 
# 
# The algorithm stops after a fixed number of iterations $\text{num\_iters}$ or when the cost converges.

# %%
# 3. Gradient descent
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)  # Number of training examples
    cost_history = []

    for _ in range(num_iters):
        # Predicted probabilities
        h = sigmoid(np.dot(X, theta))

        # Gradient of the cost function
        gradient = np.dot(X.T, (h - y)) / m

        # Update parameters
        theta -= alpha * gradient

        # Save cost for monitoring
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

# %% [markdown]
# ## Prediction
# 
# For each input \( X \), we compute the probability of belonging to each class using the sigmoid function and the learned parameters $\theta \$. The class with the highest probability is chosen as the prediction:
# 
# $$\text{Prediction} = \arg \max_{c} \sigma(\theta_c^T X)$$
# 
# Where \( c \) is the class index, and $\theta_c$ is the parameter vector for class \( c \).

# %%
# 4. Train one-vs-all classifiers
def train_one_vs_all(X, y, num_classes, alpha=0.01, num_iters=1000):
    m, n = X.shape  # m = examples, n = features
    all_theta = np.zeros((num_classes, n))  # Initialize parameters for all classes

    for c in range(num_classes):
        print(f"Training classifier for class {c}")
        binary_y = (y == c).astype(int)  # Convert to binary classification for class c
        theta = np.zeros(n)  # Initialize theta for this class
        theta, _ = gradient_descent(X, binary_y, theta, alpha, num_iters)
        all_theta[c] = theta
    
    return all_theta

# %%
# Predict using one-vs-all logistic regression
def predict_one_vs_all(X, all_theta):
    probabilities = sigmoid(np.dot(X, all_theta.T))
    return np.argmax(probabilities, axis=1)

# %% [markdown]
# ## Train with Learning Rate 0.01

# %%
# Add bias column to data
X_train_with_bias = np.hstack((np.ones((X_train_split.shape[0], 1)), X_train_split))
X_val_with_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
X_test_with_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Train the model
num_classes = 26  # Number of classes (A-Z)
alpha = 0.1  # Learning rate
num_iters = 2000  # Number of iterations

all_theta = train_one_vs_all(X_train_with_bias, y_train_split, num_classes, alpha, num_iters)

# %%
# Plot Error Curves
train_cost_history = [compute_cost(X_train_with_bias, (y_train_split == c).astype(int), all_theta[c]) for c in range(num_classes)]
val_cost_history = [compute_cost(X_val_with_bias, (y_val == c).astype(int), all_theta[c]) for c in range(num_classes)]

plt.plot(range(num_classes), train_cost_history, label='Training Error')
plt.plot(range(num_classes), val_cost_history, label='Validation Error')
plt.xlabel('Class')
plt.ylabel('Error')
plt.title('Error Curves')
plt.legend()
plt.show()

# %%
from sklearn.metrics import accuracy_score

# Calculate accuracy for each class on training and validation sets
train_accuracy = [accuracy_score((y_train_split == c).astype(int), predict_one_vs_all(X_train_with_bias, all_theta) == c) for c in range(num_classes)]
val_accuracy = [accuracy_score((y_val == c).astype(int), predict_one_vs_all(X_val_with_bias, all_theta) == c) for c in range(num_classes)]

# Plot Accuracy Curves
plt.plot(range(num_classes), train_accuracy, label='Training Accuracy')
plt.plot(range(num_classes), val_accuracy, label='Validation Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.show()

# %%
from sklearn.metrics import classification_report, confusion_matrix

# Print Classification Report
print(classification_report(y_test, test_preds, target_names=[chr(i) for i in range(65, 91)]))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=[chr(i) for i in range(65, 91)], 
            yticklabels=[chr(i) for i in range(65, 91)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# %%
from sklearn.metrics import f1_score

# Average F1 Score
average_f1 = f1_score(y_test, test_preds, average='macro')  # Use 'macro' for unweighted average
print(f"Average F1 Score: {average_f1:.2f}")

# %% [markdown]
# ### Teain with Learning Rate 0.01

# %%
X_train_with_bias = np.hstack((np.ones((X_train_split.shape[0], 1)), X_train_split))
X_val_with_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
X_test_with_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

num_classes = 26  
alpha = 0.01  
num_iters = 1000  

all_theta = train_one_vs_all(X_train_with_bias, y_train_split, num_classes, alpha, num_iters)
train_preds = predict_one_vs_all(X_train_with_bias, all_theta)
val_preds = predict_one_vs_all(X_val_with_bias, all_theta)
train_acc = np.mean(train_preds == y_train_split) * 100
val_acc = np.mean(val_preds == y_val) * 100
print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Validation Accuracy: {val_acc:.2f}%")

# %% [markdown]
# ### Train with Learning Rate 0.5

# %%
X_train_with_bias = np.hstack((np.ones((X_train_split.shape[0], 1)), X_train_split))
X_val_with_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
X_test_with_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

num_classes = 26  
alpha = 0.5  
num_iters = 1000  

all_theta = train_one_vs_all(X_train_with_bias, y_train_split, num_classes, alpha, num_iters)
train_preds = predict_one_vs_all(X_train_with_bias, all_theta)
val_preds = predict_one_vs_all(X_val_with_bias, all_theta)
train_acc = np.mean(train_preds == y_train_split) * 100
val_acc = np.mean(val_preds == y_val) * 100
print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Validation Accuracy: {val_acc:.2f}%")

# %%
train_cost_history = [compute_cost(X_train_with_bias, (y_train_split == c).astype(int), all_theta[c]) for c in range(num_classes)]
val_cost_history = [compute_cost(X_val_with_bias, (y_val == c).astype(int), all_theta[c]) for c in range(num_classes)]

plt.plot(range(num_classes), train_cost_history, label='Training Error')
plt.plot(range(num_classes), val_cost_history, label='Validation Error')
plt.xlabel('Class')
plt.ylabel('Error')
plt.title('Error Curves')
plt.legend()
plt.show()

# %%
from sklearn.metrics import accuracy_score

train_accuracy = [accuracy_score((y_train_split == c).astype(int), predict_one_vs_all(X_train_with_bias, all_theta) == c) for c in range(num_classes)]
val_accuracy = [accuracy_score((y_val == c).astype(int), predict_one_vs_all(X_val_with_bias, all_theta) == c) for c in range(num_classes)]

plt.plot(range(num_classes), train_accuracy, label='Training Accuracy')
plt.plot(range(num_classes), val_accuracy, label='Validation Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.show()


# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print(classification_report(y_test, test_preds, target_names=[chr(i) for i in range(65, 91)]))

conf_matrix = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=[chr(i) for i in range(65, 91)], 
            yticklabels=[chr(i) for i in range(65, 91)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# %%
from sklearn.metrics import f1_score

average_f1 = f1_score(y_test, test_preds, average='macro')  # Use 'macro' for unweighted average
print(f"Average F1 Score: {average_f1:.2f}")

# %% [markdown]
# ## **Experiment 3 -- Neural Networks**

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Convert labels to categorical format for multi-class classification
y_train_categorical = to_categorical(y_train, num_classes=26)  # 26 letters A-Z
y_test_categorical = to_categorical(y_test, num_classes=26)

# Split training into training and validation datasets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train_categorical, test_size=0.2, random_state=42)

# %% [markdown]
# ## Model 1 

# %% [markdown]
# ### A simple neural network with one hidden layer of 128 units and a dropout rate of 0.3 to prevent overfitting. The output layer has 26 neurons (one for each letter).

# %%
def create_model_1(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(26, activation='softmax')  # 26 output classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# %%
def train_and_plot(model, X_train, y_train, X_val, y_val, model_name):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20
    )
    
    # Plot accuracy and loss
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    return model

# %%
# Train Model 1
model_1 = create_model_1(input_shape=X_train_split[0].shape)
model_1 = train_and_plot(model_1, X_train_split, y_train_split, X_val, y_val, model_name="Model 1")

# %% [markdown]
# ### Model 1 - Training Insights
# 
# #### Overview:
# - **Input Layer**: A `Flatten` layer to preprocess input data.
# - **Hidden Layer**: A dense layer with 128 neurons, ReLU activation, and 30% dropout to prevent overfitting.
# - **Output Layer**: A dense layer with 26 neurons (one for each class) using softmax activation for classification probabilities.
# 
# The model was trained using:
# - **Optimizer**: Adam
# - **Loss Function**: Categorical Crossentropy
# - **Evaluation Metric**: Accuracy
# 
# #### Training and Validation Metrics:
# 1. **Accuracy**:
#    - Training accuracy started at **81.97%** by the end of the first epoch and progressively increased to **95.70%** by epoch 20.
#    - Validation accuracy began at **94.63%** in epoch 1 and improved steadily to **97.45%** by epoch 20.
#    - The gap between training and validation accuracy remained small, indicating good generalization.
# 
# 2. **Loss**:
#    - Training loss began at **0.6481** in epoch 1 and decreased to **0.1427** by epoch 20.
#    - Validation loss started at **0.1937** and plateaued at around **0.1044** after several epochs, showing consistent improvements.

# %%
def plot_confusion_matrix(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# %%
from sklearn.metrics import classification_report, confusion_matrix,f1_score
y_test_pred = model_1.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true_classes = np.argmax(y_test_categorical, axis=1)

# Print Classification Report
print(classification_report(y_test_true_classes, y_test_pred_classes, target_names=[chr(i) for i in range(65, 91)]))

# Plot Confusion Matrix
plot_confusion_matrix(y_test_true_classes, y_test_pred_classes, target_names=[chr(i) for i in range(65, 91)])

# Calculate Average F1 Score
average_f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='macro')
print(f"Average F1 Score: {average_f1:.2f}")

# %% [markdown]
# ## Model 2

# %% [markdown]
# ### **Optimized Neural Network Design**
# 
# 1. **Layer Architecture**:
#    - Use multiple hidden layers with fewer neurons per layer.
#    - Reduce the number of trainable parameters to prevent overfitting.
# 
# 2. **Regularization**:
#    - Apply **Dropout** with a higher rate (e.g., 30-50%).
#    - Add **L2 regularization** to the dense layers.
# 
# 3. **Activation Functions**:
#    - Use **ReLU** for hidden layers and **Softmax** for the output layer.
# 
# 4. **Early Stopping**:
#    - Monitor validation loss to avoid training too long and overfitting.
# 
# 5. **Batch Normalization**:
#    - Add after each dense layer to stabilize learning.

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def create_optimized_model():
    model = Sequential([
        Dense(1024, kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),

        Dense(512, kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),

        Dense(256, kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),

        Dense(26, activation='softmax')  # Output for 26 classes
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Callbacks for training
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit model (assuming x_train, y_train, x_val, y_val are defined)
model = create_optimized_model()
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[lr_scheduler, early_stopping]
)

# %%
import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Save Best Model and reuse it 

# %%
model_1.save("First_Try_model.h5")  # Assume Model 2 performed better
First_Try_model = tf.keras.models.load_model("First_Try_model.h5")

# %%
model.save("best_model.h5")  # Assume Model 2 performed better
best_model = tf.keras.models.load_model("best_model.h5")

# %%
y_test_pred = best_model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true_classes = np.argmax(y_test_categorical, axis=1)

# Print Classification Report
print(classification_report(y_test_true_classes, y_test_pred_classes, target_names=[chr(i) for i in range(65, 91)]))

# Plot Confusion Matrix
plot_confusion_matrix(y_test_true_classes, y_test_pred_classes, target_names=[chr(i) for i in range(65, 91)])

# Calculate Average F1 Score
average_f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='macro')
print(f"Average F1 Score: {average_f1:.2f}")

# %% [markdown]
# ### Test best Model with images 

# %%
# Function to load images from the given paths
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to match model input size (28x28 for MNIST-like models)
        images.append(np.array(img))  # Convert to numpy array
    return np.array(images)

# %%
# Plot function for images and predictions
def plot_image(index, predictions_array, true_labels, images):
    plt.imshow(images[index], cmap='gray')
    predicted_label = np.argmax(predictions_array)
    true_label = ord(true_labels[index]) - 65  # Convert letter to index (A=0, B=1, ...)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.title(f"True: {true_labels[index]}\nPred: {chr(predicted_label + 65)}", color=color)
    plt.axis('off')

# %%
def plot_value_array(index, predictions_array, true_labels):
    predicted_label = np.argmax(predictions_array)
    true_label = ord(true_labels[index]) - 65
    plt.bar(range(26), predictions_array, color="#777777")
    plt.xticks(range(26), [chr(i + 65) for i in range(26)], rotation=90)
    plt.yticks([])
    plt.ylim([0, 1])
    plt.bar(predicted_label, predictions_array[predicted_label], color='red')
    plt.bar(true_label, predictions_array[true_label], color='blue')

# %%
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_model.h5')  # Ensure the correct path to the model

# Define the paths to the images to be tested
image_paths = [
    "/kaggle/input/test-images/S_3.png",
    "/kaggle/input/test-images/R_2.png",
    "/kaggle/input/test-images/Y_1.png",
    "/kaggle/input/test-images/Z_2.png"
]

# Load the images
test_images = load_images(image_paths)

# Preprocess the images (normalize and reshape as required by the model)
test_images = test_images.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
test_images_flattened = test_images.reshape(-1, 28 * 28)  # Flatten images for the model

# Get model predictions
predictions = model.predict(test_images_flattened)

# Define the actual labels for the test images
test_labels = ['S', 'R', 'Y', 'Z']  # Corresponding true labels for the images

# Display predictions and confidence levels for each character
for label in ['S', 'R', 'Y', 'Z']:
    label_indices = [i for i, x in enumerate(test_labels) if x == label]
    num_images = len(label_indices)

    # Create subplots for images and prediction value arrays
    fig, axes = plt.subplots(2, num_images, figsize=(12, 6))

    # Ensure axes is always 2D (even when num_images = 1)
    if num_images == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, index in enumerate(label_indices):
        # Plot the test image
        axes[0, i].imshow(test_images[index], cmap='gray')
        predicted_label = np.argmax(predictions[index])
        true_label = ord(test_labels[index]) - 65  # Convert letter to index (A=0, B=1, ...)
        color = 'blue' if predicted_label == true_label else 'red'
        axes[0, i].set_title(f"True: {test_labels[index]}\nPred: {chr(predicted_label + 65)}", color=color)
        axes[0, i].axis('off')

        # Plot the prediction value array
        axes[1, i].bar(range(26), predictions[index], color="#777777")
        axes[1, i].set_xticks(range(26))
        axes[1, i].set_xticklabels([chr(i + 65) for i in range(26)], rotation=90)
        axes[1, i].set_ylim([0, 1])
        axes[1, i].bar(predicted_label, predictions[index][predicted_label], color='red')
        axes[1, i].bar(true_label, predictions[index][true_label], color='blue')

    plt.suptitle(f"Character: {label}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the title position
    plt.show()


# %%
# Load the trained model
model = load_model('best_model.h5')  # Ensure the correct path to the model

# Define the paths to the images for all test characters (3 images each)
image_paths = [
    "/kaggle/input/test-images/S_1.png", "/kaggle/input/test-images/S_2.png", "/kaggle/input/test-images/S_3.png",
    "/kaggle/input/test-images/R_1.png", "/kaggle/input/test-images/R_2.png", "/kaggle/input/test-images/R_3.png",
    "/kaggle/input/test-images/Y_1.png", "/kaggle/input/test-images/Y_2.png", "/kaggle/input/test-images/Y_3.png",
    "/kaggle/input/test-images/Z_1.png", "/kaggle/input/test-images/Z_2.png", "/kaggle/input/test-images/Z_3.png",
    "/kaggle/input/test-images/O_1.png", "/kaggle/input/test-images/O_2.png", "/kaggle/input/test-images/O_3.png",
    "/kaggle/input/test-images/U_1.png", "/kaggle/input/test-images/U_2.png", "/kaggle/input/test-images/U_3.png"
]

# Corresponding true labels for the images
test_labels = [
    'S', 'S', 'S',
    'R', 'R', 'R',
    'Y', 'Y', 'Y',
    'Z', 'Z', 'Z',
    'O', 'O', 'O',
    'U', 'U', 'U'
]

# Load and preprocess the images
test_images = load_images(image_paths)
test_images = test_images.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
test_images_flattened = test_images.reshape(-1, 28 * 28)  # Flatten images for the model

# Get model predictions
predictions = model.predict(test_images_flattened)



# Display predictions and confidence levels for each character
unique_labels = sorted(set(test_labels))  # Get unique characters

for label in unique_labels:
    label_indices = [i for i, x in enumerate(test_labels) if x == label]
    num_images = len(label_indices)

    # Create subplots for images and prediction value arrays
    fig, axes = plt.subplots(2, num_images, figsize=(15, 8))

    # Ensure axes is always 2D (even when num_images = 1)
    if num_images == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, index in enumerate(label_indices):
        # Plot the test image
        axes[0, i].imshow(test_images[index], cmap='gray')
        predicted_label = np.argmax(predictions[index])
        true_label = ord(test_labels[index]) - 65  # Convert letter to index (A=0, B=1, ...)
        color = 'blue' if predicted_label == true_label else 'red'
        axes[0, i].set_title(f"True: {test_labels[index]}\nPred: {chr(predicted_label + 65)}", color=color)
        axes[0, i].axis('off')

        # Plot the prediction value array
        axes[1, i].bar(range(26), predictions[index], color="#777777")
        axes[1, i].set_xticks(range(26))
        axes[1, i].set_xticklabels([chr(i + 65) for i in range(26)], rotation=90)
        axes[1, i].set_ylim([0, 1])
        axes[1, i].bar(predicted_label, predictions[index][predicted_label], color='red')
        axes[1, i].bar(true_label, predictions[index][true_label], color='blue')

    plt.suptitle(f"Character: {label}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the title position
    plt.show()


# %%
# Load the trained model
model = load_model('First_Try_model.h5')  # Ensure the correct path to the model

# Define the paths to the images for all test characters (3 images each)
image_paths = [
    "/kaggle/input/test-images/S_1.png", "/kaggle/input/test-images/S_2.png", "/kaggle/input/test-images/S_3.png",
    "/kaggle/input/test-images/R_1.png", "/kaggle/input/test-images/R_2.png", "/kaggle/input/test-images/R_3.png",
    "/kaggle/input/test-images/Y_1.png", "/kaggle/input/test-images/Y_2.png", "/kaggle/input/test-images/Y_3.png",
    "/kaggle/input/test-images/Z_1.png", "/kaggle/input/test-images/Z_2.png", "/kaggle/input/test-images/Z_3.png",
    "/kaggle/input/test-images/O_1.png", "/kaggle/input/test-images/O_2.png", "/kaggle/input/test-images/O_3.png",
    "/kaggle/input/test-images/U_1.png", "/kaggle/input/test-images/U_2.png", "/kaggle/input/test-images/U_3.png"
]

# Corresponding true labels for the images
test_labels = [
    'S', 'S', 'S',
    'R', 'R', 'R',
    'Y', 'Y', 'Y',
    'Z', 'Z', 'Z',
    'O', 'O', 'O',
    'U', 'U', 'U'
]

# Load and preprocess the images
test_images = load_images(image_paths)
test_images = test_images.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
test_images_flattened = test_images.reshape(-1, 28 * 28)  # Flatten images for the model

# Get model predictions
predictions = model.predict(test_images_flattened)


# Display predictions and confidence levels for each character
unique_labels = sorted(set(test_labels))  # Get unique characters

for label in unique_labels:
    label_indices = [i for i, x in enumerate(test_labels) if x == label]
    num_images = len(label_indices)

    # Create subplots for images and prediction value arrays
    fig, axes = plt.subplots(2, num_images, figsize=(15, 8))

    # Ensure axes is always 2D (even when num_images = 1)
    if num_images == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, index in enumerate(label_indices):
        # Plot the test image
        axes[0, i].imshow(test_images[index], cmap='gray')
        predicted_label = np.argmax(predictions[index])
        true_label = ord(test_labels[index]) - 65  # Convert letter to index (A=0, B=1, ...)
        color = 'blue' if predicted_label == true_label else 'red'
        axes[0, i].set_title(f"True: {test_labels[index]}\nPred: {chr(predicted_label + 65)}", color=color)
        axes[0, i].axis('off')

        # Plot the prediction value array
        axes[1, i].bar(range(26), predictions[index], color="#777777")
        axes[1, i].set_xticks(range(26))
        axes[1, i].set_xticklabels([chr(i + 65) for i in range(26)], rotation=90)
        axes[1, i].set_ylim([0, 1])
        axes[1, i].bar(predicted_label, predictions[index][predicted_label], color='red')
        axes[1, i].bar(true_label, predictions[index][true_label], color='blue')

    plt.suptitle(f"Character: {label}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the title position
    plt.show()



