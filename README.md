# Machine Learning Handwritten Alphabet Recognition

## 📖 Project Overview
This project involves recognizing handwritten English alphabets using machine learning techniques. The dataset consists of grayscale images represented in a structured CSV format, making it ideal for training models in character recognition.

### 🎯 Objectives:
- Load and preprocess the dataset.
- Perform exploratory data analysis (EDA).
- Train and evaluate multiple models:
  - **Support Vector Machine (SVM) - Linear & Non-linear**
  - **Logistic Regression (from scratch)**
  - **Neural Networks (TensorFlow-based)**
- Compare model performance and suggest the best approach.

---

## 📊 Dataset Overview
🔗 **Dataset Link:** [A-Z Handwritten Alphabets Dataset](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)

✔ **Size:** 370,000+ images of handwritten alphabets.  
✔ **Format:** CSV file where each row represents an image with pixel values and labels.  
✔ **Resolution:** 28×28 pixels, grayscale format.  
✔ **Labels:** 26 classes (A-Z) mapped numerically (0 = A, 1 = B, ..., 25 = Z).  

---

## 🛠 Data Preprocessing
### ✅ Steps:

1️⃣ **Load and Explore Dataset**
   - Inspect dataset structure and statistics.
   
2️⃣ **Preprocess Data**
   - Normalize pixel values to [0,1] range.
   - Reshape flattened vectors into 28×28 images for visualization.
   
3️⃣ **Split Data**
   - Training (80%) / Testing (20%) using `train_test_split`.
   
4️⃣ **Visualize Sample Images**
   - Display random samples to check data quality.

---

## 🤖 Machine Learning Models
### 📌 1. **Support Vector Machine (SVM)**
✔ Implemented both **Linear SVM** and **Non-Linear SVM (RBF Kernel)**.
✔ SVM provided high accuracy but required careful kernel tuning.

### 📌 2. **Logistic Regression (from Scratch)**
✔ Implemented one-vs-all multi-class classification.
✔ Tested different learning rates (`0.1`, `0.01`, `0.5`).
✔ Achieved high accuracy but was computationally expensive.

### 📌 3. **Neural Networks (TensorFlow-based)**
✔ Implemented a deep learning model with:
   - Input Layer: Flatten(28×28)
   - Hidden Layers: Fully connected (ReLU activation, Dropout)
   - Output Layer: Softmax (26 classes)
✔ Optimized model achieved **98% accuracy**.

---

## 📊 Model Performance Comparison
| Model                | Test Accuracy | Training Time |
|----------------------|--------------|--------------|
| **SVM (Linear)**    | 88.98%       | Moderate    |
| **SVM (Non-Linear)**| 94.79%       | High        |
| **Logistic Regression** | 85.00% | Very High   |
| **Neural Network (Model 1)** | 97.00% | Moderate |
| **Neural Network (Optimized)** | **98.00%** | High |

---

## 🚀 Key Takeaways
✅ **Neural Network (Optimized)** outperformed all models in accuracy.  

✅ **SVM (Non-Linear)** was an excellent alternative but required longer training time.  

✅ **Logistic Regression** was interpretable but computationally expensive.  

✅ **Data preprocessing significantly impacts final model accuracy.**  

---

## 🛠 Technologies & Libraries Used
🔹 **Python**  
🔹 **Pandas, NumPy** (Data manipulation)  
🔹 **Matplotlib, Seaborn** (Data visualization)  
🔹 **Scikit-learn** (SVM, Logistic Regression)  
🔹 **TensorFlow, Keras** (Deep Learning)  

---

## 🔮 Future Enhancements
🔹 Implement **CNNs (Convolutional Neural Networks)** for better feature extraction.  
🔹 Experiment with **data augmentation** to improve generalization.  
🔹 Optimize SVM with **different kernel functions and hyperparameters**.  

---

### 📌 **Authors: Team Members (Cairo University)**

