# 📩 SMS Spam Classifier  

This project is a **machine learning model** that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Python and scikit-learn.  

---

## 🚀 Features  
- Preprocesses text messages (cleaning, tokenization, vectorization)  
- Uses **Naive Bayes Classifier** for classification  
- Achieves high accuracy on the SMS Spam dataset  
- Easy to run and test locally  

---

## 📂 Project Structure  
sms-spam-classifier/
│── sms_classifier.py # Main Python script
│── spam.csv # Dataset (SMS messages with labels)
│── README.md # Project documentation

---

## 🖼️ Screenshots  

### 1️⃣ Dataset Sample  
![Dataset Example](./figure1)  

### 2️⃣ Training Output  
![Training Output](./figure2)  

### 3️⃣ Prediction Example  
![Prediction Example](excel_file)  

---

## ⚙️ Installation  

### 1. Clone this repo
```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier

### **2. Install dependencies**
pip install pandas scikit-learn

### **3. Run the classifier**
python sms_classifier.py


## **📊 Dataset**
This project uses the SMS Spam Collection Dataset.
Make sure spam.csv is placed in the same folder as sms_classifier.py.

## **🧑‍💻 Example Output**
Training Accuracy: 98.7%
Test Accuracy: 97.3%

Enter a message: "Congratulations! You won a free ticket"
Prediction: SPAM 🚨

## **🤝 Contributing**
Pull requests are welcome! If you’d like to improve this project, feel free to fork the repo and submit a PR.

## **⭐ Acknowledgements**

1. UCI ML Repository
2. scikit-learn documentation


