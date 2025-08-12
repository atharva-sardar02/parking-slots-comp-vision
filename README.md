# 🅿️ Parking Slots Detection – Machine Learning & Computer Vision

An intelligent parking management solution that detects the occupancy status of parking slots from CCTV images and predicts availability patterns using **Machine Learning (SVM, Random Forest, KNN)** and **Computer Vision (OpenCV)** techniques.

---

## 🚀 Features

- **Vacancy Detection:**
  - **SVM + HOG Features** — Classifies individual parking spots as vacant/occupied with **96% accuracy**.
  - **Computer Vision (OpenCV)** — Real-time detection and counting of occupied/vacant slots using background subtraction & bounding boxes.
- **Time & Date-Based Prediction:** Predicts availability patterns based on historical occupancy trends using **Decision Tree**, **Random Forest**, **KNN**, and more.
- **Annotation Workflow:** Manual parking slot labeling for dataset preparation.
- **Performance Metrics:** Accuracy, precision, recall, and F1-score tracking for each model.

---

## 🛠 Tech Stack

**Core ML & CV**
- Python, Scikit-Learn, OpenCV, NumPy, Pandas, Matplotlib

**Models Implemented**
- SVM (Support Vector Machine)
- Random Forest, Decision Tree, KNN, Logistic Regression, Naïve Bayes, Neural Network

**Data**
- [Find a Car Park – Kaggle Dataset](https://www.kaggle.com/datasets/daggysheep/find-a-car-park/data)

---

## 📂 Repository Structure

```plaintext
├── data/                        # Dataset: 'free' (vacant) and 'full' (occupied) images
├── dump/                        # Temporary files
├── free/                        # Sample vacant slot images
├── full/                        # Sample occupied slot images
├── trial data/                  # Test images for CV approach
├── sample_dataset.csv            # Metadata with time/date labels
├── predicting_vacant_space_*.py  # SVM + HOG detection scripts
├── skimage_svm4.py               # SVM classifier training
├── UC_Final_Project.py           # Main project script
├── decision_tree_plot*.png       # Model visualizations
├── README.md                     # Project documentation
```
## 🖥 Running Locally

**1️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```
**2️⃣ Run SVM-based Vacancy Detection**
```bash
python skimage_svm4.py
```
**3️⃣ Run Computer Vision (OpenCV) Real-time Detection**
```bash
python last_moment.py
```
**4️⃣ Run Time & Date Prediction Models**
```bash
python UC_Final_Project.py
```

## ⚙ Configuration
- **Dataset:** Place `free/` and `full/` folders inside the `data/` directory.
- **Parameters:** Modify image dimensions, HOG parameters, and model settings inside respective `.py` scripts.

---

## 📈 Results
- **SVM + HOG:** 96% accuracy, high precision & recall.
- **Random Forest, Decision Tree, KNN:** 92% accuracy in time/date-based prediction.
- **Real-time CV:** Successfully detects and counts slots in under 0.05s per frame on CPU.

---

## 📸 Screenshots
### SVM + HOG Detection Output
![SVM Detection](images/svm_detection_output.png)

### Computer Vision (OpenCV) Real-time Detection
![CV Real-time Detection](images/cv_realtime_detection.png)

### Decision Tree Plot
![Decision Tree](images/decision_tree_plot.png)
