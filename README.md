# **Applied-AI-ML-Capstone-Project**

## Housing Value Prediction Project

### Project Description
* This project is part of the Applied-AI-ML-Capstone-Project, where the goal is to build an end-to-end data intelligence system. It covers four stages.

### Data ingestion & exploratory analysis
* Supervised machine learning
* Ensemble modeling & pipelines
* LLM-powered feature
* We focus on cleaning the dataset, handling missing values, correcting data types, detecting outliers, and performing exploratory data analysis with visualizations and correlation studies. The cleaned dataset is saved for use in later parts.

### Dataset
* We used the California Housing dataset, a well-known structured dataset available via scikit-learn.
* Reason for choice:
  * Contains >20,000 rows (well above the minimum requirement of 500).</li>
  * Includes multiple numeric columns (e.g., MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude).
  * Includes categorical columns when extended with derived features (e.g., region bins).
  * Target variable MedHouseVal is continuous, suitable for regression, and can be binarized for classification tasks.
  * Publicly available and widely used for benchmarking ML models.
    
### Technologies Used
* Python 3.9+
* Google Colab
* Pandas for data manipulation
* NumPy for numerical operations
* Matplotlib & Seaborn for visualizations
* Scikit-learn for correlation, preprocessing, and ML models


## Q: 1
### Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

#### Task 1: Dataset Inspection
The first step of the project was to inspect the dataset to understand its overall structure, quality, and suitability for machine learning. A thorough inspection helps identify potential issues before data preprocessing and model development.

* **1. Loading the Dataset**
  * The dataset was loaded into Google Colab using the `pandas` library with the `pd.read_csv()` function. This imported the CSV file into a pandas DataFrame, making it easy to perform data exploration, cleaning, and analysis.

* **2. Previewing the Dataset**
  * The `df.head()` function was used to display the first five rows of the dataset. This provided a quick overview of the data, including the column names, sample values, and overall layout. It also helped verify that the dataset was loaded correctly without any formatting issues.

* **3. Examining Data Types**
  * The `df.dtypes` command was used to check the data type of each column. This step identified whether each feature was stored as an integer (`int64`), floating-point number (`float64`), object (categorical or text), or another data type. Verifying data types is important because machine learning algorithms require numerical inputs, and incorrect data types may need to be converted during preprocessing.

* **4. Checking Dataset Dimensions**
  * The `df.shape` attribute was used to determine the number of rows (observations) and columns (features) in the dataset. This information confirmed the dataset size and verified that it satisfied the project requirements regarding the minimum number of records and variables.

* **5. Initial Dataset Assessment**
  * The inspection process confirmed that the dataset was successfully loaded and contained the expected number of features and observations. It also provided an initial understanding of the dataset's structure, allowing potential issues such as incorrect data types, missing values, or formatting inconsistencies to be identified before proceeding to data cleaning and exploratory data analysis.

