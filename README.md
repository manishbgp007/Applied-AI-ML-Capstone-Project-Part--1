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
* We used the **California Housing dataset**, a well-known structured dataset available via scikit-learn.
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

* **Loading the Dataset**
  * The dataset was loaded into **Google Colab** using the `pandas` library with the `pd.read_csv()` function. This imported the CSV file into a pandas DataFrame, making it easy to perform data exploration, cleaning, and analysis.

* **Previewing the Dataset**
  * The `df.head()` function was used to display the first five rows of the dataset. This provided a quick overview of the data, including the column names, sample values, and overall layout. It also helped verify that the dataset was loaded correctly without any formatting issues.

* **Examining Data Types**
  * The `df.dtypes` command was used to check the data type of each column. This step identified whether each feature was stored as an integer ('int64'), floating-point number ('float64'), object (categorical or text), or another data type. Verifying data types is important because machine learning algorithms require numerical inputs, and incorrect data types may need to be converted during preprocessing.

* **Checking Dataset Dimensions**
  * The `df.shape` attribute was used to determine the number of rows and columns in the dataset. This information confirmed the dataset size and verified that it satisfied the project requirements regarding the minimum number of records and variables.

* **Initial Dataset Assessment**
  * The inspection process confirmed that the dataset was successfully loaded and contained the expected number of features and observations. It also provided an initial understanding of the dataset's structure, allowing potential issues such as incorrect data types, missing values, or formatting inconsistencies to be identified before proceeding to data cleaning and exploratory data analysis.

#### Task 2: Null Value Analysis

Missing value analysis was performed to identify incomplete or unavailable data within the dataset. Handling missing values is an important preprocessing step because they can negatively affect the performance and accuracy of machine learning models.

* **Calculating Missing Values**
  * The number of missing values in each column was calculated using the `df.isnull().sum()` function. To better understand the extent of missing data, the percentage of missing values was also computed using the following formula:
  * **Missing Value Percentage = (Number of Missing Values / Total Number of Rows) × 100**
  * This analysis helped identify which columns contained missing data and the severity of the missing values.

* **Identifying Columns with High Missing Values**
  * Columns containing more than **20% missing values** were identified and flagged for special handling. Such columns require careful consideration because a large amount of missing data may reduce data quality and affect model performance. Depending on the project requirements, these columns can be removed, ignored, or processed using advanced imputation techniques.

* **Imputing Missing Values**
  * For numeric columns with **less than 20% missing values**, the missing entries were replaced with the **median** of the respective column using the `fillna()` function.

* **Justification for Using the Median**
  * The **median** was selected instead of the mean because it is more robust to extreme values and skewed distributions. Unlike the mean, the median is not significantly affected by outliers, making it a more reliable measure of central tendency for replacing missing values. This approach preserves the overall distribution of the data and improves the quality of the dataset for further analysis and machine learning.

* **Outcome**
  * After imputing the missing values in the eligible numeric columns, the dataset became more complete and suitable for further preprocessing, exploratory data analysis, feature engineering, and machine learning model development.
 

#### Task 3: Duplicate Detection and Removal

Duplicate records were analyzed to ensure the dataset contained only unique observations. Duplicate rows can introduce bias into data analysis and machine learning models by giving unnecessary weight to repeated information. Therefore, identifying and removing duplicates is an essential step in the data cleaning process.

* **Detecting Duplicate Rows**
  * The total number of duplicate records in the dataset was identified using the `df.duplicated().sum()` function. This function counts all rows that are exact duplicates of previous rows, allowing us to measure the extent of duplicate data in the dataset.

* **Removing Duplicate Records**
  * After identifying the duplicate rows, they were removed using the `df.drop_duplicates()` function. This operation retained only the first occurrence of each record and eliminated all subsequent duplicate entries, resulting in a cleaner and more reliable dataset.

* **Reporting the Results**
  * The number of duplicate rows removed was recorded and reported. Comparing the dataset shape before and after duplicate removal helped verify that the duplicate records had been successfully eliminated.

* **Rechecking Missing Values**
  * After removing duplicate rows, the percentage of missing values in each column was recalculated. This verification ensured that duplicate removal did not unintentionally alter the distribution of missing values or introduce any inconsistencies into the dataset.

* **Outcome**
  * The duplicate detection and removal process improved the overall quality and reliability of the dataset by eliminating redundant records. The cleaned dataset became more suitable for exploratory data analysis, feature engineering, and machine learning model training, reducing the risk of biased results caused by repeated observations.


#### Task 4: Data Type Correction

Data type correction was performed to ensure that each column had the most appropriate data type for analysis and machine learning. Incorrect data types can lead to errors during preprocessing, increase memory usage, and reduce computational efficiency. Therefore, verifying and correcting data types is an important step in data preparation.

* **Identifying Incorrect Data Types**
  * The data types of all columns were examined using the `df.dtypes` command. This inspection helped identify columns that had been assigned inappropriate data types during data import, such as numeric values stored as text (`object`) or categorical variables stored as plain strings.

* **Converting Numeric Columns**
  * Columns containing numeric values but stored as the `object` data type were converted to appropriate numeric types (`int64` or `float64`) using the `pd.to_numeric()` function. This conversion ensured that mathematical operations, statistical analysis, and machine learning algorithms could process these features correctly.

* **Converting Categorical Columns**
  * Columns containing repetitive text values (such as categories or labels) were converted from the `object` data type to the `category` data type. Using the categorical data type reduces memory consumption and improves the efficiency of data processing, especially for columns with a limited number of unique values.

* **Memory Usage Comparison**
  * The memory usage of the dataset was measured before and after performing the data type conversions. The comparison showed that converting repetitive string columns to the `category` data type significantly reduced the overall memory footprint of the dataset while maintaining the same information.

* **Outcome**
  * After correcting the data types, the dataset became more efficient in terms of memory usage and computational performance. The corrected data types also ensured compatibility with subsequent preprocessing techniques, exploratory data analysis, feature engineering, and machine learning model development.

