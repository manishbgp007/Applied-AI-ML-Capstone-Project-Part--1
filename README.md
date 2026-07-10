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

### Task 1: Dataset Inspection
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

### Task 2: Null Value Analysis

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
 

### Task 3: Duplicate Detection and Removal

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


### Task 4: Data Type Correction

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


### Task 5: Descriptive Statistics and Skewness Analysis

Descriptive statistics and skewness analysis were performed to understand the distribution, central tendency, variability, and shape of the numeric features in the dataset. This analysis provided valuable insights into the characteristics of the data and helped determine appropriate preprocessing techniques.

* **Generating Descriptive Statistics** :- 
The `df.describe()` function was used to generate summary statistics for all numeric columns. The output included the following statistical measures:
  * **Count:** Number of non-missing observations.
  * **Mean:** Average value of the feature.
  * **Standard Deviation (Std):** Measure of data variability.
  * **Minimum (Min):** Smallest value in the column.
  * **25th Percentile (Q1):** First quartile.
  * **50th Percentile (Median):** Middle value of the dataset.
  * **75th Percentile (Q3):** Third quartile.
  * **Maximum (Max):** Largest value in the column.
    
These statistics provided an overall understanding of the dataset and helped identify unusual values or potential outliers.

* **Computing Skewness**
  * The skewness of each numeric column was calculated using the `df.skew()` function. Skewness measures the degree of asymmetry in the distribution of data around its mean. The column with the highest absolute skewness was identified because it represents the most unevenly distributed feature in the dataset.

* **Interpretation of Skewness**

* The skewness values were interpreted as follows:
  * **Positive Skewness:** The distribution has a long right tail, and the mean is generally greater than the median.
  * **Negative Skewness:** The distribution has a long left tail, and the mean is generally smaller than the median.
  * **Skewness Close to Zero:** The distribution is approximately symmetric, indicating that the mean and median are relatively close.

* **Importance of Skewness Analysis**
  * Understanding skewness is important because highly skewed data can influence statistical analysis and machine learning model performance. Features with large positive or negative skewness may require transformations or robust preprocessing techniques before model training.

* **Justification for Median Imputation**
  * The skewness analysis supported the decision to use the **median** for imputing missing values. Since the median is less affected by extreme values and skewed distributions than the mean, it provides a more reliable estimate of the central tendency for skewed numeric data.

* **Outcome**
  * The descriptive statistics and skewness analysis provided a comprehensive understanding of the dataset's numerical features. The results helped identify highly skewed variables, supported the choice of median imputation, and established a strong foundation for further exploratory data analysis, feature engineering, and machine learning model development.
 

### Task 6: Outlier Detection Using the Interquartile Range (IQR)

Outlier detection was performed to identify unusually high or low values in the dataset that could influence statistical analysis and machine learning model performance. Instead of removing outliers immediately, they were carefully analyzed and documented for future preprocessing.

* **Selecting Numeric Columns**
Two important numeric columns were selected for outlier analysis. These features were chosen because they play a significant role in understanding the distribution of the data and may contain extreme values.

* **Calculating Q1, Q3, and IQR**
For each selected numeric column, the following statistical measures were calculated:
  * **First Quartile (Q1):** The 25th percentile of the data.
  * **Third Quartile (Q3):** The 75th percentile of the data.
  * **Interquartile Range (IQR):** The spread of the middle 50% of the data, calculated as:
  
  **IQR = Q3 − Q1**
  
The IQR method is a widely used and robust technique for detecting outliers because it is less affected by extreme values than methods based on the mean and standard deviation.
* **Identifying Outliers**
The lower and upper bounds were calculated using the standard IQR formula:
  * **Lower Bound = Q1 − 1.5 × IQR**
  * **Upper Bound = Q3 + 1.5 × IQR**
    
Any observation with a value below the lower bound or above the upper bound was considered an outlier. The total number of outliers identified in each selected column was calculated and documented.

* **Decision on Outlier Handling**
  * Although outliers were detected, they were **not removed** during Part 1 of the project. Removing outliers without careful analysis can lead to the loss of important information, especially if the extreme values represent genuine observations rather than data entry errors.

* **Future Strategy**
  * The detected outliers were documented for future preprocessing. In **Part 2**, advanced techniques such as **outlier capping (Winsorization), transformation, scaling, or other robust preprocessing methods** will be considered based on their impact on the machine learning models.

* **Outcome**
  * The IQR-based outlier analysis successfully identified extreme observations while preserving the integrity of the original dataset. This approach ensures that the data remains complete during the initial exploration phase and allows informed decisions about outlier treatment in later stages of the project.


### Task 7: Data Visualization
Data visualization was performed to better understand the distribution, trends, relationships, and patterns within the dataset. Visual representations make it easier to identify important insights that may not be obvious from numerical summaries alone. Five different types of plots were created as required by the project.

* **Line Plot**
  * A **line plot** was created for a selected numeric variable to observe how its values change across the dataset. This visualization helped identify overall trends, fluctuations, and any noticeable increases or decreases in the values. It also made it easier to detect unusual patterns or sudden changes in the data.

* **Bar Chart**
  * A **bar chart** was generated to compare the **mean values** of a numeric variable across different categories. This plot clearly illustrated the differences between groups and helped identify which category had the highest or lowest average value. The visualization provided a simple comparison of categorical data.

* **Histogram**
  * A **histogram** was created for the numeric column with the highest absolute skewness. This plot showed the frequency distribution of the data and revealed whether the variable was normally distributed, positively skewed, or negatively skewed. The histogram also helped identify the presence of outliers and the overall spread of the data.

* **Scatter Plot**
  * A **scatter plot** was produced using two correlated numeric variables to examine the relationship between them. The plot helped determine whether a positive, negative, or weak correlation existed between the selected features. It also allowed the identification of clusters, trends, and potential outliers within the data.

* **Box Plot**
  *A **box plot** was created for a numeric variable grouped by a categorical variable. This visualization displayed the median, quartiles, spread of the data, and potential outliers for each category. Comparing multiple box plots provided insights into differences in data distribution among the categories.

* **Interpretation of Visualizations** :-
Each visualization was carefully analyzed and explained in the README. The interpretations focused on:
  * Identifying trends and patterns in the data.
  * Comparing average values across categories.
  * Understanding the distribution of numeric variables.
  * Detecting skewness and possible outliers.
  * Examining relationships between correlated features.
  * Comparing the spread and variability of data across different groups.

* **Outcome**
  * The visualizations provided valuable insights into the dataset and supported the findings from descriptive statistics and exploratory data analysis. They improved the understanding of feature distributions, relationships, and data quality, forming a strong foundation for feature engineering and machine learning model development in the subsequent parts of the project.


### Task 8: Correlation Heatmap

Correlation analysis was performed to measure the strength and direction of relationships between the numeric variables in the dataset. The Pearson correlation coefficient was calculated for all numeric features, and the results were visualized using a correlation heatmap. This analysis helped identify highly correlated features that may influence machine learning models and feature selection.

* **Computing the Correlation Matrix**  
The Pearson correlation matrix was calculated using the `df.corr()` function. The correlation coefficient ranges from **-1 to +1**, where:
  * **+1** indicates a perfect positive correlation.
  * **-1** indicates a perfect negative correlation.
  * **0** indicates no linear relationship between the variables.

This matrix provided a numerical summary of the relationships among all numeric features.

* **Visualizing the Correlation Heatmap**
  * The correlation matrix was visualized using a **heatmap**. Different colors represented different correlation strengths, making it easy to identify variables that were strongly positively or negatively correlated. The heatmap provided a clear overview of the relationships among all numeric features in the dataset.

* **Identifying the Strongest Correlation Pair**
  * The pair of numeric variables with the highest absolute correlation coefficient was identified from the correlation matrix. A strong positive correlation indicates that the two variables tend to increase or decrease together, while a strong negative correlation indicates that one variable increases as the other decreases.

* **Correlation Does Not Imply Causation**
  * Although two variables may have a strong correlation, this does **not** necessarily mean that one variable causes the other. Correlation only measures the strength of a linear relationship between variables and does not establish a cause-and-effect relationship.

* **Possible Alternative Explanation**
  * A strong correlation may be explained by the presence of a **third variable** that influences both features. For example, if **House Size** and **House Price** are highly correlated, the relationship may also be affected by factors such as **Location**, **Neighborhood Quality**, or **Number of Rooms**. Similarly, in other datasets, variables may appear strongly correlated because they are both influenced by another hidden factor rather than directly affecting each other.

* **Outcome**
  * The correlation heatmap provided valuable insights into the relationships between numeric features and helped identify strongly correlated variables. These findings will support feature selection, multicollinearity analysis, and machine learning model development in the next stages of the project while avoiding incorrect conclusions about causation.
