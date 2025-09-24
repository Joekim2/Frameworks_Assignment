CORD-19 Metadata Analysis

Documentation and Reflection

1. Code Documentation

The Python and Streamlit scripts were written with comments to explain each step. Examples:

# Load the dataset
df = pd.read_csv("metadata.csv")

# Drop columns with more than 70% missing values
missing_percent = df.isnull().mean() * 100
cols_to_drop = missing_percent[missing_percent > 70].index
df_clean = df.drop(columns=cols_to_drop)

# Convert publish_time to datetime and extract year
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
df_clean['pub_year'] = df_clean['publish_time'].dt.year


This ensures the code is understandable and maintainable.

2. Brief Report of Findings
Dataset Overview

Source: CORD-19 metadata.csv

Contains metadata of coronavirus-related research papers.

After cleaning, unnecessary columns with >70% missing values were removed.

Important fields retained: title, abstract, journal, publish_time, source_x.

Key Findings

Publication Trends: Huge spike in papers around 2020, coinciding with the COVID-19 outbreak.

Top Journals: Several leading journals (e.g., The Lancet, Nature, BMJ) contributed heavily.

Frequent Words in Titles: Most common keywords were vaccine, virus, protein, recombinant, reflecting scientific priorities.

Sources: A few repositories dominate the dataset, such as PMC and Elsevier.

Visualizations

Line chart: publications per year.

Bar chart: top journals publishing COVID-19 research.

Bar chart: most frequent words in paper titles.

Bar chart: distribution of publications by source.

3. Reflection
Challenges

Missing data: Many columns were incomplete, requiring filtering.

Data inconsistencies: Journal names appeared in multiple formats.

Visualization issues: Bar chart orders had to be fixed (largest value at top).

Streamlit debugging: Empty filters sometimes caused crashes (solved with error handling).

Learning Outcomes

Gained confidence in data cleaning using pandas.

Practiced regular expressions and text frequency analysis.

Built an interactive Streamlit app for real-time visualization.

Improved debugging skills when handling runtime errors in Python.

Future Improvements

Add word cloud visualizations for better keyword insight.

Perform topic modeling/NLP for deeper text analysis.

Standardize journal/source names for more accurate counts.

Deploy Streamlit app online for broader access.