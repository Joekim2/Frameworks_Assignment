 # ---Part 1: Data Loading and Basic Exploration---
 # 1.Download and load the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import re
import os
from collections import Counter

# load dataset
n  = 100000
metadata_df = pd.read_csv('metadata.csv', nrows=n)
metadata_df.to_csv('metad.csv', index=False)

# Now load metad.csv as the main data
df = pd.read_csv('metad.csv', low_memory=False)

#  Display the first few rows of the dataframe
print(df.head(50))

 # 2. Basic data exploration
   # Check the DataFrame dimensions (rows, columns)
print("DataFrame shape:", df.shape)
# Column data types
print(df.dtypes)
 # missing values per column
print(df.isnull().sum())
  # Generate basic statistics for numerical columns
print(df.describe())

 #---Part 2: Data Cleaning and preparation---
    # =============================
print("Cleaning dataset...")

# Drop columns with >70% missing values
missing_percent = df.isnull().mean() * 100
cols_to_drop = missing_percent[missing_percent > 70].index
df_clean = df.drop(columns=cols_to_drop)

# Convert publish_time to datetime & extract year
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
df_clean['pub_year'] = df_clean['publish_time'].dt.year

# Word counts
df_clean['title_word_count'] = df_clean['title'].fillna("").apply(lambda x: len(re.findall(r'\w+', str(x))))
df_clean['abstract_word_count'] = df_clean['abstract'].fillna("").apply(lambda x: len(re.findall(r'\w+', str(x))))

# Drop rows without title
df_clean = df_clean.dropna(subset=['title'])

# Save cleaned dataset
df_clean.to_csv("cleaned_metadata.csv", index=False)
print("Cleaned shape:", df_clean.shape)
print("Saved cleaned dataset as cleaned_metadata.csv")

# =============================
# 3. Data Analysis & Visualization
# =============================
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Publications by year
pubs_by_year = df_clean['pub_year'].value_counts().sort_index()
plt.figure(figsize=(10,5))
plt.plot(pubs_by_year.index, pubs_by_year.values)
plt.title("Publications by Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "publications_by_year.png"))
plt.close()

# Top journals
top_journals = df_clean['journal'].value_counts().head(20)
plt.figure(figsize=(10,6))
plt.barh(top_journals.index[::-1], top_journals.values[::-1])
plt.title("Top Journals")
plt.xlabel("Number of Papers")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "top_journals.png"))
plt.close()

# Word frequency in titles
all_titles = " ".join(df_clean['title'].fillna("").astype(str).tolist()).lower()
tokens = re.findall(r'\b[a-zA-Z]{2,}\b', all_titles)
stopwords = {"the","and","of","in","to","a","on","for","with","by","an","is","from","using"}
tokens = [t for t in tokens if t not in stopwords]
word_counts = Counter(tokens).most_common(30)

words, counts = zip(*word_counts)
plt.figure(figsize=(10,6))
plt.barh(words[::-1], counts[::-1])
plt.title("Top Words in Titles")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "top_words_in_titles.png"))
plt.close()

# Source distribution
top_sources = df_clean['source_x'].value_counts().head(20)
plt.figure(figsize=(10,6))
plt.barh(top_sources.index[::-1], top_sources.values[::-1])
plt.title("Top Sources")
plt.xlabel("Number of Papers")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "top_sources.png"))
plt.close()

print("Saved plots in 'plots/' directory")

# 4. Create Streamlit App
# =============================
streamlit_code = """import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_metadata.csv")

df = load_data()

st.title("CORD-19 Metadata Explorer")
st.write("Interactive exploration of COVID-19 research papers.")

# Sidebar filters
if 'pub_year' in df.columns:
    min_year, max_year = int(df['pub_year'].min()), int(df['pub_year'].max())
    year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))
    df = df[(df['pub_year'] >= year_range[0]) & (df['pub_year'] <= year_range[1])]

if 'journal' in df.columns:
    journals = ["All"] + list(df['journal'].value_counts().head(20).index)
    selected = st.sidebar.selectbox("Select Journal", journals)
    if selected != "All":
        df = df[df['journal'] == selected]

# Data sample
st.subheader("Sample Data")
st.dataframe(df.head(20))

# Publications by year
if 'pub_year' in df.columns:
    pubs = df['pub_year'].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.plot(pubs.index, pubs.values, marker="o")
    ax.set_title("Publications by Year")
    st.pyplot(fig)

# Top journals
if 'journal' in df.columns:
    top = df['journal'].value_counts().head(20)
    fig2, ax2 = plt.subplots()
    ax2.barh(top.index[::-1], top.values[::-1])
    ax2.set_title("Top Journals")
    st.pyplot(fig2)

# Word frequency in titles
all_titles = " ".join(df['title'].fillna("").astype(str).tolist()).lower()
tokens = re.findall(r'\\\\b[a-zA-Z]{2,}\\\\b', all_titles)
stopwords = {"the","and","of","in","to","a","on","for","with","by","an","is","from","using"}
tokens = [t for t in tokens if t not in stopwords]
from collections import Counter
word_counts = Counter(tokens).most_common(20)

words, counts = zip(*word_counts)
fig3, ax3 = plt.subplots()
ax3.barh(words[::-1], counts[::-1])
ax3.set_title("Top Words in Titles")
st.pyplot(fig3)

# Source distribution
if 'source_x' in df.columns:
    top_sources = df['source_x'].value_counts().head(20)
    fig4, ax4 = plt.subplots()
    ax4.barh(top_sources.index[::-1], top_sources.values[::-1])
    ax4.set_title("Top Sources")
    st.pyplot(fig4)
"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)

print("Streamlit app saved as app.py")
print("Run it with: streamlit run app.py")