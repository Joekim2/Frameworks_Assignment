import streamlit as st
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
tokens = re.findall(r'\\b[a-zA-Z]{2,}\\b', all_titles)
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
