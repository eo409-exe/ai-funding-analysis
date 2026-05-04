# Following the Money: What Really Drives AI Startup Funding?

**BEE2041 — Data Science in Economics**

📊 **[Read the live blog here](https://eo409-exe.github.io/ai-funding-analysis/blog.html)**

An analysis of 502 AI-adjacent companies across 40 countries, covering 
over 1,000 funding rounds from 2020 to 2026. This project investigates 
what factors drive AI startup funding success using web-scraped data, 
descriptive analysis, and regression modelling.

---

## Research Questions

1. How has AI startup funding evolved between 2020 and 2026?
2. Which sectors and geographies attract the most capital?
3. What company characteristics predict funding success?

---

## Replication Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/eo409-exe/ai-funding-analysis.git
cd ai-funding-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebooks in Order

Execute the notebooks sequentially from the `notebooks/` folder:

| Notebook | Description |
|---|---|
| `01_data_collection.ipynb` | Scrapes funding data |
| `02_exploratory_analysis.ipynb` | Produces descriptive visualisations |
| `03_modelling.ipynb` | Runs OLS and logistic regression models |

Each notebook reads from and writes to the `data/` and `outputs/` 
folders automatically. No manual file moving is required.

### 4. Render the Blog

```bash
quarto render blog.qmd
```

This produces `blog.html` in the project root.

---

## Data Sources

Data was scraped from **[Latestrounds.com](https://latestrounds.com)**, 
an aggregator of AI startup funding announcements. The dataset covers 
funding rounds announced between 2020 and 2026.

**Limitations:**
- The data source is English-language, meaning Chinese AI investment 
  activity is significantly underrepresented
- The dataset captures funding announcements, not valuations or 
  longer-run outcomes such as exits or IPOs
- 118 companies (23.5%) are missing founding year data

---

## Methodology

**Data cleaning**: Company names standardised using fuzzy matching; 
funding amounts parsed from string format; locations split into city 
and country.

**Regression models**: Two models are estimated on 384 companies with 
complete data, excluding OpenAI, xAI, and Anthropic as extreme outliers:
- OLS regression predicting log(total funding)
- Logistic regression predicting probability of a mega-round ($50M+)

HC3 robust standard errors are used throughout to account for 
heteroskedasticity.

