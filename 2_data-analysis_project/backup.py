# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# # A. An overview of the dataset

# %% [markdown]
# 1. Import the dataset

# %%
file = "./data_385k.pkl/data_385k.pkl"
df = pd.read_pickle(file, compression="zip")

# %% [markdown]
# 2. Check number of columns and rows

# %%
n_rows, n_columns = df.shape

print(
    f"The dataset has {n_columns} columns (data fields), and {n_rows} rows (data entries)."
)

# %% [markdown]
# 3. Display first 5 entries

# %%
df.head()

# %% [markdown]
# 4. Column datatypes

# %% [markdown]
# In order to create separate lists to hold the names of columns of the same data type, I first check the unique dataypes for all columns. And then use this information to filter out the dtypes series.

# %%
dtypes = df.dtypes
dtypes_unique = dtypes.unique()

print(f"The different columns datatypes are :\n{dtypes_unique}")

# Next we create a list of column names corresponding to each datatype
columns_float64 = dtypes[dtypes == "float64"]
columns_object = dtypes[dtypes == "object"]
columns_int64 = dtypes[dtypes == "int64"]

# %% [markdown]
# 5. Type of information contained in columns

# %% [markdown]
# After obtaining a list with all the column names, the lists of column names containing 'per_hundred', 'per_portion' and 'unit' are obtained by using lis comprehensions with a comparison agains these strings. The list of remaining column names are obtained by subtracting sets of these sub-lists from all column names.

# %%
columns = df.columns

cols_per_hundred = [col for col in columns if col[-12:] == "_per_hundred"]
cols_per_portion = [col for col in columns if col[-12:] == "_per_portion"]
cols_unit = [col for col in columns if col[-5:] == "_unit"]
cols_other = list(
    set(columns) - set(cols_per_hundred) - set(cols_per_portion) - set(cols_unit)
)  # also contains the 'id' column name

# %% [markdown]
# # B. Data Cleaning

# %% [markdown]
# 1. Duplicated  products

# %%
# duplicated rows
duplicate_rows = df.duplicated(keep="first")
print(sum(duplicate_rows))
df[duplicate_rows]

# %%
df_clean_dup = df.drop_duplicates(keep="first")

n_rows, n_columns = df_clean_dup.shape
print(
    f"The dataset has {n_columns} columns (data fields), and {n_rows} rows (data entries)."
)

# %% [markdown]
# The original dataset contains 10500 duplicated rows. After removing all the deplicates, i.e. keeping a single entry for each repeated row, the new dataset `df_clean_duplicate` contains (385384-10500) = 374884 rows while maintaining the same number of columns 99.

# %% [markdown]
# 2. Missing values

# %% [markdown]
# The instructions below are use to change the pandas display options such that:
# - all rows of a dataframe are displayed
# - only two decimal places are shown for floats

# %%
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.2f}".format)

# %% [markdown]
# A new dataframe `df_missing` is created containing the total number of missing values for each column, and their respective percentage. The total number of rows in `df_clean` - 374884 - was obtained in the previous point.

# %%
df_missing = df_clean_dup.isna().sum().to_frame()
df_missing.rename(columns={0: "missing total"}, inplace=True)
df_missing["missing percentage"] = df_missing["missing total"] / 374884 * 100
df_missing.sort_values("missing total", ascending=False, inplace=True)
df_missing

# %% [markdown]
# Next we'll be using the Missingno library to gain insight into the distribution of missing values

# %%
import missingno as msno

msno.bar(df_clean_dup)
msno.matrix(df_clean_dup)

# %%
import seaborn as sns

pd.set_option("display.max_rows", 20)


# %% [markdown]
# Next we the percentage of missing values for the 4 types of columns previously obtained in task A5. We decided to present these as horizontal bar plots, with column name in the y-axis. Since the latter are categorical variables we could not directly use the `sns.barplot` plotting function, but had to use the `sns.catplot`

# %%

# Creation of auxiliary containing percentage of missing values per column
df_misspercentage = (df_clean_dup.isna().sum() / len(df_clean_dup) * 100).to_frame(
    name="missing_percentage"
)

# Columns with name ending in "_per_hundred"
sns.catplot(
    data=df_misspercentage.loc[cols_per_hundred].reset_index(),
    y="index",
    x="missing_percentage",
    kind="bar",
)

# Columns with name ending in "_per_portion"
sns.catplot(
    data=df_misspercentage.loc[cols_per_portion].reset_index(),
    y="index",
    x="missing_percentage",
    kind="bar",
)

# Columns with name ending in "_unit"
sns.catplot(
    data=df_misspercentage.loc[cols_unit].reset_index(),
    y="index",
    x="missing_percentage",
    kind="bar",
)

# All other columns
sns.catplot(
    data=df_misspercentage.loc[cols_other].reset_index(),
    y="index",
    x="missing_percentage",
    kind="bar",
)

# %% [markdown]
# 3. Cleaning of missing values entries

# %% [markdown]
# a)
# We start by checking how many rows and columns are present before any removal, as well as how many null values are present overall.

# %%
print(f"// Dataframe size before addressing missing values")
print(f"Number of samples  : {len(df_clean_dup)}")
print(f"Number of features : {len(df_clean_dup.columns)}")
print(f"Number of nan's    : {df_clean_dup.isnull().sum().sum()}")

# %%
df_clean_dup_miss = df_clean_dup.dropna(how="all")
df_clean_dup_miss = df_clean_dup_miss.dropna(how="all", axis=1)

print(f"// Dataframe size after removing fully empty rows and columns")
print(f"Number of samples  : {len(df_clean_dup_miss)}")
print(f"Number of features : {len(df_clean_dup_miss.columns)}")
print(f"Number of nan's    : {df_clean_dup_miss.isnull().sum().sum()}")

# %% [markdown]
# b)
# Next we'll see how to best fill the nan values present in categorical columns ending with name ending in "_unit". For this we check the occurrence of difference values in each of these columns. The results below show that for the large majority there's a single value - 'unit' - which means this should be a good candidate for filling the nan entries. For the few entries where there are multiple values, the top occuring one is still 10x more frequent than the second.

# %%
df_clean_dup_miss

# %%
# configure pandas to not truncate the display of columns in the dataframe and show all of them
pd.set_option("display.max_columns", None)

# cycle through all '_unit' columns
for col in cols_unit:
    print("**************")

    # compute occurences of diffent values
    aux = df_clean_dup_miss[col].value_counts()
    print(aux)

    # fill nan with most frequent value for each column
    df_clean_dup_miss[col].fillna(value=aux.idxmax(), inplace=True)
    # print(f'{col}: {values}')

# %%
print(f"// Dataframe size after removing fully empty rows and columns")
print(f"Number of samples  : {len(df_clean_dup_miss)}")
print(f"Number of features : {len(df_clean_dup_miss.columns)}")
print(f"Number of nan's    : {df_clean_dup_miss.isnull().sum().sum()}")

# %% [markdown]
# c) Let’s fill up the missing values of a column using other columns that hold similar information. Apply this approach to the product_name_en column. Find columns that hold similar information and use them to replace the missing value in product_name_en

# %%
df_clean_dup_miss[cols_other].head()

# %% [markdown]
# Looking at the available features, the best candidates for finding the correct product name to fill 'product_name_en' would their name eith in 'product_name_fr' or 'product_name_de'.

# %%
df_clean_dup_miss["product_name_en"].fillna(
    df_clean_dup_miss["product_name_fr"], inplace=True
)
df_clean_dup_miss["product_name_en"].fillna(
    df_clean_dup_miss["product_name_de"], inplace=True
)

# %%
print(f"// Dataframe size after removing fully empty rows and columns")
print(f"Number of samples  : {len(df_clean_dup_miss)}")
print(f"Number of features : {len(df_clean_dup_miss.columns)}")
print(f"Number of nan's    : {df_clean_dup_miss.isnull().sum().sum()}")

# %% [markdown]
# d) Next we fill in the missing values of all the columns `..._per_hundred` with the value of 0, using the assumption that the missing values correspond to products which don't contain the specific ingredient. Before and after the filling, we count the number of missing values in each column. It was decided to apply this rule for all `..._per_hundred` columns and not just `iron_per_hundred` since we verified that in task C.3, when removing outlier, if this was not done many points were considered as outliers.

# %%
missing_before = df_clean_dup_miss["iron_per_hundred"].isnull().sum()
for col_per_hundred in cols_per_hundred:
    df_clean_dup_miss[col_per_hundred].fillna(value=0, inplace=True)
missing_after = df_clean_dup_miss["iron_per_hundred"].isnull().sum()

print(
    f"Missing values in 'iron_per_hundred' before fill is {missing_before}; and after fill is {missing_after}"
)


# %%
print(f"// Dataframe size after removing fully empty rows and columns")
print(f"Number of samples  : {len(df_clean_dup_miss)}")
print(f"Number of features : {len(df_clean_dup_miss.columns)}")
print(f"Number of nan's    : {df_clean_dup_miss.isnull().sum().sum()}")

# %% [markdown]
# e) Next we fill the missing values of the column `ingredient_missing` with the string `ingredient_missing`. As previously, before and after the filling, we count the number of missing values in this column

# %%
missing_before = df_clean_dup_miss["ingredients_en"].isnull().sum()
df_clean_dup_miss["ingredients_en"].fillna(value="ingredient_missing", inplace=True)
missing_after = df_clean_dup_miss["ingredients_en"].isnull().sum()

print(
    f"Missing values in 'ingredients_en' before fill is {missing_before}; and after fill is {missing_after}"
)


# %%
print(f"// Dataframe size after removing fully empty rows and columns")
print(f"Number of samples  : {len(df_clean_dup_miss)}")
print(f"Number of features : {len(df_clean_dup_miss.columns)}")
print(f"Number of nan's    : {df_clean_dup_miss.isnull().sum().sum()}")

# %% [markdown]
# f) Dropping columns with more than 95% of missing values.

# %%
df_len = len(df_clean_dup_miss)

cols_to_drop = [
    col
    for col in df_clean_dup_miss.columns
    if df_clean_dup_miss[col].isnull().sum() / df_len > 0.95
]

print(f"Number of columns with miss >95% is: {len(cols_to_drop)}")
df_clean_dup_miss.drop(cols_to_drop, axis=1, inplace=True)


# %%
print(f"// Dataframe size after removing fully empty rows and columns")
print(f"Number of samples  : {len(df_clean_dup_miss)}")
print(f"Number of features : {len(df_clean_dup_miss.columns)}")
print(f"Number of nan's    : {df_clean_dup_miss.isnull().sum().sum()}")

# %% [markdown]
# # C. Preliminary Exploratory Data Analysis (EDA)

# %%
df = df_clean_dup_miss.copy()

# %% [markdown]
# 1. We start by exploring the categorical variables in more detail. For this first, we get a reduced dataframe with categorial columns only

# %%
pd.set_option("display.max_rows", None)
df_cat = df.loc[:, df.dtypes == "object"].copy()
df_cat.nunique()

# %% [markdown]
# Next we want to know the proportion of samples in columns `country` and `unit`, for each categorical level .

# %%
pd.options.display.float_format = "{:.2f}".format

prop_country = df_cat["country"].value_counts() / len(df_cat["country"]) * 100
prop_country.to_frame(name="percentage")

# %%
prop_unit = df_cat["unit"].value_counts() / len(df_cat["unit"]) * 100
prop_unit.to_frame(name="percentage")

# %% [markdown]
# 2. Descriptive statistics and informative plots of the numerical variables

# %% [markdown]
# Below are several descriptive statistics (including min/max and mean, along with std and different percentils) of the numerical columns

# %%
df.describe()

# %%
df.hist(bins=30, figsize=(25, 15))
plt.show()

# %% [markdown]
# 3. Removal of outliers

# %% [markdown]
# The histogram of most variables show that a single or just a few bins around 0 appear to be populated. This is because even though the large majority of values are concentrated in a small range around 0, there are outlier points that are located far away from these values, which extends the histograms range resulting in small resolution of the binning.
#
# The same effect can also be observed from the descriptive statistics for the cases where the max values are much larger than the mean value plus the standard deviations.

# %% [markdown]
# In order to remove the outlier points we restrict the value for each ´..._per_hundred´ column using indicated values. To facilitate this task we first print all columns units and verify that each ´..._unit´ columns contains a single unit for all entries. Which is the case since the if condition in following code block is never executed.

# %%
for col_per_hundred in cols_per_hundred:
    col_unit = col_per_hundred[:-12] + "_unit"
    print(f"{col_per_hundred} has unit: {df[col_unit].unique()[0]}")
    if len(df[col_unit].unique()) > 1:
        print(f"Col {col_unit} contains more than 1 unit")


# %%
# auxiliary function used to return outlier exclusion rules for different units
def outlier_rule(df, col_per_hundred, unit):
    if unit == "g":
        rule = (df[col_per_hundred] >= 0) & (df[col_per_hundred] <= 100)
    elif unit == "mg":
        rule = (df[col_per_hundred] >= 0) & (df[col_per_hundred] <= 1e5)
    elif unit == "µg":
        rule = (df[col_per_hundred] >= 0) & (df[col_per_hundred] <= 1e8)
    elif unit == "kJ":
        rule = (df[col_per_hundred] >= 0) & (df[col_per_hundred] <= 3700)
    elif unit == "kCal":
        rule = (df[col_per_hundred] >= 0) & (df[col_per_hundred] <= 885)
    elif unit == "IU":
        if "vitamin_a" in col_per_hundred:
            rule = (df[col_per_hundred] >= 0) & (df[col_per_hundred] <= 3.3e8)
        elif "vitamin_d" in col_per_hundred:
            rule = (df[col_per_hundred] >= 0) & (df[col_per_hundred] <= 4e9)
        else:
            print(f"Unit not covered for {col_per_hundred} and {unit}")

    else:
        print(f"Unit not covered for {col_per_hundred} and {unit}")

    return rule


# %%
df_keep = df.copy()

for col_per_hundred in cols_per_hundred:
    unit = df[f"{col_per_hundred[:-12]}_unit"].unique()[0]

    filter_rule = outlier_rule(df, col_per_hundred, unit)
    df_keep = df_keep.loc[filter_rule]

# %% [markdown]
# We verify how many rows were removed by entries where the value for at least of the `.._per_hundred` columns was out of range.

# %%
print(f"Initial dataframe length:               {len(df)}")
print(f"Dataframe length after outlier removal: {len(df_keep)}")

# %% [markdown]
# Next we derive again the statiscal information for the numerical columsn as well as the histograms. We observe from the statistical table that the max values are now according to the ranges we have previously defined for the different units. The same can be observed in the histograms. E.g. for the `energy_per_hundred` in `kJ` a destribution in the range `[0; 3700]` is clearly visible

# %%
df_keep.describe()

# %%
df_keep.hist(bins=30, figsize=(25, 15))
plt.show()

# %% [markdown]
# 4. Columsn that are dependent on one another

# %% [markdown]
# a)
# The columns `energy_per_hundred` in `kJ` and the `energy_kcal_per_hundred` in `kcal` should be linearly related. But from observing the plot below, we see that this is not the case for all entries. For many of inconsistent entries we see that one of values was probably missing and was set to zero. For these we can assume that value in the other column is correct and we can use this to compute the other, knowing that `1kcal = 4.184kJ`.

# %%
df_keep.plot(
    kind="scatter", x="energy_per_hundred", y="energy_kcal_per_hundred", color="r"
)
plt.show()

# %% [markdown]
# In order to address this problem we'll create a new column `energy_fix_per_hundred` in `kJ` which will contain the energy values we deem to be correct. As a first step we'll populate this new column with: the values from `energy_per_hundred`; and for the cases where these are zero we'll convert from the `kcal` to `kJ` using the values in `energy_kcal_per_hundred`. As shown in the scatter plot below, the points where `energy_per_hundred` were zero, have now been corrected.


# %%
def fix_energy(x):
    if x["energy_per_hundred"] == 0 and x["energy_kcal_per_hundred"] != 0:
        return x["energy_kcal_per_hundred"] * 4.18
    elif x["energy_per_hundred"] != 0 and x["energy_kcal_per_hundred"] == 0:
        return x["energy_per_hundred"]
    else:
        return x["energy_per_hundred"]


df_keep["energy_fix_per_hundred"] = df_keep.apply(fix_energy, axis=1)

df_keep.plot(
    kind="scatter", x="energy_fix_per_hundred", y="energy_kcal_per_hundred", color="r"
)
plt.show()

# %% [markdown]
# We can still observe two cases where the data points are not related as they should.
# - When `energy_kcal_per_hundred` is zero: we deciede to ignore this since we assume the values in `kcaç` were missing but the values in `kJ` are ok.
# - When `energy_kcal_per_hundred` and `energy_per_hundred` don't follow the expected relation `1kcal = 4.184kJ`: we decide to not use these entries since we have no means of knowing which value is correct.
#
# To implement these new rules we rerun the previous code block to generate `energy_fix_per_hundred` in `kJ`, replacing with `np.nan` whenever the there's a discrepency of `>20%` in the expected relation.


# %%
def fix_energy(x):
    if x["energy_per_hundred"] == 0 and x["energy_kcal_per_hundred"] != 0:
        return x["energy_kcal_per_hundred"] * 4.18
    elif x["energy_per_hundred"] != 0 and x["energy_kcal_per_hundred"] == 0:
        return x["energy_per_hundred"]
    elif (
        np.abs(x["energy_per_hundred"] - x["energy_kcal_per_hundred"] * 4.18)
        / x["energy_per_hundred"]
        > 0.2
    ):
        return np.nan
    else:
        return x["energy_per_hundred"]


df_keep["energy_fix_per_hundred"] = df_keep.apply(fix_energy, axis=1)

df_keep.plot(
    kind="scatter", x="energy_fix_per_hundred", y="energy_kcal_per_hundred", color="r"
)
plt.show()

# %% [markdown]
# Next we check how many values in the `energy_fix_per_hundred` column are "eliminitaed" that is converted to `np.nan`

# %%
n_total = len(df_keep)
n_energy_fix = n_total - df_keep["energy_fix_per_hundred"].isnull().sum()
print(f"Number of null values is {n_energy_fix} out of {n_total}")

# %% [markdown]
# b)
# Next we'll check potential inconsistencies between the total energy now represented by column `energy_fix_per_hundred` and the sum of energy content per macronutrient obtained from the following formula
#
# ```energy_per_hundred [kJ]= 37* fat_per_hundred[g]+ 17*(protein_per_hundred[g] + carbohydrates_per_hundred[g])```
#
# - We start by verifying the units of the macronutrient columns
# - Then we create a new energy column `energy_macronutrient_per_hundred` obtained as the total of energy per macronutrient
#

# %%
for col in ["fat_per_hundred", "protein_per_hundred", "carbohydrates_per_hundred"]:
    col_unit = f"{col[:-12]}_unit"
    print(f"Units of {col} are: {df_keep[col_unit].unique()[0]}")


# %%
def macronutrient_energy(x):
    energy_macro = (
        37 * x["fat_per_hundred"]
        + 17 * x["protein_per_hundred"]
        + 17 * x["carbohydrates_per_hundred"]
    )

    return energy_macro


df_keep["energy_macronutrient_per_hundred"] = df_keep.apply(
    macronutrient_energy, axis=1
)

df_keep.plot(
    kind="scatter",
    x="energy_fix_per_hundred",
    y="energy_macronutrient_per_hundred",
    color="r",
)
plt.axis("scaled")
plt.show()

# %% [markdown]
# Observing the previous scatter plot, we see that for many datapoints both ways of obtaining the energy value don't really match. In order to address this, we next modify the calcualtion of the total energy from the different macronutrients, such that when this differs by no more than `20%`, we fill the `energy_macronutrient_per_hundred` with a null value.


# %%
def macronutrient_energy(x):
    energy_macro = (
        37 * x["fat_per_hundred"]
        + 17 * x["protein_per_hundred"]
        + 17 * x["carbohydrates_per_hundred"]
    )

    if x["energy_fix_per_hundred"] > 0:
        if (
            np.abs(energy_macro - x["energy_fix_per_hundred"])
            / x["energy_fix_per_hundred"]
            < 0.2
        ):
            return energy_macro
        else:
            return np.nan
    else:
        return energy_macro


df_keep["energy_macronutrient_per_hundred"] = df_keep.apply(
    macronutrient_energy, axis=1
)

df_keep.plot(
    kind="scatter",
    x="energy_fix_per_hundred",
    y="energy_macronutrient_per_hundred",
    color="r",
)
plt.axis("scaled")
plt.show()

# %% [markdown]
# Next we check how many values in the `energy_fix_per_hundred` column are "eliminated", that is populated with null value.

# %%
n_total = len(df_keep)
n_macronutrient = n_total - df_keep["energy_macronutrient_per_hundred"].isnull().sum()
print(f"Number of null values is {n_energy_fix} out of {n_total}")

# %% [markdown]
# c)
# Checking if macronutrient content adds up to total mass of product.


# %%
def total_macronutrient_mass(x):
    total_mass = (
        x["fat_per_hundred"]
        + x["protein_per_hundred"]
        + x["carbohydrates_per_hundred"]
        + x["fiber_per_hundred"]
    )

    return total_mass


df_keep["macronutrient_mass_per_hundred"] = df_keep.apply(
    total_macronutrient_mass, axis=1
)

ax = df_keep.plot(kind="hist", y="macronutrient_mass_per_hundred", color="r", bins=100)
ax.set_xlabel("Sum of macronutrient mass content per 100g [g]")
plt.show()

# %% [markdown]
# We observe that the maximum of the sum of macronutrients is 100g which is expected consistent. However, for many entries this sum is smaller than 100g, this could be due to the content of water, alcohol, or eventually micronutrients which are not being taken into account

# %% [markdown]
# 5. Address outliers in few `..._per_hundred` columns

# %%
df_keep[cols_per_hundred].hist(bins=30, figsize=(25, 15))
plt.show()

# %%
df_keep[cols_per_hundred].describe()

# %% [markdown]
# From observing the histogram plots and statistics of all `_per_hundred` variables, we see that actually for most of them, there are still "extreme" outlier points present in the data. E.g. `calcium_per_hundred`, `cholesterol_per_hundred`, `copper_cy_per_hundred`, etc... In order to address these, we will apply removal strategy based on the Z-score. We next exemplify this for a few columns

# %%
df_remove_outliers = df_keep.copy()

for col in ["calcium_per_hundred", "cholesterol_per_hundred", "copper_cu_per_hundred"]:
    zfilter_col = np.abs(df_keep[col] - df_keep[col].mean()) > (1 * df_keep[col].std())

    df_remove_outliers = df_remove_outliers.loc[~zfilter_col]

# %%
df_remove_outliers[cols_per_hundred].hist(bins=30, figsize=(25, 15))
plt.show()

# %%
df_remove_outliers[cols_per_hundred].describe()

# %%
len(df_keep)

# %%
len(df_remove_outliers)

# %% [markdown]
# Next we check how many rows have been removed by removing the outliers values in the `energy_fix_per_hundred` column are "eliminated", that is populated with null value.

# %%
n_initial = len(df_keep)
n_final = len(df_remove_outliers)
print(
    f"Number of entries after removing extreme outliers is {n_final} out of {n_initial} in the beginning."
)

# %% [markdown]
# # D. EDA: Text data

# %%
import re

# %%
ingredients = df_remove_outliers["ingredients_en"].copy()

# %%
ingredients

# %% [markdown]
# 1. Product with the longest ingredient list

# %%
number_ingredients = ingredients.apply(lambda x: len(x.split(",")))
number_ingredients.sort_values(ascending=False).head()

# %%
df_remove_outliers[["product_name_en", "ingredients_en"]]

# %%
df_remove_outliers.loc[350670, "product_name_en"]


# %%
def preprocess_text(text):
    """
    This function preprocesses raw samples of text:
    - Converts to lowercase
    - Replaces common punctuation marks with whitespace
    - Removes stop words
    - Splits text on whitespace

    INPUT:
    - text: "raw" text (string)

    OUTPUT:
    - processed_sample_tokens: list of tokens (list of strings)
    """

    # Convert to lowercase
    clean_text = text.lower()

    # Replace single hyphens with whitespace
    clean_text = re.sub(r",\s*", "_", clean_text)

    # Replace common punctuation marks with whitespace
    clean_text = re.sub(r"[,.()]", " ", clean_text)

    # Replace single hyphens with whitespace
    clean_text = re.sub(r"\s-\s", " ", clean_text)

    # Remove stop words and split on whitespace
    # processed_sample_tokens = [tok for tok in clean_text.split() if tok not in stopwords_english]

    return clean_text


# %%
ingredients_clean = ingredients.apply(lambda x: preprocess_text(x))

# Show an example of preprocessed and tokenized text
ingredients_clean.sample()

# %%
ingredients_clean.head()

# %% [markdown]
# 2. Products with shortest ingredient list

# %%


# %% [markdown]
# 3. Most frequent ingredients in products

# %%
