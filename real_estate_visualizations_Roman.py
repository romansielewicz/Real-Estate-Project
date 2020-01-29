## Analysis of Real Estate Data for real_estate
## Before Running, Make sure to run real_estate_cleaning_Roman.py

# %% Importing Packages
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import seaborn as sns
#from linearmodels.iv import IV2SLS
path = "/Users/romansielewicz/Google Drive/Fall2019/real_estate_Project/Datasets/"

# %% Importing Dataset
origin = pd.read_csv(path + "origin.csv", low_memory = False)
origin.shape

# %% Creating df, the dataframe we will use to analyze sale price
# Subsetting for only columns with valid Sale Price data
df = origin.dropna(axis = 0, subset = ["Sale Price"])
df = df[df["Sale Price"] > 1000000]
df.shape

# %% Creating rentdf, the dataframe we will use to analyze Rental Revenue
# Subsetting for only columns with valid Rental Revenue data
rentdf = origin.dropna(axis = 0, subset = ["Rental_Revenue"])
rentdf = rentdf[rentdf["Rental_Revenue"] > 10000]
rentdf.shape

# %% Histogram of sale price
image_loc = "/Users/romansielewicz/Google Drive/Fall2019/real_estate_Project/Images/"
t = np.arange(0, 100000000, 20000000)
tlabs = [str(int(i/1000000))+ "M" for i in t]

plt.figure(figsize = (12,8))

ax = plt.gca()
plt.hist("Sale Price", data = df, bins = 300)
plt.xlim(0, 100000000)
plt.axvline(x = df["Sale Price"].median(), color = "red")
plt.xticks(t, tlabs)
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.title("Distribution of Sale Price")
plt.savefig(image_loc + "Sale Price Dist.png")

plt.show()
df["Sale Price"].median()


# %% Histogram of price per sf
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.boxplot("Price Per SF", data = df)
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.title("Distribution of Sale Price")

plt.show()

# %% md **Preliminary Scatterplots**

# %% codecell
for i in df.columns:
    if "Sale" in i:
        print(i)

# %% Sale Price by year built

yticks = np.arange(0, 100000000, 20000000)

plt.figure(figsize = (12,8))

ax = plt.gca()
plt.scatter("Building_Age", "Sale Price", data = df, alpha = 0.2)
plt.xlim(0, 70)
plt.ylim(0, 80000000)
plt.yticks(yticks, [str(int(i/1000000)) + "M" for i in yticks])
plt.xlabel("Building Age", size = 20)
plt.ylabel("Sale Price", size = 20)
plt.title("Sale Price by Building Age", size = 30)
plt.savefig(image_loc + "Sale Price by Building Age.png")

plt.show()

# %% Sale Price by year built

yticks = np.arange(0, 100000000, 20000000)

plt.figure(figsize = (18,8))

ax = plt.gca()
plt.scatter("Building_Age", "Sale Price", data = df, alpha = 0.6)
plt.xlim(0, 80)
plt.ylim(0, 80000000)
plt.yticks(yticks, [str(int(i/1000000)) + "M" for i in yticks])
plt.xlabel("Building Age", size = 20)
plt.ylabel("Sale Price", size = 20)
plt.title("Sale Price by Building Age", size = 30)
plt.savefig(image_loc + "Sale Price by Building Age.png")

plt.show()

# %% Sale Price by year Renovated (Seaborn)
age = sns.regplot("Building_Age", "Sale Price", data = df)
axes = age.axes
axes.set_ylim(0,80000000)
axes.set_ylim(0,80)

# %% Creating new dataframe with average price per square foot by state
states = df[["State", "Sale Price"]].groupby("State").mean()
states.reset_index(inplace = True)
states.columns = ["State", "Avg_Sale_Price"]
states

# Avg Sale Price by Market bar chart
plt.figure(figsize = (15, 12))

ax = plt.gca()
plt.bar("State", "Avg_Sale_Price", data = states)
plt.ylim(0, 30000000)
plt.xlabel("State", size = 25)
plt.ylabel("Avg. Sale Price", size = 25)
plt.title("Avg. Sale Price by State", size = 30)
plt.savefig(image_loc + "Avg Price by State.png")
plt.savefig(image_loc + "Avg Sale Price by State.png")

plt.show()

affordableDict = {  "Rent Restricted": df[df["Affordable_Housing"] == 1]["Sale Price"].mean(),
                    "Unrestricted": df[df["Affordable_Housing"] == 0]["Sale Price"].mean()}
affordableDict


# %% Price per square unit by parking spaces per unit
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.scatter("Parking Spaces/Unit", "Price Per Unit", data = df)

plt.xlim(0, 5)
plt.ylim(0, 400000)
plt.xlabel("Parking Spaces per Unit")
plt.ylabel("Price Per Unit")
plt.title("Avg. Price/Sqft by Parking Space")
plt.savefig(image_loc + "Price per sqft by parking spaces.png")

plt.show()

# %% Sale Price by parking spaces
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.scatter("Number Of Parking Spaces", "Sale Price", data = df)

plt.xlim(0, 1000)
plt.ylim(0, 15000000)
plt.xlabel("Total Parking Spaces")
plt.ylabel("Sale Price")
plt.title("Sale Price by Total Parking Spaces")
plt.savefig(image_loc + "Sale Price by number of parking spaces.png")

plt.show()

# %% Sale Price by parking spaces per unit
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.scatter("Parking Spaces/Unit", "Sale Price", data = df)

plt.xlim(0, 4)
plt.ylim(0, 150000000)
plt.xlabel("Parking Spaces/Unit")
plt.ylabel("Sale Price")
plt.title("Sale Price by Parking Space Per Unit")
plt.savefig(image_loc + "Sale Price by Parking space per unit.png")

plt.show()

# %% Creating new dataframe with average price per unit vs star Rating
ratings = df[[   "Star Rating", "Price Per SF", "Price Per Unit", \
                "Sale Price", "Price Per SF (Net)"]].groupby("Star Rating").mean()
ratings.reset_index(inplace = True)
ratings

# Price Per Unit vs Star Rating
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.bar("Star Rating", "Price Per Unit", data = ratings)
plt.ylim(0, 250000)
plt.xlabel("Star Rating")
plt.ylabel("Avg. Price Per Unit")
plt.title("Avg. Price/Unit by Star Rating")
plt.savefig(image_loc + "Avg Price per unit by star rating.png")

plt.show()

# %% Price Per SF vs Star Rating
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.bar("Star Rating", "Price Per SF", data = ratings)
plt.ylim(0, 250000)
plt.xlabel("Star Rating")
plt.ylabel("Avg. Price Per SF")
plt.title("Avg. Price/Sqft by Star Rating")

plt.show()

# %% Sale Price vs Star Rating
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.bar("Star Rating", "Sale Price", data = ratings)
plt.ylim(0, 100000000)
plt.xlabel("Star Rating")
plt.ylabel("Avg Sale Price")
plt.title("Avg. Sale Price by Star Rating")
plt.savefig(image_loc + "Avg Sale Price by star rating.png")

plt.show()

# %% Creating Dictionary with 5 items, 1 for each star rating
starDict = {}
for i in range(1,6):
    starDict[str(i) + " Star"] = list(df[df["Star Rating"] == i]["Sale Price"][:])

starDict

# %% Sale Price vs Star Rating
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.boxplot(list(starDict.values()), labels = list(starDict.keys()))
plt.ylim(0, 200000000)
plt.xlabel("Star Rating")
plt.ylabel("Avg Sale Price")
plt.title("Avg. Sale Price by Star Rating")
plt.savefig(image_loc + "Star Rating Boxplot.png")

plt.show()
