## Analysis of Real Estate Data for Real Estate
## Before Running, Make sure to run real_estate_cleaning_Roman.py

# %% Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
#from linearmodels.iv import IV2SLS
path = "/Users/romansielewicz/Google Drive/Fall2019/real_estate_Project/Datasets/"

origin = pd.read_csv(path + "combo.csv", low_memory = False)
origin.shape


# %% Cleaning variables
# Variables to clean
# Star Rating to int
origin["Star Rating"] = origin["Star Rating"].apply(str)
origin["Star Rating"] = origin["Star Rating"].apply(lambda x: x[0])
origin["Star Rating"] = origin["Star Rating"].replace("n", np.nan)
origin["Star Rating"] = origin["Star Rating"].apply(float)
origin["Star4"] = origin["Star Rating"] - 1
# Last Sale Price to int
origin["Last Sale Price"] = origin["Last Sale Price"].str.replace(",", "")
origin["Last Sale Price"] = pd.to_numeric(origin["Last Sale Price"], errors='coerce')
# Capitalizing All State names
origin["State"] = origin["State"].apply(lambda x: x.upper())
origin["State"] = origin["State"].apply(lambda x: x[:2])
# Stringifying Amenities
origin["Amenities"] = origin["Amenities"].apply(str)
# Stringifying Owner City State zip
origin["Owner City State Zip"] = origin["Owner City State Zip"].apply(str)
# Converting Rental Prices to to numeric Variables
unitList = []
for i in ["One", "Two", "Three", "Four"]:
    x = i+ " Bedroom Effective Rent/Unit"
    unitList.append(x)
unitList = ["Studio Effective Rent/Unit"] + unitList
for i in unitList:
    origin[i] = (origin[i].apply(str)).str.replace(",", "")
    origin[i] = origin[i].replace("nan", np.nan)
    origin[i] = origin[i].apply(float)

bedroomCols = ["Number Of Studios"]
for i in range(4):
    item = "Number Of " + str(i + 1) + " Bedrooms"
    bedroomCols.append(item)
origin[bedroomCols + unitList].head()

# %% Creating Rental Revenue Column
for i in range(len(unitList)):
    origin["Rent_Unit_" + str(i)] = origin[unitList[i]]

rent_unit_list = []
for i in range(len(bedroomCols)):
    item = "Rent_Unit_" + str(i)
    rent_unit_list.append(item)
rent_unit_list
origin[bedroomCols + unitList + rent_unit_list].head()

for i in range(len(bedroomCols)):
    origin.loc[origin[bedroomCols[i]] == 0, rent_unit_list[i]] = 0
origin[bedroomCols + unitList + rent_unit_list].head()
## Retrying Rental Revenue column
origin["Rental_Revenue"] =  origin["Rent_Unit_0"]*origin["Number Of Studios"] \
                            + origin["Rent_Unit_1"]*origin["Number Of 1 Bedrooms"] \
                            + origin["Rent_Unit_2"]*origin["Number Of 2 Bedrooms"] \
                            + origin["Rent_Unit_3"]*origin["Number Of 3 Bedrooms"] \
                            + origin["Rent_Unit_4"]*origin["Number Of 4 Bedrooms"]

origin[bedroomCols + unitList + rent_unit_list + ["Rental_Revenue"]].head()

# %% Creating Owner_City, Owner_State, Owner_Zip
# Defining functions to extract city, state, and zip
def cityFinder(x):
    output = ""
    if "," in x:
        index = x.index(",")
        output = x[:index]
    return output
cityFinder("Irvine, CA 926147209")

def stateFinder(x):
    output = ""
    if "," in x:
        index = x.index(",")
        output = x[index + 2:index + 4].upper()
    return output

stateFinder("Irvine, CA 926147209")

def zipFinder(x):
    zip = ""
    if "," in x:
        index = x.index(",")
        fullZip = x[index + 5:]
        zip = fullZip[0:5]
    return(zip)

zipFinder("Irvine, CA 926147209")
# applying functions to extract city state and zip
origin["Owner_City"] = origin["Owner City State Zip"].apply(cityFinder)
origin["Owner_State"] = origin["Owner City State Zip"].apply(stateFinder)
origin["Owner_Zip"] = origin["Owner City State Zip"].apply(zipFinder)

# Creating a Binary Variable to show if an owner is out of state
origin["Out_Of_State"] = (origin["State"] == origin["Owner_State"]).apply(int)

# Creating Age and Years since Renovation variables
origin["Building_Age"] = 2019 - origin["Year Built"]
origin["Years_Since_Renovation"] = 2019 - origin["Year Renovated"]

# Creating Building Material Dummy variables
materialList = list(origin["Construction Material"].unique())
materialList = materialList[1:]
for i in materialList:
    origin[i] = origin["Construction Material"].apply(lambda x: 1 if i == x else 0)

# %% Separating Amenities into distinct categories
# Note "Features" and "Amenities" in the original dataframe are exactly the same
# proof below
lf = origin[origin["Amenities"] != origin["Features"]][["Amenities", "Features"]]
def nanFinder(df):
    output = {}
    for i in df.columns:
        count = 0
        column = df[i].apply(str)
        for j in column:
            if j != 'nan':
                count += 1
        output.update({i:count})
    return output

lfColDict = nanFinder(lf)
lfColDict

#### --------------------------------------- ###
# %% md **Subsetting for Sales Data**

# %% Creating df, the dataframe we will use to analyze sale price
# Subsetting for only columns with valid Sale Price data
df = origin.dropna(axis = 0, subset = ["Sale Price"])
df = df[df["Sale Price"] > 10000]
df.shape

# %% Creating rentdf, the dataframe we will use to analyze Rental Revenue
# Subsetting for only columns with valid Rental Revenue data
rentdf = origin.dropna(axis = 0, subset = ["Rental_Revenue"])
rentdf.shape

# Finding each of the actual Amenities
df["Amenities"][0] # to give a template of entries in the amenities column

amenityBigList = []
for i in df["Amenities"]:
    item = i.split(", ")
    for j in item:
            amenityBigList.append(j)
amenityList = list(pd.Series(amenityBigList).unique())
len(amenityList)

amenityFreq = {}
for i in amenityList:
    count = amenityBigList.count(i)
    amenityFreq.update({i:count})

amenityFreq = pd.DataFrame(list(amenityFreq.items()), columns=['Amenity', 'Count'])
amenityFreq = amenityFreq.sort_values(by = "Count", ascending = False).reset_index(drop = True)
amenityFreq = amenityFreq[amenityFreq.Count > 2000]

amenities_relevant = list(amenityFreq["Amenity"])
amenities_relevant = [  \
                        'Fitness Center',
                        'Laundry Facilities',
                        'Clubhouse',
                        'Property Manager on Site',
                        'Business Center',
                        'Playground',
                        'Grill',
                        'Maintenance on site',
                        'Picnic Area',
                        'Gated',
                        'Tennis Court',
                        'Package Service',
                        'Controlled Access',
                        'Sundeck',
                        'Laundry Service',
                        'Storage Space',
                        'Spa',
                        'Breakfast/Coffee Concierge',
                        'Courtyard',
                        'Volleyball Court',
                        'Pet Play Area' \
                        ] # amenities_relevant is a list of all relevant amenities to make dummy vars

# Creating Dummy Variables for each amenity

for i in amenityFreq["Amenity"]:
    df[i] = df["Amenities"].apply(lambda x: 1 if i in x else 0)

for i in list(amenityFreq["Amenity"]):
    rentdf[i] = rentdf["Amenities"].apply(lambda x: 1 if i in x else 0)

# %% Subsetting Variables to analyze
#Choosing Variables to Analyze
indepedentVars = [  "Year Built",
                    "Year Renovated",
                    "Construction Material",
                    #"Features",
                    #"Amenities",
                    "Number Of Parking Spaces",
                    "Parking Spaces/Unit",
                    "Star Rating",
                    "Owner Name",
                    "Owner Address",
                    "Owner City State Zip",
                    "Last Sale Date",
                    "Building Class",
                    "Building Park",
                    "Sale Date",
                    "Market",
                    "Submarket Code",
                    "Submarket Cluster",
                    "Zoning"
                    ]

priceVars = [       "Sale Price",
                    "Price Per SF",
                    "Price Per SF (Net)",
                    "Price Per Unit",
                    "Last Sale Price",
                    "Price Per AC Land",
                    #"Asking Price"
                    ]

minidf = df[indepedentVars + priceVars]
minidf.head()
minidf[minidf["Sale Price"] == minidf["Last Sale Price"]].shape

# %% md **Preliminary Scatterplots**

# %% Sale Price by year built
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.scatter("Year Built", "Sale Price", data = df)
plt.xlim(1935, 2020)
plt.ylim(0, 100000000)
plt.xlabel("Year Built")
plt.ylabel("Sale Price")
plt.title("Sale Price by Year Built")

plt.show()

# %% Sale Price by year Renovated
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.scatter("Year Renovated", "Sale Price", data = df)
plt.ylim(0, 150000000)
plt.xlabel("Year Renovated")
plt.ylabel("Sale Price")
plt.title("Sale Price by Year Renovated")

plt.show()

# %% Creating new dataframe with average price per square foot by state
states = df[["State", "Sale Price"]].groupby("State").mean()
states.reset_index(inplace = True)
states.columns = ["State", "Avg_Sale_Price"]
states

# Avg Sale Price by Market bar chart
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.bar("State", "Avg_Sale_Price", data = states)
plt.ylim(0, 30000000)
plt.xlabel("State")
plt.ylabel("Avg. Sale Price")
plt.title("Avg. Sale Price by State")

plt.show()

# %% Price per square unit by parking spaces per unit
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.scatter("Parking Spaces/Unit", "Price Per Unit", data = df)

plt.xlim(0, 5)
plt.ylim(0, 400000)
plt.xlabel("Parking Spaces per Unit")
plt.ylabel("Price Per Unit")
plt.title("Avg. Price/Sqft by Parking Space")

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

plt.show()

# %% Sale Price by parking spaces
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.scatter("Parking Spaces/Unit", "Sale Price", data = df)

plt.xlim(0, 4)
plt.ylim(0, 150000000)
plt.xlabel("Parking Spaces/Unit")
plt.ylabel("Sale Price")
plt.title("Sale Price by Parking Space Per Unit")

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

plt.show()

# %% Price Per SF vs Star Rating
plt.figure(figsize = (10,8))

ax = plt.gca()
plt.bar("Star Rating", "Price Per SF (Net)", data = ratings)
plt.ylim(0, 250000)
plt.xlabel("Star Rating")
plt.ylabel("Avg. Price Per SF (Net)")
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

plt.show()


# %% Correlation Matrix
#Choosing Variables to Analyze
vars =  [ \
        "Out_Of_State",
        "Building_Age",
        "Years_Since_Renovation",
        "Number Of Parking Spaces",
        "Parking Spaces/Unit",
        "Star4",
        ] \
        + list(amenityFreq["Amenity"]) \
        + materialList
# Generating Correlation Matrix
corrdf_sale = df[["Sale Price"] + vars]
corrdf_sale.corr()

# %% Linear Regression Analysis
reg1 = sm.OLS(endog=df["Sale Price"], exog=df[vars], \
    missing='drop')
results1 = reg1.fit()
print(results1.summary())

# Retrying Regression Model with fewer independent variables
vars =  [
        "Out_Of_State",
        "Building_Age",
        #"Years_Since_Renovation",
        "Number Of Parking Spaces",
        "Parking Spaces/Unit",
        "Star4",
        "Masonry",
        "Reinforced Concrete",
        "Wood Frame",
        "Fitness Center",
        "Property Manager on Site",
        "Business Center", "Playground",
        "Maintenance on site",
        "Tennis Court",
        "Package Service",
        "Laundry Service",
        "Storage Space",
        "Number Of Units", # start of Team Variables
        "Vacancy %",
        ]
len(vars)
regSale = sm.OLS(endog=df["Sale Price"], exog=df[vars], \
    missing='drop')
resultsSale = regSale.fit()
print(resultsSale.summary())

regSaleLog = sm.OLS(endog=log(df["Sale Price"]), exog=df[vars], \
    missing='drop')
resultsSaleLog = regSaleLog.fit()
print(resultsSaleLog.summary())

df[["Sale Price"] + vars]
len(vars)

# %% Choosing Several Markets with high concentration of sales to focus on
topMarkets =    [
                "Atlanta",
                "Dallas/Ft Worth",
                "Tampa/St Petersburg",
                "Houston",
                "Orlando",
                "South Florida",
                "Austin"
                ]

for i in topMarkets:
    cityName = i
    if "/" in cityName:
        cityName = cityName[:cityName.index("/")]
    df[df["Market"] == i].to_csv(path + cityName + ".csv", index = False)

# %% Atlanta Map of Sales
Atlanta = df[df["Market"] == "Atlanta"]
BBox = (Atlanta.Longitude.min(),   Atlanta.Longitude.max(),
         Atlanta.Latitude.min(), Atlanta.Latitude.max())

BBox
image_loc = "/Users/romansielewicz/Google Drive/Fall2019/real_estate_Project/Images/"

Atlanta_m = plt.imread(image_loc + "Atlanta_map.png")

# Plotting on map of Atlanta

fig, ax = plt.subplots(figsize = (20, 15))

#c = Atlanta["Sale Price"]
c = Atlanta["Price Per SF"]

ax.scatter( Atlanta["Longitude"], Atlanta["Latitude"],
            zorder=1, alpha= 0.2, c=c, s=10,
            cmap = 'seismic')
ax.set_title('Commercial Real Estate Sales in Atlanta')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(Atlanta_m, zorder=0, extent = BBox, aspect= 'equal')

plt.show()

# %% md **Correlation Matrixes for Rental prices**

# %% Automating Regression Code
for i in range(len(unitList)):
    print(  "# %" + "% " + str(i) + " Bedroom Correlation Matrix\n" \
            + "corrdf" + str(i) + " = rentdf[['" + unitList[i] + "'] + vars][rentdf['" + unitList[i] + "'] != 0]\n" \
            + "corrdf" + str(i) + ".corr()")

# %% 0 Bedroom Correlation Matrix
corrdf0 = rentdf[['Studio Effective Rent/Unit'] + vars][rentdf['Studio Effective Rent/Unit'] != 0]
corrdf0.corr()

# %% 1 Bedroom Correlation Matrix
corrdf1 = rentdf[['One Bedroom Effective Rent/Unit'] + vars][rentdf['One Bedroom Effective Rent/Unit'] != 0]
corrdf1.corr()
# %% 2 Bedroom Correlation Matrix
corrdf2 = rentdf[['Two Bedroom Effective Rent/Unit'] + vars][rentdf['Two Bedroom Effective Rent/Unit'] != 0]
corrdf2.corr()
# %% 3 Bedroom Correlation Matrix
corrdf3 = rentdf[['Three Bedroom Effective Rent/Unit'] + vars][rentdf['Three Bedroom Effective Rent/Unit'] != 0]
corrdf3.corr()
# %% 4 Bedroom Correlation Matrix
corrdf4 = rentdf[['Four Bedroom Effective Rent/Unit'] + vars][rentdf['Four Bedroom Effective Rent/Unit'] != 0]
corrdf4.corr()

# %% md Regression Analysis for Rent variables

# %% Linear Regression Analysis
# Automating Regression Code
for i in range(len(unitList)):
    print(  "# %" + "% " + str(i) + " Bedroom Regression Model\n" \
            + "reg" + str(i) + " = sm.OLS(endog=df['" + unitList[i] + "'], \\" + "\n" \
            + "exog = df[vars], \\" + "\n" \
            + "missing='drop')" + "\n" \
            + "results" + str(i) + " = reg" + str(i) + ".fit()\n" \
            + "print(results" + str(i) +".summary())\n")

# %% Studio Regression Model
reg0 = sm.OLS(endog=df['Studio Effective Rent/Unit'], \
exog = df[vars], \
missing='drop')
results0 = reg0.fit()
print(results0.summary())

# %% 1 Bedroom Regression Model
reg1 = sm.OLS(endog=df['One Bedroom Effective Rent/Unit'], \
exog = df[vars], \
missing='drop')
results1 = reg1.fit()
print(results1.summary())

# %% 2 Bedroom Regression Model
reg2 = sm.OLS(endog=df['Two Bedroom Effective Rent/Unit'], \
exog = df[vars], \
missing='drop')
results2 = reg2.fit()
print(results2.summary())

# %% 3 Bedroom Regression Model
reg3 = sm.OLS(endog=df['Three Bedroom Effective Rent/Unit'], \
exog = df[vars], \
missing='drop')
results3 = reg3.fit()
print(results3.summary())

# %% 4 Bedroom Regression Model
reg4 = sm.OLS(endog=df['Four Bedroom Effective Rent/Unit'], \
exog = df[vars], \
missing='drop')
results4 = reg4.fit()
print(results4.summary())

# %% Analysis for Rental Revenue Variable
vars
corrdfRental = df[['Rental_Revenue'] + vars]
corrdfRental.corr()

regRental = sm.OLS(endog=df['Rental_Revenue'], \
exog = df[vars], \
missing='drop')
resultsRental = regRental.fit()
print(resultsRental.summary())
