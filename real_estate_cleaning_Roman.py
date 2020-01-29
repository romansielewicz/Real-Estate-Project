# %% Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = "/Users/romansielewicz/Google Drive/Fall2019/real_estate_Project/Datasets/"

# %% Importing Datasets
property = pd.read_csv(path + "property_data.csv", low_memory = False)
sales = pd.read_csv(path + "sales_data.csv", low_memory = False)
key = pd.read_csv(path + "key.csv", low_memory = False)

property.shape
sales.shape
key.shape

# %% Making list of columns to keep based on desired variables
keepRoman = ["Year Built",
            "Year Renovated",
            "Construction Material",
            "Features",
            "Amenities",
            "Number Of Parking Spaces",
            "Parking Spaces/Unit",
            "Star Rating",
            "Owner Name",
            "Owner Address",
            "Owner City State Zip",
            "Last Sale Date",
            "Last Sale Price",
            "Building Class",
            "Building Park"]

keepGeneral = ["PropID",
                "Property Address",
                "Property Name",
                "Building Status",
                "Secondary Type",
                "Market Name",
                "Submarket Name",
                "City",
                "State",
                "Zip",
                "County Name"
                "Longitude",
                "Latitude"]

keepList = ["PID",
            "PropID",
            "Property Address",
            "Property Name",
            "Building Status",
            "Secondary Type",
            "Market Name",
            "Submarket Name",
            "City",
            "State",
            "Zip",
            "County Name"
            "Longitude",
            "Latitude",
            "Number Of Stories",
            "Land Area (AC)",
            "Days On Market"
            "Floor Area Ratio",
            "Number Of Floors",
            "Land Area AC",
            "Price Per AC Land",
            "Market Time",
            "Net Income"
            "Year Built",
            "Year Renovated",
            "Construction Material",
            "Features",
            "Amenities",
            "Number Of Parking Spaces",
            "Parking Spaces/Unit",
            "Star Rating",
            "Owner Name",
            "Owner Address",
            "Owner City State Zip",
            "Last Sale Date",
            "Last Sale Price",
            "Building Class",
            "Building Park",
            "% 1-Bed",
            "% 2-Bed",
            "% 3-Bed",
            "% 4-Bed",
            "Number Of 1 Bedrooms",
            "Number Of 2 Bedrooms",
            "Number Of 3 Bedrooms",
            "Number Of 4 Bedrooms",
            "Number Of Studios",
            "PropertyType",
            "Four Bedroom Vacancy %",
            "Four Bedroom Vacant Units",
            "One Bedroom Vacancy %",
            "One Bedroom Vacant Units",
            "Studio Vacancy %",
            "Studio Vacant Units",
            "Submarket Cluster",
            "Three Bedroom Vacancy %",
            "Three Bedroom Vacant Units",
            "Total Buildings",
            "Two Bedroom Vacancy %",
            "Two Bedroom Vacant Units",
            "RBA",
            "Parcel Number 1(Min)",
            "Parcel Number 2(Max)",
            "Last Sale Date",
            "Latitude",
            "Longitude",
            "Situs_Num",
            "Situs_Num_Remainder",
            "SITUS_DIR",
            "SITUS_NAM",
            "SCP",
            "SCSitus_NumNam",
            "SCAPN",
            "Number of Units",
            "Vacancy %",
            "Avg Concessions %",
            "Affordable Type",
            "Market Segment",
            "Amenities",
            "Building Class",
            "Closest Transit Shop",
            "Closest Transit Stop Dist (mi)",
            "Closest Transit Stop Walk Time (min)",
            "Features",
            "Four Bedroom Concessions %",
            "Four Bedroom Vacancy %",
            "Number Of Studios",
            "One Bedroom Concessions %",
            "One Bedroom Vacancy %",
            "One Bedroom Vacant Units",
            "Percent Leased",
            "Studio Concessions %",
            "Studio Vacancy %",
            "Studio Vacant Units",
            "Three Bedroom Concessions %",
            "Three Bedroom Effective Rent/Unit",
            "Three Bedroom Vacancy %",
            "Three Bedroom Vacant Units",
            "Total Buildings",
            "Two Bedroom Concessions%",
            "Two Bedroom Vacancy %",
            "Two Bedroom Vacant Units",
            "2010 Avg Age(1m)",
            "2010 Med Age(1m)",
            "2010 Pop Age 0-4(1m)",
            "2010 Pop Age 10-14(1m)",
            "2010 Pop Age 15-19(1m)",
            "2010 Pop Age 20-24(1m)",
            "2010 Pop Age 45-49(1m)",
            "2010 Pop Age 50-54(1m)",
            "2010 Pop Age 55-59(1m)",
            "2010 Pop Age 5-9(1m)",
            "2010 Pop Age 60-64(1m)",
            "2010 Pop Age 65+(1m)",
            "2010 Pop Age 85+(1m)",
            "2019 Avg Age(1m)",
            "2019 Avg Age&#044; Female(1m)",
            "2019 Avg Age&#044; Male(1m)",
            "2019 HH Age 15-24(1m)",
            "2019 HH Age 25-34(1m)",
            "2019 HH Age 35-44(1m)",
            "2019 HH Age 45-54(1m)",
            "2019 HH Age 55-64(1m)",
            "2019 HH Age 65-74(1m)",
            "2019 HH Age 75-84(1m)",
            "2019 HH Age 85+(1m)",
            "2019 Med Age(1m)",
            "2019 Med Age&#044; Female(1m)",
            "2019 Med Age&#044; Male(1m)",
            "2019 Median HH Age(1m)",
            "2019 Pop Age <19(1m)",
            "2019 Pop Age 0-4(1m)",
            "2019 Pop Age 10-14(1m)",
            "2019 Pop Age 15-19(1m)",
            "2019 Pop Age 20-24(1m)",
            "2019 Pop Age 20-64(1m)",
            "2019 Pop Age 25-29(1m)",
            "2019 Pop Age 30-34(1m)",
            "2019 Pop Age 35-39(1m)",
            "2019 Pop Age 40-44(1m)",
            "2019 Pop Age 45-49(1m)",
            "2019 Pop Age 50-54(1m)",
            "2019 Pop Age 55-59(1m)",
            "2019 Pop Age 5-9(1m)",
            "2019 Pop Age 60-64(1m)",
            "2019 Pop Age 65+(1m)",
            "2019 Pop Age 65-69(1m)",
            "2019 Pop Age 70-74(1m)",
            "2019 Pop Age 75-79(1m)",
            "2019 Pop Age 80-84(1m)",
            "2019 Pop Age 85+(1m)",
            "2024 Avg Age(1m)",
            "2024 Avg Female Age(1m)",
            "2024 Avg Male Age(1m)",
            "2024 HH Age 15-24(1m)",
            "2024 HH Age 25-34(1m)",
            "2024 HH Age 35-44(1m)",
            "2024 HH Age 45-54(1m)",
            "2024 HH Age 55-64(1m)",
            "2024 HH Age 65-74(1m)",
            "2024 HH Age 75-84(1m)",
            "2024 HH Age 85+(1m)",
            "2024 Med Age(1m)",
            "2024 Median HH Age(1m)",
            "2024 Pop Age <19(1m)",
            "2024 Pop Age 0-4(1m)",
            "2024 Pop Age 10-14(1m)",
            "2024 Pop Age 15-19(1m)",
            "2024 Pop Age 20-24(1m)",
            "2024 Pop Age 20-64(1m)",
            "2024 Pop Age 25-29(1m)",
            "2024 Pop Age 30-34(1m)",
            "2024 Pop Age 35-39(1m)",
            "2024 Pop Age 40-44(1m)",
            "2024 Pop Age 45-49(1m)",
            "2024 Pop Age 50-54(1m)",
            "2024 Pop Age 55-59(1m)",
            "2024 Pop Age 5-9(1m)",
            "2024 Pop Age 60-64(1m)",
            "2024 Pop Age 65+(1m)",
            "2024 Pop Age 65-69(1m)",
            "2024 Pop Age 70-74(1m)",
            "2024 Pop Age 75-79(1m)",
            "2024 Pop Age 80-84(1m)",
            "2024 Pop Age 85+(1m)",
            "Situs_Num",
            "Situs_Num_Remainder",
            "SITUS_DIR",
            "SITUS_NAM",
            "Actual Cap Rate",
            "Asking Price",
            "Assessed Value",
            "Assessed Year",
            "Bldg SF",
            "Land Area AC",
            "Land Area SF",
            "Market",
            "Number Of Parking Spaces",
            "Number Of Units",
            "Parking Ratio",
            "Price Per SF",
            "Price Per SF (Net)",
            "Price Per Unit",
            "Sale Date",
            "Sale Price",
            "Star Rating",
            "Submarket Code",
            "Submarket Cluster",
            "Year Built",
            "Zoning",
            "Number Of Units",
            "Avg Unit SF",
            "Parking Spaces/Unit",
            "Building Class",
            "One Bedroom Avg SF"
            "One Bedroom Asking Rent/SF",
            "One Bedroom Asking Rent/SF",
            "One Bedroom Effective Rent/SF",
            "One Bedroom Effective Rent/Unit",
            "Four Bedroom Avg SF"
            "Four Bedroom Asking Rent/SF",
            "Four Bedroom Asking Rent/Unit",
            "Four Bedroom Effective Rent/SF",
            "Four Bedroom Effective Rent/Unit",
            "Studio Asking Rent/SF"
            "Studio Asking Rent/Unit",
            "Studio Avg SF",
            "Studio Effective Rent/SF",
            "Studio Effective Rent/Unit",
            "Three Bedroom Asking Rent/SF",
            "Three Bedroom Asking Rent/Unit",
            "Three Bedroom Avg SF",
            "Three Bedroom Effective Rent/Unit",
            "Three Bedroom Effective Rent/SF",
            "Two Bedroom Asking Rent/SF",
            "Two Bedroom Asking Rent/Unit",
            "Two Bedroom Avg SF",
            "Two Bedroom Effective Rent/Unit",
            "Two Bedroom Effective Rent/SF"]
len(keepList)
# %% Merging Sales and key to match PropID then merging with property
key["PropID"] = key["PID"]
key

sales_key = pd.merge(sales, key, on='SID')
sales_key.shape
df = pd.merge(sales_key, property, on = "PropID")

# %% Dropping unnecessary columns
df.shape
len(keepList)
finalCols = []

for i in range(len(keepList)):
    if ((keepList[i] in sales.columns) & (keepList[i] in property.columns)):
        keepList[i] = keepList[i] + "_x"

for i in list(df.columns):
    if i in keepList:
        finalCols.append(i)
len(finalCols)

df = df[finalCols]
df.shape

### %% Function nanFinder finds the number of null values for each column
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

dfColDict = nanFinder(df)

dropDict = {}
for (key, value) in dfColDict.items():
    if value < 5000:
        dropDict.update({key:value})

dropDict

# %% Replacing Null values in bedroom variables
bedroomCols = ["Number Of Studios_x","Number Of 1 Bedrooms_x", \
                "Number Of 2 Bedrooms_x", "Number Of 3 Bedrooms_x", \
                "Number Of 4 Bedrooms" ]

for i in bedroomCols:
    df[i].replace(np.nan, 0)
df["Number Of 4 Bedrooms"] = df["Number Of 4 Bedrooms"].replace(np.nan, 0)

df.shape

# %% Removing "_x" from certain Variables
dfColList = list(df.columns)

for i in range(len(dfColList)):
    if dfColList[i][-2:] == "_x":
        dfColList[i] = dfColList[i][:len(dfColList[i])-2]

df.columns = dfColList
df.head()

# %% Deleting row with unreasonable square footage and sale price
df = df[df["Bldg SF"] > 10000]
df = df[df["Sale Price"] > 20000]

# %% Exporting df to csv
df.to_csv(path + "combo.csv", index = False)

# %% Exporting to a new dataframe called origin
origin = df

# %% Cleaning Origin
# Variables to clean
# Star Rating to int
origin["Star Rating"] = origin["Star Rating"].apply(str)
origin["Star Rating"] = origin["Star Rating"].apply(lambda x: x[0])
origin["Star Rating"] = origin["Star Rating"].replace("n", np.nan)
origin["Star Rating"] = origin["Star Rating"].apply(float)
origin["Star4"] = origin["Star Rating"] - 1
# Creating Star Rating Dummy Variables
for i in [2,3,4,5]:
    origin["Star_" + str(i)] = origin["Star Rating"].apply(lambda x: 1 if x == i else 0)
origin["Star_2"]
# Converting % 4-Bed to to numeric
origin["% 4-Bed"] = origin["% 4-Bed"].str.replace("%", "")
origin["% 4-Bed"] = origin["% 4-Bed"].apply(float)
# Failed Cleaning of Unit Mix variables
#unitMixList = ["% 1-Bed", "% 2-Bed", "% 3-Bed", "% 4-Bed"]
#origin["percent"] = origin[unitMixList].sum(skipna = True, axis = 1)
#origin[origin["% 4-Bed"].isna() & df[]]["% 4-Bed"] = 100 - origin[unitMixList].sum(skipna = True, axis = 1)
#origin[(origin["% 4-Bed"].isna() & origin["% 1-Bed"].isna() & origin["% 2-Bed"].isna() & origin["% 3-Bed"].isna())][unitMixList]
#origin[unitMixList+["percent"]]

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

# Creating Floor To Area Ratio column
origin["Floor_to_Area_Ratio"] = origin["Bldg SF"]/origin["Land Area SF"]

# Creating Affordable type column
origin["Affordable_Housing"] = origin["Affordable Type"].apply(lambda x: 1 if type(x) == str else 0)

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

# %% Creating Dummy Variables for each of the amenities
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
                        ]

# Creating Dummy Variables for each amenity

for i in amenities_relevant:
    origin[i] = origin["Amenities"].apply(lambda x: 1 if i in x else 0)

# %% Exporting Dataset to csv
origin.to_csv(path + "origin.csv", index = False)

#origin[["Sale Date", "Sale Price", "Assessed Value"]]
