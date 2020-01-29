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
#from linearmodels.iv import IV2SLS
path = "/Users/romansielewicz/Google Drive/Fall2019/real_estate_Project/Datasets/"

# %% Importing Dataset
origin = pd.read_csv(path + "origin.csv", low_memory = False)
origin.shape

# %% Creating df, the dataframe we will use to analyze sale price
# Subsetting for only columns with valid Sale Price data
df = origin.dropna(axis = 0, subset = ["Sale Price"])
df = df[df["Sale Price"] > 10000]
df.shape

# %% Creating rentdf, the dataframe we will use to analyze Rental Revenue
# Subsetting for only columns with valid Rental Revenue data
rentdf = origin.dropna(axis = 0, subset = ["Rental_Revenue"])
rentdf = rentdf[rentdf["Rental_Revenue"] > 10000]
rentdf.shape

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

# %% Correlation Matrix
#Choosing Variables to Analyze
# Making Materials List: A list of construction materials used
materialList = list(origin["Construction Material"].unique())
materialList = materialList[1:]

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

vars =  [ \
        "Out_Of_State",
        "Building_Age",
        "Years_Since_Renovation",
        "Number Of Parking Spaces",
        "Parking Spaces/Unit",
        "Star4",
        ] \
        + amenities_relevant \
        + materialList
# Generating Correlation Matrix
corrdf_sale = df[["Sale Price"] + vars]
corrdf_sale.corr()

# %% Linear Regression Analysis
reg1 = sm.OLS(endog=df["Sale Price"], exog=df[vars], \
    missing='drop')
results1 = reg1.fit()
print(results1.summary())

# %% Retrying Regression Model with fewer independent variables
vars =  [
        "Out_Of_State",
        "Building_Age",
        #"Years_Since_Renovation",
        #"Number Of Parking Spaces",
        #"Parking Spaces/Unit",
        #"Star4",
        "Masonry",
        "Reinforced Concrete",
        #"Wood Frame",
        #"Fitness Center",
        "Property Manager on Site",
        #"Business Center",
        "Playground",
        "Maintenance on site",
        #"Tennis Court",
        #"Package Service",
        "Laundry Service",
        "Storage Space",
        "Star_2",
        "Star_3",
        "Star_4",
        "Star_5"] + \
        [
        "Bldg SF", # Start of Team Variables
        "Number Of Units",
        "Avg Concessions %",
        "Number Of Floors",
        #"Land Area (AC)",
        "Closest Transit Stop Dist (mi)",
        "2019 Avg Age(1m)",
        "Floor_to_Area_Ratio",
        "Affordable_Housing",
        #"Market Time"
        ]
print(len(vars))

# defining independent and dependent variables
y = df["Sale Price"]
x = sm.add_constant(df[vars])

regSale = sm.OLS(endog=y, exog=x, \
    missing='drop')
resultsSale = regSale.fit()
print(resultsSale.summary())

#  %% Log Model
vars =  [
        "Out_Of_State",
        "Building_Age",
        #"Years_Since_Renovation",
        "Number Of Parking Spaces",
        #"Parking Spaces/Unit",
        #"Star4",
        "Masonry",
        "Reinforced Concrete",
        "Wood Frame",
        "Fitness Center",
        #"Property Manager on Site",
        "Business Center",
        "Playground",
        "Maintenance on site",
        "Tennis Court",
        #"Package Service",
        #"Laundry Service",
        #"Storage Space",
        "Star_2",
        "Star_3",
        "Star_4",
        "Star_5"] + \
        [
        "Bldg SF", # Start of Team Variables
        "Number Of Units",
        "Avg Concessions %",
        "Number Of Floors",
        #"Land Area (AC)",
        "Closest Transit Stop Dist (mi)",
        "2019 Avg Age(1m)",
        "Floor_to_Area_Ratio",
        "Affordable_Housing",
        #"Market Time"
        ]

df["Log_Sale_Price"] = df["Sale Price"].apply(math.log)
y = df["Log_Sale_Price"]
x = sm.add_constant(df[vars])

regSaleLog = sm.OLS(endog=y, exog=x, missing='drop')
resultsSaleLog = regSaleLog.fit()
print(resultsSaleLog.summary())

# %% Model to predict Price Per SF
df["Star Rating"].value_counts()

vars =  [
        "Out_Of_State",
        "Building_Age",
        #"Years_Since_Renovation",
        #"Number Of Parking Spaces",
        "Parking Spaces/Unit",
        #"Star4",
        "Masonry",
        "Reinforced Concrete",
        "Wood Frame",
        "Fitness Center",
        "Property Manager on Site",
        "Business Center",
        "Playground",
        "Maintenance on site",
        "Tennis Court",
        "Package Service",
        "Laundry Service",
        "Storage Space",
        "Star_2",
        "Star_3",
        "Star_4",
        "Star_5"] + \
        [
        #"Bldg SF", # Start of Team Variables
        "Number Of Units",
        "Avg Concessions %",
        "Number Of Floors",
        #"Land Area (AC)",
        "Closest Transit Stop Dist (mi)",
        "2019 Avg Age(1m)",
        "Floor_to_Area_Ratio",
        "Affordable_Housing",
        #"Market Time"
        ]
y = df["Price Per SF"]
x = sm.add_constant(df[vars])

regSF = sm.OLS(endog=y, exog=x, missing='drop')
resultsSF = regSF.fit()
print(resultsSF.summary())

# %% Log Model for Price Per SF
vars =  [
        "Out_Of_State",
        "Building_Age",
        #"Years_Since_Renovation",
        #"Number Of Parking Spaces",
        "Parking Spaces/Unit",
        #"Star4",
        "Masonry",
        "Reinforced Concrete",
        "Wood Frame",
        "Fitness Center",
        #"Property Manager on Site",
        #"Business Center",
        "Playground",
        #"Maintenance on site",
        #"Tennis Court",
        #"Package Service",
        "Laundry Service",
        #"Storage Space",
        "Star_2",
        "Star_3",
        "Star_4",
        "Star_5"] + \
        [
        #"Bldg SF", # Start of Team Variables
        "Number Of Units",
        "Avg Concessions %",
        #"Number Of Floors",
        #"Land Area (AC)",
        "Closest Transit Stop Dist (mi)",
        "2019 Avg Age(1m)",
        #"Floor_to_Area_Ratio",
        "Affordable_Housing",
        #"Market Time"
        ]

df["Log_Price_Per_SF"] = df["Price Per SF"].apply(math.log)
y = df["Log_Price_Per_SF"]
x = sm.add_constant(df[vars])

regSFLog = sm.OLS(endog=y, exog=x, missing='drop')
resultsSFLog = regSFLog.fit()
print(resultsSFLog.summary())

# %% Model for Total Rental_Revenue
vars =  [
        #"Out_Of_State",
        "Building_Age",
        #"Years_Since_Renovation",
        "Number Of Parking Spaces",
        "Parking Spaces/Unit",
        #"Star4",
        "Masonry",
        "Reinforced Concrete",
        #"Wood Frame",
        "Fitness Center",
        "Property Manager on Site",
        #"Business Center",
        "Playground",
        #"Maintenance on site",
        "Tennis Court",
        #"Package Service",
        "Laundry Service",
        #"Storage Space",
        "Star_2",
        "Star_3",
        "Star_4",
        "Star_5"] + \
        [
        "Bldg SF", # Start of Team Variables
        "Number Of Units",
        "Avg Concessions %",
        "Number Of Floors",
        #"Land Area (AC)",
        "Closest Transit Stop Dist (mi)",
        "2019 Avg Age(1m)",
        #"Floor_to_Area_Ratio",
        "Affordable_Housing",
        #"Market Time"
        ]

y = rentdf["Rental_Revenue"]
x = sm.add_constant(rentdf[vars])

regRental = sm.OLS(endog=y, exog=x, missing='drop')
resultsRental = regRental.fit()
print(resultsRental.summary())

# %% Model for Log Rental_Revenue
vars =  [
        #"Out_Of_State",
        "Building_Age",
        #"Years_Since_Renovation",
        "Number Of Parking Spaces",
        "Parking Spaces/Unit",
        #"Star4",
        "Masonry",
        "Reinforced Concrete",
        "Wood Frame",
        "Fitness Center",
        "Property Manager on Site",
        "Business Center",
        #"Playground",
        "Maintenance on site",
        "Tennis Court",
        #"Package Service",
        #"Laundry Service",
        #"Storage Space",
        "Star_2",
        "Star_3",
        "Star_4",
        "Star_5"] + \
        [
        "Bldg SF", # Start of Team Variables
        "Number Of Units",
        "Avg Concessions %",
        "Number Of Floors",
        #"Land Area (AC)",
        "Closest Transit Stop Dist (mi)",
        "2019 Avg Age(1m)",
        #"Floor_to_Area_Ratio",
        "Affordable_Housing",
        #"Market Time"
        ]
rentdf["Log_Rental_Revenue"] = rentdf["Rental_Revenue"].apply(math.log)

y = rentdf["Log_Rental_Revenue"]
x = sm.add_constant(rentdf[vars])

regRental = sm.OLS(endog=y, exog=x, missing='drop')
resultsRental = regRental.fit()
print(resultsRental.summary())
