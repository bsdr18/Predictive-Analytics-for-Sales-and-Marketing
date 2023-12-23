# Predictive-Analytics-for-Sales-and-Marketing

## Project Title: A Comprehensive Time Series Analysis for Dynamic Sales Forecasting

### PROJECT OVERVIEW
- ```Sales prediction``` poses a formidable challenge for businesses, attributed to factors such as limited ```historical data```, unpredictable external variables like weather, natural occurrences, government regulations, and primarily the ```dynamic nature``` of the market.
- This project is dedicated to implementing diverse ```forecasting``` techniques for predicting sales at Olist, a Brazilian E-Commerce startup.
- The project encompasses two notebooks:
  1. "Data_cleaning" serves as the initial notebook, loading the raw data, executing cleaning procedures, and amalgamating all tables into the master dataset.
  2. "Time_series" is the subsequent notebook, where various modeling techniques for time series forecasting have been explored.
 
### DATA DESCRIPTION
The dataset obtained from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) presents a public collection of Brazilian ecommerce orders conducted at Olist Store. 
- Covering 100,000 orders spanning the period from 2016 to 2018, these transactions took place across various marketplaces in Brazil. 
- The dataset provides a comprehensive view of each order, encompassing aspects such as order status, pricing, payment and freight performance, customer location, product attributes, and customer reviews.
- Additionally, a geolocation dataset linking Brazilian zip codes to corresponding latitude and longitude coordinates has been shared.

#### Data Schema
The data is divided in multiple datasets for better understanding and organization. Please refer to the following data schema when working with it:

![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/ceb99da2-4e0f-4e24-bb54-c3c0723e9b7c)
[Source](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

### PROJECT PIPELINE
1. Data Collection and Pre-Processing
2. Time Series Data Pre-Processing
3. Time Series Analysis Using SARIMA Model
4. Time Series Modelling Using Facebook Prophet
5. Issues With Hourly Sampled Data
6. Conclusion
   
### 1. DATA COLLECTION AND PRE-PROCESSING
- I have started by loading all the datasets individually and aimed to do the following tasks:
  - Understand data dictionary to get an overview of numerical and categorical columns.
  - Correct the data format.
  - Clean the data by deleting redundant columns, imputing the null values and deleting the duplicate rows and columns.

#### Quick Findings
##### 1. Customers dataset:
- Customers dataset has information about geolocation of customers.
- We have a total of 99441 customer ids which is the primary key for this table. These customers ids are created when a user makes a purchase. They are actually transaction ids.
- We have a total of 96096 unique customer ids. It shows that we have around 96.6 % of new customers. Only 3.4% of the customers have made repeat purchase from the olist platform. It is because olist was founded in 2015 and they started selling online in 2016. The data we downloaded from Kaggel is from 2016 to 2018, when it was fairly new thus we only have new customers.
- This dataset has ```four``` columns of object datatype and ```one``` column with numeric datatype.
- There are no duplicates across rows or columns.
- There is no null value.

##### 2. Geoloc dataset:
- We have a huge number of duplicates here. We can drop the duplicates only keeping the first of the matching row.
  ```
  #checking duplicates across rows by keeping only the frist value and dropping the next matching value in place
  geoloc.drop_duplicates(keep='first', inplace=True)
  ```
- Out of 738332 rows only 19015 are the unique zip code prefix. A zip code postfix can define a complete zip code but we don't have any. For sanity check we can check how many different latitude and longitude values do we have for a particular zip code prefix.
  ```
  #checking no of rows with same zip code.
  geoloc['geolocation_zip_code_prefix'].value_counts()
  ```
  ![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/f378004d-ffc1-42f4-a10e-9922dc688d7a)

- Geoloc dataset has information about latitudes and longitudes of various cities and states of Brazil.
- We have a total of 19015 unique zip code prefix and there could be multiple latitude and longitude associated with that code that differentiate different locations within that zip code.
For example for zip code prefix 38400 we have ```779 rows with same city and state value but slightly different latitutde and longitude values.```
- We found some duplicate values and deleted them.
- This dataset has ```two``` columns of object datatype and ```three``` column with numeric datatype.
- There is no null value.
**Since we don't have the complete zip code in both geoloc and customer dataset, we can take mean of the latitude and longitude coordinates for each zip code prefix and save it in separate dataframe.**
  ```
  #creating another dataframe which has zip code prefix and mean values for latitude and longitude
  coordinates= geoloc.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
  ```
  ```
  customer_loc=pd.merge(left=customers, right=coordinates, left_on='customer_zip_code_prefix', 
                      right_on='geolocation_zip_code_prefix', how='left')
  ```
  ```
  #drop unnecesary column
  customer_loc.drop(columns=['geolocation_zip_code_prefix'], inplace=True)
  ```
- The approach I am using to fill NANs:
  - I will aggreagate the columns on the basis of customer city and state and will calculate the mean of the available latitude and longitude for that city state combination.
  - Using these calaculated latitude and longitudes I will fill the missing latitude and longitude where the city and state is matching.
```
# grouping by customer state and imputing the misisng latitude by mean of that group or customer state
d_tally['customer_lat']=d_tally.groupby('customer_state', sort=False)['customer_lat'].transform(lambda x: x.fillna(x.mean()))
# grouping by customer state and imputing the misisng longitude by mean of that group or customer state
d_tally['customer_lng']=d_tally.groupby('customer_state', sort=False)['customer_lng'].transform(lambda x: x.fillna(x.mean()))
```

##### 3. Sellers dataset:
- Sellers dataset has information about seller location.
- We have a total of 3095 unique seller ids which is the primary key for this dataset
- This dataset has ```three``` columns of object datatype and ```one``` column with numeric datatype.
- There are no duplicates across rows or columns.
- There is no null value.

##### 4. Payments dataset:
- Payments dataset has information about the way customer made payment for each order.
- We have a total of 99441 customer id which is equal to the total order ids but we have payment information for 99440 orders.
- This dataset has ```three``` columns of object datatype and ```two``` column with numeric datatype.
- There are no duplicates across rows or columns.
- There is no null value.
- order_id is the foreign key in this table.
We will keep this table aside beacuse we are not interested in this table for our present scope of work.

##### 5. Order item dataset:
- Order item dataset has information about order item. It tells us about number of items in each order, shipping limit and fright value
- We have a total of 98666 order ids which is less than 99441.
- This dataset has ```four``` columns of object datatype and ```three``` column with numeric datatype.
- The column shipping limit date is of date time format so we need to convert it into correct format.
- There are no duplicates across rows or columns.
- There is no null values.

##### 6. Order dataset
- Orders dataset has information about the orders. Each order contains a customer id, order status, purchase timestamp and actual and estimated delivery information.
- We have a total of 99441 unique orders which is the primary key for this table.
- This dataset has ```eight``` columns of object datatype.
- **There are a total of five columns of date time values but saved in object format. We need to convert them into date time format.**
  ```
  #convert all the rows with date time data to date-time format.
  orders['order_purchase_timestamp']=pd.to_datetime(orders['order_purchase_timestamp'])
  orders['order_approved_at']=pd.to_datetime(orders['order_approved_at'])
  orders['order_delivered_carrier_date']=pd.to_datetime(orders['order_delivered_carrier_date'])
  orders['order_delivered_customer_date']=pd.to_datetime(orders['order_delivered_customer_date'])
  orders['order_estimated_delivery_date']=pd.to_datetime(orders['order_estimated_delivery_date'])
  ```
- There are no duplicates across rows or columns.
- There are null values in the order_approved_at, order_delivered_carrier_date, and order_delivered_customer_date.
We have null values in 3 columns: order_approved_at, order_delivered_carrier_date, and order_delivered_customer_date. Does it have to do anything with the Order status?
```
#percentage of order according to different order status
order_st_per= round(orders['order_status'].value_counts(normalize=True)*100, 2)

#ploting the orders to visually check what all status do we have.
plt.figure()
graph=plt.bar(order_st_per.index, order_st_per.values)
plt.title("Percentage with respect to total orders")
plt.xticks(rotation=45)
for p in graph:
    height=p.get_height()
    plt.annotate( "{}%".format(height),(p.get_x() + p.get_width()/2, height+.05),ha="center",va="bottom",fontsize=9)
plt.show()
```
![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/1553ad12-338b-4036-8fab-a53c2fa26df6)

We have a total of eight order status. Our data has 97% of orders that were delivered. Only 0.63% of orders were cancelled. There are diferent order status that specify at which stage our order is.

We want to understand if there is any relationship with missing values and the order status. We will filter the rows with missing values and check what is the status of that order.

Customer made payment for the order → Order created
Seller approved the order → Order approved
Seller preparing the order → Order processing
Seller invoiced the order → Order invoiced
Seller shipped the order and was handed over to logistic partner → Order shipped
Logistic partner delivered the product to end customer → Order delivered

**We might not need the rows with order status as unavailable, invoiced, processing, created, approved, shipped so we will go ahead and delete these rows.** Logically these rows are missing the correct values as per the status. If we were to impute the missing values then we will have to change the status also. These rows have very less number compared to the total rows and there should not be much effect after removing these rows.

**Need to impute the missing values for delivered and canceled order status**.
##### How to impute the missing values?
- We can find the difference in days between the known values i.e order_purchase_timestamp and the columns with NANs.
- We can see the distribution of days that we will get from above step by plotting the box plot.
- Then we can decide if we want to impute the missing values with mean, median or mode of the days difference.
- Once we are clear of that, we can determine the missing value by adding the days difference to the order purchase date.

###### Observations:
- We can see that there are a bunch of outliers in these plots.
- The number of days between purchase date and carrier date has a negative observation. If we are assuming that the data is recorded at Olist server where time zone factor has been removed. This is an incorrect observation and can be removed.
- Upon looking at the box plots we have a bunch of outliers and therefore imputation using median is the right approach.
We will simply drop these columns as we do not want to deal with them in our current scope. If we were to impute then we would calaculate the median number of days from each of diff_approved, diff_logist, diff_del and add it to order_purchase_timestamp to get the order_approved_at, order_delivered_carrier_date, order_delivered_customer_date respectively.

##### 7. Product dataset:
- Products dataset has information about product categories, and their attributes.
- We have a total of 32951 products ids which is the primary key for this table. There are a total of 73 product categories.
- This dataset has two columns of object datatype and seven column with numeric datatype.
- There are no duplicates across rows or columns.
- There are few null values and we need to impute these values.

###### How can we impute these missing values?
- We have seen that the rows which are missing values, a majority are missing a categorical data ie. product category name and respective, description, name length and photo quantities which are numerical.
- We can find out the rows which are exactly matching the columns (weight, lenght, height, width) and we can fill the null value with the product category name of the matching category. - We can fill rest of the values ie product description lenght, product name lenght and photo quantity with either mean, median or mode of that known category.
  - If there are multiple matches for product category, we can filter out the category with most occuring match.
  - If there is no match we can create a separate category 'other' and fill rest of the values with either of mean, median or mode (determined after making a boxplot).
- We will separate the rows from products dataframe with the missing values in separate dataframe (missing) and create another dataframe (all_values) where there is no null values.
- We will find the element wise match of missing with all_values.
- We only have two rows with missing product weight, height, lenght and width. We will use mean to fill these values.

##### 8. Review dataset:
- Reviews dataset has information about reviews given by the customers. It consists of review score, comment, review creation date and review submission timestamp.
**- We have a total of 99224 review ids of which 98410 are the unique review ids. It means there are 814 reviews which have been resubmitted. These are the ones that need to be tackled.
- We have 98673 unique order ids and 98410 unique review ids. It means that there are 263 reviews with same order id. It is possible that these reviews are for different products ordered under same order id.**
- This dataset has ```six``` columns of object datatype and ```one``` column with numeric datatype.
- There are no duplicates across rows or columns.
- There are 145903 null values.

##### 9. Product_eng dataset:
- I have joined the product category with english names of the products and dropping the original product category name with names in portugese.

#### Joining All The Tables
- We will be joining the tables to get a master table for addressing the business problem of Sales prediction.
- Starting from the orders dataset, we will first join the order_items and then the products dataset.
- Since we have already cleaned the orders dataset and saved it in a csv file, we will load that data set and start from there.
- So, I have merged orders and order_items, products table with Order_comp, Sellers dataset with Order_cons, merged customer dataset, and the reviews dataset.
We have learnt from the Kaggel website that the total order value is calculated using qty and price. Since price tells us about the unit price, the total order value= qty* price.
- Creating column ```total_amount```.
  ```final_cleaned['total_amount']=final_cleaned['qty']*final_cleaned['price']```

#### Scraping Holiday Data
- Since, we will be predicting sales amount we need to collect Holiday information to help our model in understanding impacts of holidays. We will be doing a simple scape to collect the Brazilian National Holiday information from this [website](https://www.officeholidays.com/countries/brazil/)
- We can append the above URL with 2017 and 2018 to get the pages with holiday information for year 2017 and 2018.
```
  # For web scraping (the requests package allows you to send HTTP requests using Python)
import requests
from bs4 import BeautifulSoup

# For performing regex operations
import re

# For adding delays so that we don't spam requests
import time
#define empty dictionary to save content
content={}

#scaping holiday information for pages 2017 and 2018
for i in [2017, 2018]:
    url = 'https://www.officeholidays.com/countries/brazil/'
    url = url+str(i)
    response = requests.get(url)
    soup = BeautifulSoup(response.content)
    content[i]=soup.find_all('time')

#extracting Holiday information from the scarpped data
#empty list
holidays=[]
for key in content:
    dict_size=len(content[key])
    dict_val=content[key]
    for j in range(0,dict_size):
        holidays.append(dict_val[j].attrs['datetime'])

#creating a dataframe for the holiday information
holidays_df=pd.DataFrame(index=[holidays], data=np.ones(len(holidays)), columns=['is_holiday'])
holidays_df.head()
```
- This dataframe has only one column 'is_holiday' which is one meaning it is an holiday. The index are the dates of the holiday.
- These dates are for year 2017 and 2018. The index is not continuous, these are just the holiday dates. We have saved the data like this so that we can use it for time series.

#### Final Data Insights
- A total of 96K unique orders.
- Olist platform has 96.79% of new customers and 3.21% have made repeat purchase.
- A total of 32K different products belonging to 74 categpries are sold.
- The overall revenue earned as of Aug 2018 is 14.9 milion Brzailian Real.
- There was a highest sale of 184K R$ that was recorded on Black Friday 2017.
- The top five categories are:
  
  ![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/72e34a82-1e4f-4ac8-a026-42bc06e97563)

- The monthly orders and revenue earned showed a growth.

### 2. TIME SERIES DATA PRE-PROCESSING

#### Data Dictionary
We have a total of 110013 rows of orders with 28 features, I am specifying all the high level details about the data which we extracted during data cleaning and wrangling. Each row in the table specifies a order with the product category bought, quantity of item purchased, unit price of the product and has details about purchase time, delivery details, review score and customer and seller information. 

- **order_id** : Specifies the unique order. We have 95832 unique orders. Of 110K rows an order_id can reappear in the  dataframe but it will have another product category and number of items bought in that category.                     
- **customer_id**: Specifies the customer id for the order. We have a customer ids associated with each order. There are a total of 95832 unique customer ids.   
- **order_purchase_timestamp** : The time stamp for the order. It includes date and time.       
- **order_estimated_delivery_date** : Estimated delivery date at the time of purchase.  
- **qty** : Number of items bought in a product category                           
- **product_id** : This specify the actual product in a product category. We have 32072 unique products within 74 overall product categories.                       
- **seller_id** : We have 2965 unique sellers.                        
- **shipping_limit_date** : This date informs seller of the shipping limit so they can dispatch the order at the earliest.   
- **price** : Unit price for each product.                          
- **freight_value** : The freight charges based on product weight and dimension. This value is for one item. If there are three items the total freight will be equal to three times the freight_value.                 
- **product_name_lenght** : Number of characters extracted from the product name.          
- **product_description_lenght** : Number of characters extracted from the product description.    
- **product_photos_qty** : Number of product published photos.             
- **product_weight_g** : Product weight measured in grams.               
- **product_length_cm** : Product length measured in centimeters.              
- **product_height_cm** : Product height measured in centimeters.              
- **product_width_cm** : Product width measured in centimeters.               
- **product_category_name_english** : English names of product categories.  
- **seller_city** : It is the city where seller is located.                    
- **seller_state** : It is the state where seller is located.                  
- **seller_lat** : It is the latitude of seller location.                     
- **seller_lng** : : It is the longitude of seller location.                     
- **customer_unique_id** : There are 92755 unique customers which make up 96.79 % of the total customers in database. Only 3.21% of the customers have made repeat purchase. It may be because the data we have is the initial data when Olist had just started its business and therefore we have all the new customers in the database.            
- **customer_city** : It is the city where customer is located.                  
- **customer_state** : It is the state where customer is located.                
- **customer_lat** : It is the latitude of customer location.
- **customer_lng** : It is the longitude of customer location.                  
- **review_score** : Reviews submitted by the customers range from 1-5.

**`Target Variable`** : **total_amount** : We have calculated this value after multiplying **qty** and **price**. This is the actual sales amount important for the business. We will be predicting sales amount to help business prepare for the the future. 

`Note`: We have not considered freight charges in the calculation of 'total_amount' beacuse we found that when olist started its business it was outsourcing the logistics to third party and therefore we want to give business insight of only the sales from the products sold at the Olist platform.

We also found that Olist had accquired PAX, its logistic partner later in the year 2020, check [here](https://www.bloomberglinea.com/english/olist-becomes-brazils-newest-unicorn-raises-186m/) here for more details.

#### Processing Data for Time Series
We have seen that the 'order_purchase_timestamp' has incorrect format. We will start with converting this column to date-time format and we will try to extract some features from dates for analysis.

We can extract year, date, moth , weekday and day information from the dates.
```
#converting date columns which are in object format to datetime format
master['purchase_year']=pd.to_datetime(master['order_purchase_timestamp']).dt.year
master['purchase_month']=pd.to_datetime(master['order_purchase_timestamp']).dt.month
master['purchase_MMYYYY']=pd.to_datetime(master['order_purchase_timestamp']).dt.strftime('%m-%Y')
master['purchase_week']=pd.to_datetime(master['order_purchase_timestamp']).dt.isocalendar().week
master['purchase_dayofweek']=pd.to_datetime(master['order_purchase_timestamp']).dt.weekday
master['purchase_dayofmonth']=pd.to_datetime(master['order_purchase_timestamp']).dt.day
```

We will aggregate the total_amount by dates so that we can get a time series, meaning a dataframe with the total_amount column arranged in order as per dates. We will set the dates as index.

#### Exploratory Data Analysis
##### 1. Heatmap: 
- To see which numerical features are highly correlated with the total_amount. This is just a high level overview to see which features can impact sales and also the correlation among the features.
  ![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/b83a5e95-1aeb-4791-8b73-f830e8eba1d3)

**Observations:**
- We can see that total_amount is highly correlated with price. This is obvious because we know that total_amount was calculated using price.
- purchase_week and purchase_month are highly correlated.
- product_weight and freight values are positively correlated as frieght is calaculated as per product weight as it was specified by the data publishers on Kaggle.
- We don't see any other feature standing out to have high correlation with total_amount.

##### 2. Histogram:
- To see the distribution of total_amount.
![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/a2cf238b-8f3b-4585-812f-19a28f680a69)

**Observations:**
- There is a peak at zero amount because we don't have any observation for most of the days in 2016.
- If we ignore that, our overall distribution is normal with some outliers at the right side. These outlier observations are from the peak sales time.

##### 3. Bar plot:
![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/b6a6a409-8704-4074-815f-83076e1266cc)

**Observations:**
- Health_beauty , watches_gift, bed_bath_table, computer_asscesories and sports_leisure are the top category by sales amount.
- PC_games, cds_dvds_musicals, fashion_children_clothes are the lowest earning products categories.

#### Decomposing Time Series
**We will be decomposing the time series using additive decomposition so that we can observe the underlying trend, seasonality and residuals**. 

Additive Decomposition : $Trend$+$Seasonality$+$Residual$

```
# decompose the time series
decomposition = tsa.seasonal_decompose(daily_data, model='additive')
#saving copy to new datafrme
daily_df=daily_data.copy()
# add the decomposition data
daily_df['Trend'] = decomposition.trend
daily_df['Seasonal'] = decomposition.seasonal
daily_df['Residual'] = decomposition.resid
```
```
#plotting the actual and decomposed componenets of time series
cols = ["total_amount","Trend", "Seasonal", "Residual"]

fig = make_subplots(rows=4, cols=1, subplot_titles=cols)

for i, col in enumerate(cols):
    fig.add_trace(
        go.Scatter(x=daily_df.index, y=daily_df[col]),
        row=i+1,
        col=1
    )

fig.update_layout(height=1200, width=1200, showlegend=False)
# fig.show()
fig.show("svg")
```
![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/e02d15f9-8f51-4355-a9fd-65d15146485a)

**Observations:**
- We can see that there is a slightly upward trend. Trend has a peak on Nov 26, 2017 beacuse of the black friday sale on Nov 24, 2017. It falls afterwards but then rises again. Although this black friday is an outlier but we should consider it in our calculatiobn as it is an important factor.
- There is a weekly seasonlality. It peaks once in the week and then falls. 
- There is no clear pattern in Residual. It has captured the peaks of Nov 24, 2017 and Sept 29, 2017.

#### Preparing for Modeling
1. Train and test split
2. Defining functions for plotting predictions and forecast
3. Defining functions for evaluation
We will be defining functions to calculate MAPE and RMSE. If we have Y as actual value and Predictions as predicted value for n number of observations then:

MAPE (Mean Absolute Percentage Error): It is a simple average of absolute percentage errors. It is calculated by 

$$ \frac{1}{n} \sum_{i=1}^{n} {| \frac{Y_{actual_i} - Predictions_{i}}{Y_{actual_i}} |} \times{100} $$

RMSE (Root Mean Sqaured Error) : It is the square root of the average of the squared difference between the original and predicted values in the data set. 

$$ \sqrt{\frac{1}{n} \sum_{i=1}^{n} {{(Y_{actual_i} - Predictions_{i})}^2 }} $$

### 3. MODELLING 
#### 1. SARIMA
We will start with SARIMA model to account for the seasonality in our model. SARIMA is Seasonal Autoregressive Integrated Moving Average, which explicitly supports univariate time series data with a seasonal component. Before jumping on to modelling, we need to get a basic understanding of what orders for Auto gregressive and Moving average to choose. We will plot the ACF and PACF plots to find it out.

ACF : Auto correlation function, describes correlation between original and lagged series.
PACF : Partial correlation function is same as ACF but it removes all intermediary effects of shorter lags, leaving only the direct effect visible.

##### Plotting ACF and PACF plot
```
def plot_acf_pacf(df, acf_lags: int, pacf_lags: int) -> None:
    """
    This function plots the Autocorrelation and Partial Autocorrelation lags.
    ---
    Args:
        df (pd.DataFrame): Dataframe contains the order count and dates.
        acf_lags (int): Number of ACF lags
        pacf_lags (int): Number of PACF lags
    Returns: None
    """
    
    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,9), facecolor='w')
    
    # ACF & PACF
    plot_acf(df, ax=ax1, lags=acf_lags)
    plot_pacf(df, ax=ax2, lags=pacf_lags, method='ywm')

    # Labels
    ax1.set_title(f"Autocorrelation {df.name}", fontsize=15, pad=10)
    ax1.set_ylabel("Sales amount", fontsize=12)
    ax1.set_xlabel("Lags (Days)", fontsize=12)

    ax2.set_title(f"Partial Autocorrelation {df.name}", fontsize=15, pad=10)
    ax2.set_ylabel("Sales amount", fontsize=12)
    ax2.set_xlabel("Lags (Days)", fontsize=12)
    
    # Legend & Grid
    ax1.grid(linestyle=":", color='grey')
    ax2.grid(linestyle=":", color='grey')

    plt.show()
```
![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/42e6d067-e8e2-4957-a9f1-714b32cb9f58)

**Observation:**
ACF plot:
 - It shows that there are a lot of significant lags. *In ACF plot none of the lags are becoming zero.* **It means that our data is not stationary as we have explained using statistic test and observation of rolling mean and standard deviation.**
 - It will be hard to determing the AR and MA order, we will need to difference it so that we can identify some significant lags.
 - We can see that Lag peaks after evry 7 days. This is the seasonlity of the model.
    
PACF plot:
 - PACF model has a few significant lags but the plot is not decaying much and has a very little oscillation. So it is hard to say or identify if moving averages can be utilized on this model.
 
We will try to plot the ACF and PACF plot by double differncing means differencing the day_difference with seasonal differnce data.

##### Applying SARIMA Model
The SARIMA model is specified 

$$SARIMA(p, d, q) \times (P, D, Q)_s$$

Where:
- Trend Elements are:
    - p: Autoregressive order
    - d: Difference order
    - q: Moving average order
- Seasonal Elements are:
    - P: Seasonal autoregressive order.
    - D: Seasonal difference order. D=1 would calculate a first order seasonal difference
    - Q: Seasonal moving average order. Q=1 would use a first order errors in the model
    - s: Single seasonal period

#### Theoretical estimates:
- **s**: In our PACF plot there is peak that reappears every 7 days. Thus, we can set seasonal period to **s = 7**. This also backed by our seasonal component after additive decomposition.
- **p**: We observed that there is some tappering in ACF plot and we found the significant lags of 1,2,3 from PACF plot. We can start with **p=1** and see how it works. 
- **d**: We observed that our series has some trend, so we can remove it by differencing, so **d = 1**.
- **q**: Based on our ACF correlations we can set **q = 1** since its the most significant lag. 
- **P**: **P = 0** as we are using ACF plot to find seasonl significant lag. 
- **D**: Since we are dealing with seasonality and we need to differnce the series, **D = 1**
- **Q**: The seasonal moving average will be set to **Q = 1** as we found only one significant seasonal lag in ACF plot. 
Here we go:

$$ SARIMA(1, 1, 1) \times (0, 1, 1)_{7} $$

#### Baseline Sarima Model
```
# Set Hyper-parameters
p, d, q = 1, 1, 1
P, D, Q = 0, 1, 1
s = 7

# Fit SARIMA
sarima_model = SARIMAX(train_df['total_amount'], order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_model_fit = sarima_model.fit(disp=0)
print(sarima_model_fit.summary())
```
![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/537453ee-6611-4229-84f7-e68b8ee97b65)

#### Observations:
- **The standardize residual plot:**  The residuals appear as white noise. It looks like the residual of the decomposed time series.
- **The Normal Q-Q-plot:** Shows that the ordered distribution of residuals follows the linear trend of the samples taken from a standard normal distribution with N(0, 1). There are some outlier as we have seen earlier.
- **Histogram and estimated density plot:**  The KDE follows the N(0,1) line however with noticeable differences. As mentioned before our distribution has heavier tails.
- **The Correlogram plot:** Shows that the time series residuals have low correlation with lagged versions of itself. Meaning there are no patterns left to extract in the residuals.

Lets test the model on our training set:

#### Plotting Predictions and Evaluating SARIMA Model
**Prediction using SARIMA**
```
# defining prediction period
pred_start_date = test_df.index[0]
pred_end_date = test_df.index[-1]

sarima_predictions = sarima_model_fit.predict(start=pred_start_date, end=pred_end_date)
sarima_residuals = test_df['total_amount'] - sarima_predictions
```

**Evaluation of SARIMA**
```
# Get evaluation data
sarima_root_mean_squared_error = rmse_metrics(test_df['total_amount'], sarima_predictions)
sarima_mape_error = mape_metrics(test_df['total_amount'], sarima_predictions)

print(f'Root Mean Squared Error | RMSE: {sarima_root_mean_squared_error}')
print(f'Mean Absolute Percentage Error | MAPE: {sarima_mape_error}')
```
Root Mean Squared Error | RMSE: 13810.6
Mean Absolute Percentage Error | MAPE: 68.99
We are able to get a MAPE of 69.99 % and RMSE of 13810.6.

#### SARIMA Forecast
We will try to forecast the sales for next 180 days. We have the 121 days known from our test data and we will try to see what our model forcasts for next 60 days.
```
# Forecast Window
days = 180

sarima_forecast = sarima_model_fit.forecast(days)
sarima_forecast_series = pd.Series(sarima_forecast, index=sarima_forecast.index)

# Since negative orders are not possible we can trim them.
sarima_forecast_series[sarima_forecast_series < 0] = 0
```

**Plotting Forecast using baseline SARIMA**
```
plot_forecast(train_df['total_amount'], test_df['total_amount'], sarima_forecast_series)
```
![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/0f4f94dd-6ff9-4e6f-a8cb-a4c1c190f23e)

Observations:
- The model predicts the overall daily patterns pretty well. 
- Is not performning well to pick up the variation between weeks and months.
- It positively trending and is not capturing the peaks and toughs.
- We will need to tune it further and should also add another feature holiday so that it can pick some informations from that.
- While this model doesn't have a great long term predictive power it can serve as a solid baseline for our next models.

### 4. TIME SERIES MODELLING WITH FB PROPHET
FB Prophet is a forecasting package in Python that was developed by Facebook’s data science research team. The goal of the package is to give business users a powerful and easy-to-use tool to help forecast business results without needing to be an expert in time series analysis. We will apply this model and see how it performs.

**Preparing data for FB Prophet**
Faecbook prophet needs data in a certain format to be able to process it. The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement here in our case it is total_amount.
```
#preparing the dataframe for fbProphet

prophet_df=dfex['total_amount'].reset_index()
prophet_df.rename(columns={"index": "ds", "total_amount": "y"}, inplace=True)

#using our original train_df and test_df we will convert them into prophet train andt test set.
prophet_train = train_df["total_amount"].reset_index()
prophet_train.rename(columns={"order_purchase_timestamp": "ds", "total_amount": "y"}, inplace=True)
prophet_test = test_df["total_amount"].reset_index()
prophet_test.rename(columns={"order_purchase_timestamp": "ds", "total_amount": "y"}, inplace=True)
```

**Applying a Baseline FB Prophet**
Since we observed that our data has positive trend and seasonality, we will set growth ='linear' and let the model find out appropriate seasonality by making yearly_seaonality, daily_seasonality and weekly_seasonality = True.
```
#instantiate the model
fb_baseline = Prophet(growth='linear', 
                yearly_seasonality=True, 
                daily_seasonality=True, 
                weekly_seasonality=True)
fb_baseline.fit(prophet_train)
```

**Predictions using baseline Prophet**
```
#make predictions dataframe 
future_base = fb_baseline.make_future_dataframe(periods=len(test_df), freq="D")
#make a forecast
forecast_base = fb_baseline.predict(future_base)
forecast_base[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

**Plotting and Evaluating Baseline model**
```
#evaluating on test set
fb_baseline_mape = mape_metrics(prophet_test['y'], forecast_base[-121:].reset_index()['yhat'] )
fb_baseline_rmse = rmse_metrics(prophet_test['y'], forecast_base[-121:].reset_index()['yhat'] )

print(f'Root Mean Squared Error | RMSE: {fb_baseline_rmse}')
print(f'Mean Absolute Percentage Error | MAPE: {fb_baseline_mape}')
```
Root Mean Squared Error | RMSE: 14904.05
Mean Absolute Percentage Error | MAPE: 75.28

**Plotting the forecast using Baseline FB Prophet**
```
from fbprophet.plot import plot_plotly

fig = plot_plotly(fb_baseline, forecast_base) 
fig.update_layout(
    title="Daily Sales amount",
    xaxis_title="Date",
    yaxis_title="Revenue amount"
    )
# fig.show()
fig.show("svg")
```

![image](https://github.com/bsdr18/Predictive-Analytics-for-Sales-and-Marketing/assets/76464269/66f0d14c-b462-494c-a5fa-ffd87351dfaf)

Observations:
- Although the prophet didn't give us a good MAPE or RMSE yet form the plot we can see that it is able to capture seasonality, trend, some peaks and troughs.
- It is worth to explore futher by tuning the hyper parameters and include the holiday impact.

### 5. ISSUES WITH HOURLY SAMPLED DATA
In an attempt to increase the data points, I tried to resample the dataset at hourly level and discovered that I got a lot of zero values at certain time of day because no order was placed during that time. I tried applying SAIMA model but it resulted in negative predictions and decreasing trend. Upon further reading and consulting, found that we will need to do some transfromations or apply differnt approaches to handle such data. Therfore, I limited myself to the daily data only. 

### 6. CONCLUSION
#### Summary:

| Model                          | MAPE     | 
| -------------------------------|:--------:| 
| SARIMA(1,1,1)(0,1,1)(7)        | 68.99    | 
| Baseline Prophet               | 71.78    |   
| Baseline Prophet with holiday  | 77.88    |   
