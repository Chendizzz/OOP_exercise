import pandas as pd
'''clean the data and rewrite the csv
'''
df = pd.read_csv("D:\\ML_data\\raw_csv_data\\Automobile_data.csv", na_values={
'price':["?","n.a"],
'stroke':["?","n.a"],
'horsepower':["?","n.a"],
'peak-rpm':["?","n.a"],
'average-mileage':["?","n.a"]})
print (df)

# df.to_csv("D:\\ML_data\\raw_csv_data\\Automobile_data.csv")

'''Find the most expensive car company name
'''
df = pd.read_csv("D:\\ML_data\\raw_csv_data\\Automobile_data.csv")
c_p = df[['company', 'price']][df.price == df['price'].max()]
print(c_p)

'''Print All Toyota Cars details
'''
df = pd.read_csv("D:\\ML_data\\raw_csv_data\\Automobile_data.csv")
car_manufactor = df.groupby('company')
toyota = car_manufactor.get_group('toyota')
print(toyota)

'''Count total cars per company
'''
print(df['company'].value_counts())

'''Find each company’s Highest price car
'''

h_price = car_manufactor['price'].max()
print(h_price)

'''Find the average mileage of each car making company
'''

average_mil = car_manufactor['average-mileage'].mean()
print(average_mil)

''' Sort all cars by Price column
'''
sort_by_price = df.sort_values(by=['price'], ascending=False)
print(sort_by_price.head(5))

'''Concatenate two data frames using the following conditions
'''
GermanCars = {'Company': ['Ford', 'Mercedes', 'BMV', 'Audi'], 'Price': [23845, 171995, 135925 , 71400]}
japaneseCars = {'Company': ['Toyota', 'Honda', 'Nissan', 'Mitsubishi '], 'Price': [29995, 23600, 61500 , 58900]}

cdf_g = pd.DataFrame.from_dict(GermanCars)
cdf_j = pd.DataFrame.from_dict(japaneseCars)

print(pd.concat([cdf_g, cdf_j], keys=['germany', 'japan']))

'''Merge two data frames using the following condition
'''

Car_Price = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'Price': [23845, 17995, 135925 , 71400]}
car_Horsepower = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'horsepower': [141, 80, 182 , 160]}

cdf_0 = pd.DataFrame.from_dict(Car_Price)
cdf_1 = pd.DataFrame.from_dict(car_Horsepower)

print(pd.merge(cdf_0, cdf_1, on='Company'))