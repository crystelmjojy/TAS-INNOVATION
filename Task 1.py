import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 

os.getcwd()
os.chdir("C:\\Users\\Master\\Desktop\\Datasets")

# Load the dataset
covid = pd.read_csv('covid_data.csv')

print(covid.head())
print(covid.isnull().sum())
data=covid.drop(["Province/State"],axis=1)
print(data.columns)
print(data.dtypes)


# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

#Line Chart of Confirmed Cases over Time (Confirmed cases vs Date)
plt.figure(figsize=(6, 3))
plt.plot(data['Date'], data['Confirmed'], marker='o')
plt.title('Confirmed Cases over Time')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.show()

# Bar Chart of Top 10 Countries by Total Deaths
top_10_countries = data.groupby('Country/Region')['Deaths'].sum().nlargest(10)
plt.figure(figsize=(6, 3))
top_10_countries.plot(kind='bar')
plt.title('Top 10 Countries by Total Deaths')
plt.xlabel('Country')
plt.ylabel('Total Deaths')
plt.show()

# Pie Chart of the distribution of cases (Confirmed, Deaths, Recovered)
global_totals = data[['Confirmed', 'Deaths', 'Recovered']].sum()
plt.figure(figsize=(6, 4))
plt.pie(global_totals, labels=global_totals.index, autopct='%1.1f%%', startangle=140)
plt.title('Global Cases Distribution')
plt.axis('equal')
plt.show()

# Scatter Plot of Confirmed Cases vs. Deaths
plt.figure(figsize=(5, 4))
plt.scatter(data['Confirmed'], data['Deaths'], color='r')
plt.title('Total Confirmed Cases vs. Total Deaths')
plt.xlabel('Total Confirmed Cases')
plt.ylabel('Total Deaths')
plt.grid(True)
plt.show()

# Box Plot of distribution of Deaths by WHO region
plt.figure(figsize=(5, 4))
sns.boxplot(x='WHO Region', y='Deaths', data=data)
plt.title('Distribution of Deaths by WHO Region')
plt.xlabel('WHO Region')
plt.ylabel('Deaths')
plt.xticks(rotation=45)
plt.show()