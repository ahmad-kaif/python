import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('sample_data.csv')

# Display the first 5 rows
print("Dataset Preview:")
print(df.head())

# Check the shape of the dataset
print("\nShape of Dataset (Rows, Columns):")
print(df.shape)


# View column names
print("\nColumns:")
print(df.columns)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Basic statistics
print("\nDataset Statistics:")
print(df.describe())


# Filter rows where Confirmed cases > 2000
filtered_df = df[df['Confirmed'] > 2000]
print("\nFiltered Data (Confirmed > 2000):")
print(filtered_df)

# Sort by Confirmed cases in descending order
sorted_df = df.sort_values('Confirmed', ascending=False)
print("\nSorted Data by Confirmed Cases (Descending):")
print(sorted_df)

# Plot Confirmed cases for each country
plt.bar(df['Country'], df['Confirmed'])
plt.xlabel('Country')
plt.ylabel('Confirmed Cases')
plt.title('Confirmed Cases by Country')
plt.show()