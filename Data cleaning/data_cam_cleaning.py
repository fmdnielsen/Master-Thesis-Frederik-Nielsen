# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:25:51 2023

@author: fmdni
"""

#%% Reading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Loading data
df1 = pd.read_csv("ele.csv")
df2 = pd.read_csv("site_weather.csv")

df1['date'] = pd.to_datetime(df1['date'], format='%d/%m/%Y %H:%M')
df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d %H:%M:%S')

df = pd.merge(df2, df1, on='date', how='left')

#%% Sum electricity consumption coloumn
columns_to_sum = [6,7,8,9,10]
df['Electricity_cons'] = df.iloc[:, columns_to_sum].sum(axis=1)



# Drop the original columns
df.drop(df.columns[columns_to_sum], axis=1, inplace=True)
# Drop double air temp and nan coloumn
df.drop(df.columns[[2]], axis=1, inplace=True)

#%% Renaiming columns
df.columns = ['date', 'air_temp', 'dew_point_temp', 'humidity', 'solar_radiation', 'electricity_cons']



#%% Convert from 15 minute interval to 1 hour

# Define a custom aggregation function to take the first value within each hour
def first_value(series):
    return series.iloc[0]

# Resample to hourly intervals, using different aggregation functions for each column
hourly_df = df.set_index('date').resample('H').agg({
    'electricity_cons': 'sum',
    'air_temp': first_value,
    'humidity': first_value,
    'solar_radiation': first_value,
    'dew_point_temp': first_value
})

#%% Add day values going from 1 to 7
#hourly_df['new_column'] = range(1, len(hourly_df) + 1)
#hourly_df['new_column'] = hourly_df['new_column'] % 7 + 1


#%% If Electricity conusmption is 0 then call it nan and check for nan values
hourly_df['electricity_cons'] = hourly_df['electricity_cons'].replace(0, pd.NA)
print(hourly_df.isna().sum())
sns.heatmap(hourly_df.isna(), cmap='viridis')
plt.show()
#%% Replace missing consumption values with values from previous year or next year

def replace_nan_with_adjacent_year(series):
    for i in range(len(series)):
        if pd.isna(series.iloc[i]):
            # Check the next year
            next_year_value = series.shift(8760).iloc[i]
            if not pd.isna(next_year_value):
                series.iloc[i] = next_year_value
            else:
                # Check the previous year
                prev_year_value = series.shift(-8760).iloc[i]
                if not pd.isna(prev_year_value):
                    series.iloc[i] = prev_year_value
    return series

# Apply the custom function to the 'value_with_nan' column
hourly_df['electricity_cons'].transform(replace_nan_with_adjacent_year)





#%% Check for outliers

outlier_index = list()

for col in range(hourly_df.shape[1]):
    quan25, quan75 = np.percentile(hourly_df.iloc[:,col], 25), np.percentile(hourly_df.iloc[:,col], 75)
    iqr = quan75 - quan25
    print('Percentiles: 25th=%.3f, 75th=%.3f, iqr=%.3f' % (quan25, quan75, iqr))
    
    #outlier cutoff
    cutoff = iqr * 1.5
    low, high = quan25 - cutoff, quan75 + cutoff
    
    #Finding outliers
    outlier = set([x for x in hourly_df.iloc[:,col] if x < low or x > high])
    outlier_index.extend([i for i, item in enumerate(hourly_df.iloc[:,col].to_numpy()) if item in outlier])
    print('Found outliers: %d' % len(outlier))
    
    #Deleting outliers
    outliers_deleted = [x for x in hourly_df.iloc[:,col] if x >= low and x <= high]
    print('Non-outlier found: %d' % len(outliers_deleted))
    #train = train.drop(outlier_index, axis=0)

outlier_index = np.unique(outlier_index)

#Uncomment if outliers should be removed
#data = data.drop(outliers_index, axis = 0)

#%% plot data
# Get the first four columns
selected_columns = hourly_df.iloc[:, :4]

# Create subplots for each column in a 2x2 grid
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Flatten the 2D array of subplots for easier indexing
axes = axes.flatten()

# Plot each column on its own subplot
for i, col in enumerate(selected_columns.columns):
    axes[i].plot(selected_columns.index, selected_columns[col], label=col)
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('Values')
    axes[i].set_title(f'Plot of {col}')
    axes[i].legend(loc='upper right')  # Add legend if needed

# Adjust layout to prevent overlap of subplots
plt.tight_layout()

# Display the figure
plt.show()

#%% Add features
hourly_df['month'] = hourly_df.index.month
hourly_df['week'] = hourly_df.index.isocalendar().week
hourly_df['hour'] = hourly_df.index.hour
hourly_df['day_of_the_week'] = hourly_df.index.dayofweek
hourly_df['is_weekend'] = (hourly_df['day_of_the_week'] >= 5).astype(int)

def cyclical(data, col, cycle_length):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/cycle_length)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/cycle_length)
    return data

hourly_df = cyclical(hourly_df, "hour", 24)
hourly_df = cyclical(hourly_df, "week", 52)
hourly_df = cyclical(hourly_df, "month", 12)
hourly_df = cyclical(hourly_df, "day_of_the_week", 7)

hourly_df=hourly_df.drop(['hour'], axis=1)
hourly_df=hourly_df.drop(['week'], axis=1)
hourly_df=hourly_df.drop(['month'], axis=1)
hourly_df=hourly_df.drop(['day_of_the_week'], axis=1)
hourly_df = hourly_df[:-1]



#%% average daily data
hourly_df['day_of_the_week'] = hourly_df.index.dayofweek
hourly_df.to_csv('data_4_clean.csv')



average_hourly_consumption = hourly_df.groupby(['day_of_the_week', hourly_df.index.hour])['electricity_cons'].mean().unstack()
Label = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', "Sunday"]
# Plotting
plt.figure(figsize=(12, 6))
for day in average_hourly_consumption.index:
    plt.plot(average_hourly_consumption.columns, average_hourly_consumption.loc[day], label=Label[day],linewidth=2.5)

plt.title('Average Hourly Electricity Consumption by Weekday',fontsize=16)
plt.xlabel('Hour of Day',fontsize=14)
plt.ylabel('Average Electricity Consumption [kW]',fontsize=14)
plt.legend(fontsize=12)
plt.xticks(range(24))
plt.xlim(0, 23)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.grid(True)
plt.savefig('hour_plot.png', dpi=300)
plt.show()

#%%
import seaborn as sns

#Plot heatmap
plt.figure()
plt.figure(figsize = (12,8))

#Heat map for combined data:
sns.heatmap(hourly_df.corr(),annot=True, cbar=False, cmap='coolwarm', fmt='.1f')


#%%
fig, ax1 = plt.subplots(figsize=(10, 6))


# Plot electricity_cons on ax1 with a pleasant blue color
ax1.plot(hourly_df['electricity_cons'], label='Electricity Consumption', color='#1f77b4')
ax1.set_xlabel('Date',fontsize=14)
ax1.set_ylabel('Electricity Consumption (kWh)', color='#1f77b4',fontsize=14)
ax1.tick_params('y', colors='#1f77b4',labelsize=12)
ax1.grid(True)

# Create secondary axis
ax2 = ax1.twinx()

# Plot air_temp on ax2 with a pleasant red color
ax2.plot(hourly_df['air_temp'], label='Air Temperature', color='#d62728', alpha=0.8)
ax2.set_ylabel('Air Temperature (°C)', color='#d62728',fontsize=14)
ax2.tick_params('y', colors='#d62728',labelsize=12)

# Combine legends from ax1 and ax2
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left',fontsize=12)

# Title
plt.title('Electricity Consumption vs Air Temperature',fontsize=16)



plt.margins(x=0)
# Show plot
plt.tight_layout()
plt.savefig('cons_temp.png', dpi=300)
plt.show()





#%%

import pandas as pd
import numpy as np

# Sample hourly_df (replace this with your actual DataFrame)
# hourly_df = pd.read_csv('your_dataframe.csv')

# Resample data by year
yearly_stats = {}

# Overall statistics
overall_stats = {
    'Electricity Consumption (kWh)': [],
    'Temperature (°C)': []
}

for year in range(2018, 2021):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    df_year = hourly_df[start_date:end_date]
    
    # Calculate yearly statistics
    yearly_stats[year] = {
        'Mean Electricity Consumption (kWh)': df_year['electricity_cons'].mean(),
        'Median Electricity Consumption (kWh)': df_year['electricity_cons'].median(),
        'Std Dev Electricity Consumption (kWh)': df_year['electricity_cons'].std(),
        'Min Electricity Consumption (kWh)': df_year['electricity_cons'].min(),
        'Max Electricity Consumption (kWh)': df_year['electricity_cons'].max(),
        'Mean Temperature (°C)': df_year['air_temp'].mean(),
        'Median Temperature (°C)': df_year['air_temp'].median(),
        'Std Dev Temperature (°C)': df_year['air_temp'].std(),
        'Min Temperature (°C)': df_year['air_temp'].min(),
        'Max Temperature (°C)': df_year['air_temp'].max()
    }
    
    # Update overall statistics
    overall_stats['Electricity Consumption (kWh)'].extend(df_year['electricity_cons'].values)
    overall_stats['Temperature (°C)'].extend(df_year['air_temp'].values)

# Calculate overall statistics
overall_stats_df = {
    'Overall Mean Electricity Consumption (kWh)': np.mean(overall_stats['Electricity Consumption (kWh)']),
    'Overall Median Electricity Consumption (kWh)': np.median(overall_stats['Electricity Consumption (kWh)']),
    'Overall Std Dev Electricity Consumption (kWh)': np.std(overall_stats['Electricity Consumption (kWh)']),
    'Overall Min Electricity Consumption (kWh)': np.min(overall_stats['Electricity Consumption (kWh)']),
    'Overall Max Electricity Consumption (kWh)': np.max(overall_stats['Electricity Consumption (kWh)']),
    'Overall Mean Temperature (°C)': np.mean(overall_stats['Temperature (°C)']),
    'Overall Median Temperature (°C)': np.median(overall_stats['Temperature (°C)']),
    'Overall Std Dev Temperature (°C)': np.std(overall_stats['Temperature (°C)']),
    'Overall Min Temperature (°C)': np.min(overall_stats['Temperature (°C)']),
    'Overall Max Temperature (°C)': np.max(overall_stats['Temperature (°C)'])
}

# Convert yearly statistics to DataFrame
yearly_stats_df = pd.DataFrame(yearly_stats).transpose()

# Convert overall statistics to DataFrame
overall_stats_df = pd.DataFrame(overall_stats_df, index=['Overall'])

# Combine yearly and overall statistics
statistics_df = pd.concat([yearly_stats_df, overall_stats_df])

# Save DataFrame to Excel file
statistics_df.to_excel('statistics.xlsx', float_format='%.2f')
