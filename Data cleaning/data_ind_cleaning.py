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
df_el = pd.read_csv("Meterdata_new.csv")
df_temp = pd.read_excel("weather.xlsx","Temp")
df_hum = pd.read_excel("weather.xlsx","Humidity")
df_rad = pd.read_excel("weather.xlsx","Radiation")


#%% drop coloumn
columns_to_drop = [0,2,4,5,6]

# Drop the original columns
#df_el.drop(df_el.columns[columns_to_drop], axis=1, inplace=True)

hourly_df = df_el

hourly_df['date'] = pd.to_datetime(hourly_df['date'], format='%Y-%m-%d %H:%M:%S')


hourly_df = hourly_df.set_index('date')



# hourly_df['date'] = hourly_df['Fra_dato']

# hourly_df = hourly_df.drop(columns='Fra_dato')

# hourly_df['date'] = pd.to_datetime(hourly_df['date'], format='%Y-%m-%d %H:%M:%S')


# hourly_df = hourly_df.set_index('date').resample('H').mean()

# hourly_df['electricity_cons'] = hourly_df['Mængde']

# hourly_df = hourly_df.drop(columns='Mængde')

#%% Convert from different interval to 1 hour

# Define a custom aggregation function to take the first value within each hour
def first_value(series):
    return series.iloc[0]


#%% Adding weather data and reomving unessecary data

start_date = '2021-01-01'
end_date = '2023-12-31'
hourly_index = pd.date_range(start=start_date, end=end_date, freq='H')
hourly_index = hourly_index[0:-1]

df_temp['Time'] = pd.to_datetime(df_temp['Time'], format='%Y-%m-%d %H:%M:%S')
df_hum['Time'] = pd.to_datetime(df_hum['Time'], format='%Y-%m-%d %H:%M:%S')
df_rad['Time'] = pd.to_datetime(df_rad['Time'], format='%Y-%m-%d %H:%M:%S')

df_temp = df_temp.sort_values(by='Time', ascending=False)
df_hum = df_hum.sort_values(by='Time', ascending=False)
df_rad = df_rad.sort_values(by='Time', ascending=False)

df_temp = df_temp.set_index('Time').resample('H').mean()
df_hum = df_hum.set_index('Time').resample('H').mean()
df_rad = df_rad.set_index('Time').resample('H').mean()


hourly_df = pd.concat([hourly_df, df_temp], axis=1)
hourly_df = pd.concat([hourly_df, df_hum], axis=1)
hourly_df = pd.concat([hourly_df, df_rad], axis=1)

#hourly_df = hourly_df[["electricity_cons","air_temp","dew_point_temp","humidity","solar_radiation"]]

#Converting temperature data to celsius
#hourly_df[['temperature', "dewPoint"]]=(hourly_df[['temperature', "dewPoint"]]-32)/1.8
#%% Renaming columns
hourly_df.columns = ['electricity_cons', 'air_temp' ,'dew_point_temp','humidity', 'solar_radiation']


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

#Replace other nan values by interpolation
hourly_df['dew_point_temp'].interpolate(method='linear', inplace=True)
hourly_df['humidity'].interpolate(method='linear', inplace=True)
hourly_df['solar_radiation'].interpolate(method='linear', inplace=True)
hourly_df['air_temp'].interpolate(method='linear', inplace=True)

#%% Outliers after cleaning
print(hourly_df.isna().sum())


#%% Check for outliers

outlier_index = list()

for col in range(hourly_df.shape[1]):
    quan25, quan75 = np.percentile(hourly_df.iloc[:,col], 25), np.percentile(hourly_df.iloc[:,col], 75)
    iqr = quan75 - quan25
    print('Percentiles: 25th=%.3f, 75th=%.3f, iqr=%.3f' % (quan25, quan75, iqr))
    
    #outlier cutoff
    cutoff = iqr * 2
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


#hourly_df['electricity_cons'].interpolate(method='linear', inplace=True)


#%% plot data
# Get the first four columns
selected_columns = hourly_df.iloc[:, :5]

# Create subplots for each column in a 2x2 grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

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

plt.figure(figsize=(20, 16))
hourly_df.plot.scatter('hour_sin','hour_cos').set_aspect('equal')
plt.savefig('scatter_plot.png', dpi=300)
plt.show()

hourly_df=hourly_df.drop(['hour'], axis=1)
hourly_df=hourly_df.drop(['week'], axis=1)
hourly_df=hourly_df.drop(['month'], axis=1)
hourly_df=hourly_df.drop(['day_of_the_week'], axis=1)
hourly_df = hourly_df[:-1]



#%% Save data
hourly_df.to_csv('data_5_clean.csv')

#%%
import seaborn as sns

#Plot heatmap
plt.figure()
plt.figure(figsize = (15,8))

#Heat map for combined data:
sns.heatmap(hourly_df.corr(),annot=True, cbar=False, cmap='coolwarm', fmt='.1f')

#%%
from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(figsize=(6, 2))
plot_acf(hourly_df["electricity_cons"], ax=ax, lags=60)
plt.show()

#%% average daily data
hourly_df['day_of_the_week'] = hourly_df.index.dayofweek


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

for year in range(2021, 2024):
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
