import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
import seaborn as sns

print(pd.read_csv("Hotel Reservations.csv"))
df = pd.read_csv("Hotel Reservations.csv")
print("The most popular month of arrival:",np.nanmedian(df["arrival_month"]))
unique_values, counts = np.unique(df["room_type_reserved"], return_counts=True)
mode_value = unique_values[np.argmax(counts)]
print(f"Mode of room_type_reserved: {mode_value}")
plt.figure(figsize=(10, 6))
plt.bar(unique_values, counts, color='skyblue')
plt.title('Comparision of type of meal and price of room')
plt.xlabel('type_of_meal_plan')
plt.ylabel('avg_price_per_room')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.boxplot(df['avg_price_per_room'], vert=True)
plt.title('Box Plot of Average Price per Room and Room type')
plt.xlabel('average_price_per_room')
plt.ylabel('room_type')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(df['lead_time'])

sns.histplot(x='booking_status', data=df, stat='probability')

plt.title('Lead time compared with Booking status')
plt.xlabel('Lead time/Booking Status')
plt.ylabel('Density/Probability')
plt.show()
numerical_cols = ['lead_time', 'avg_price_per_room', 'no_of_adults', 'no_of_special_requests']
pair_plot_df = df[numerical_cols]

sns.pairplot(pair_plot_df)
plt.show()


column_name = 'lead_time'
row_name = 'booking_status'

plt.figure(figsize=(10, 6))

for status in df[row_name].unique():
    plt.hist(df[df[row_name] == status][column_name], bins=20, alpha=0.5, label=status, edgecolor='black')

plt.title(f'Distribution of {column_name} by {row_name}')
plt.xlabel(column_name)
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

numerical_cols = ['lead_time', 'avg_price_per_room', 'no_of_special_requests']
correlation_df = df[numerical_cols]


correlation_matrix = correlation_df.corr()


plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numeric Features')
plt.show()


grouped_data = df.groupby(['market_segment_type', 'booking_status'])['Booking_ID'].count().unstack()

#
ax = grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title('Booking Status by Market Segment Type')
plt.xlabel('Market Segment Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

plt.legend(title='Booking Status')

plt.tight_layout()
plt.show()
