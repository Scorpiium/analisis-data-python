import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard Bike Sharing")
st.title("🚲 Dashboard Bike Sharing 🚲")

def load_data():
    try:
        df = pd.read_csv("dashboard/main_data.csv")
    except Exception as e:
        st.error("Gagal memuat file main_data.csv. Pastikan file ada di direktori yang sama.")
        return None, None

    if 'dteday' in df.columns and 'hr' in df.columns:
        df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    else:
        st.warning("Kolom 'dteday' dan 'hr' tidak ditemukan, pastikan dataset sesuai format.")

    df_hour_full = df.copy()

    weather_labels = {
        1: "Clear/Few clouds",
        2: "Mist/Cloudy",
        3: "Light Snow/Rain",
        4: "Heavy Rain/Snow"
    }
    if 'weathersit' in df_hour_full.columns:
        df_hour_full['weather_label'] = df_hour_full['weathersit'].map(weather_labels)
    else:
        st.warning("Kolom 'weathersit' tidak ditemukan pada dataset.")

    if 'mnth' in df_hour_full.columns and 'yr' in df_hour_full.columns:
        df_hour_full['Year'] = df_hour_full['yr'].map({0: 2011, 1: 2012})
        monthly_data = df_hour_full.groupby(['mnth', 'Year'], as_index=False)['cnt'].sum()
        monthly_data = monthly_data.rename(columns={'mnth': 'Month', 'cnt': 'Total_Rentals'})
    else:
        if 'datetime' in df_hour_full.columns:
            df_hour_full['Month'] = df_hour_full['datetime'].dt.month
            df_hour_full['Year'] = df_hour_full['datetime'].dt.year
            monthly_data = df_hour_full.groupby(['Month', 'Year'], as_index=False)['cnt'].sum()
            monthly_data = monthly_data.rename(columns={'cnt': 'Total_Rentals'})
        else:
            monthly_data = pd.DataFrame()

    return df_hour_full, monthly_data

df_hour_full, monthly_data = load_data()
if df_hour_full is None or monthly_data is None:
    st.stop()

st.sidebar.header("Filter Tren Peminjaman")
selected_years = st.sidebar.multiselect(
    "Pilih Tahun:",
    options=sorted(monthly_data['Year'].unique()) if not monthly_data.empty else [2011, 2012],
    default=sorted(monthly_data['Year'].unique()) if not monthly_data.empty else [2011, 2012]
)

st.subheader("📄 Dataset")
st.dataframe(df_hour_full.head())

st.subheader("🌻 Pengaruh Kondisi Cuaca terhadap Total Penggunaan Sepeda")
weather_summary = df_hour_full.groupby('weather_label', as_index=False)['cnt'].sum()
weather_order = [
    "Clear/Few clouds",
    "Mist/Cloudy",
    "Light Snow/Rain",
    "Heavy Rain/Snow"
]
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(
    x='weather_label',
    y='cnt',
    data=weather_summary,
    order=weather_order,
    palette='viridis',
    ax=ax1
)
ax1.set_title('')
ax1.set_xlabel('Kondisi Cuaca')
ax1.set_ylabel('Total Jumlah Peminjaman (Juta)')
plt.xticks(rotation=45)
for p in ax1.patches:
    height = p.get_height()
    ax1.annotate(f'{height:,.0f}',
                 (p.get_x() + p.get_width() / 2., height),
                 ha='center', va='bottom', fontsize=8, color='black')
st.pyplot(fig1)

st.subheader("📈 Tren Peminjaman Sepeda Per Bulan")
if not monthly_data.empty:
    filtered_data = monthly_data[monthly_data['Year'].isin(selected_years)]
    if filtered_data.empty:
        st.warning("Data tidak tersedia untuk tahun yang dipilih. Silakan pilih tahun lain.")
    else:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=filtered_data,
            x='Month',
            y='Total_Rentals',
            hue='Year',
            marker='o',
            palette={2011: 'blue', 2012: 'red'},
            ax=ax2
        )
        ax2.set_title("")
        ax2.set_xlabel("Bulan")
        ax2.set_ylabel("Total Peminjaman")
        ax2.set_xticks(np.arange(1, 13, 1))
        ax2.legend(title='Tahun')
        st.pyplot(fig2)
else:
    st.warning("Data agregat bulanan tidak tersedia.")

monthly_data_pivot = df_hour_full.groupby(['yr', 'mnth'])['cnt'].sum().reset_index()
monthly_data_pivot['Year'] = monthly_data_pivot['yr'].map({0: 2011, 1: 2012})
monthly_data_pivot.rename(columns={'mnth': 'Month', 'cnt': 'Total_Rentals'}, inplace=True)

pivot_table = monthly_data_pivot.pivot(index='Month', columns='Year', values='Total_Rentals')

pivot_table['Change'] = pivot_table[2012] - pivot_table[2011]
pivot_table['Change_str'] = pivot_table['Change'].apply(lambda x: f"+{x:,.0f}" if x >= 0 else f"{x:,.0f}")

def calc_pct(row):
    if row[2011] == 0:
        return None
    return (row[2012] - row[2011]) / row[2011] * 100

pivot_table['Change_pct'] = pivot_table.apply(calc_pct, axis=1)
pivot_table['Change_pct_str'] = pivot_table['Change_pct'].apply(
    lambda x: f"{x:+.2f}%" if x is not None and pd.notnull(x) else "N/A"
)

final_table = pivot_table[[2011, 2012, 'Change_str', 'Change_pct_str']].rename(
    columns={2011: "2011", 2012: "2012", 'Change_str': "Change (abs)", 'Change_pct_str': "Change (%)"}
)

total_2011 = final_table["2011"].sum()
total_2012 = final_table["2012"].sum()
total_change = total_2012 - total_2011
total_change_str = f"+{total_change:,.0f}" if total_change >= 0 else f"{total_change:,.0f}"
if total_2011 != 0:
    total_pct = (total_2012 - total_2011) / total_2011 * 100
    total_pct_str = f"{total_pct:+.2f}%"
else:
    total_pct_str = "N/A"

total_row = pd.DataFrame({
    "2011": [total_2011],
    "2012": [total_2012],
    "Change (abs)": [total_change_str],
    "Change (%)": [total_pct_str]
}, index=["Total"])

final_table = pd.concat([final_table, total_row])

st.dataframe(final_table)
