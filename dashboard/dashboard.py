import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Fungsi untuk load data dengan caching agar performa lebih optimal
@st.cache_data
def load_data():
    # Pastikan file main_data.csv berada di direktori yang sama
    df = pd.read_csv("main_data.csv")

    # Konversi kolom tanggal ke tipe datetime
    df['dteday'] = pd.to_datetime(df['dteday'])

    # Mapping untuk musim (season)
    season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    df['season_label'] = df['season'].map(season_mapping)

    # Mapping untuk kondisi cuaca (weathersit)
    weather_mapping = {1: "Cerah / Berawan", 2: "Berkabut / Mendung", 3: "Hujan Ringan / Salju Ringan",
                       4: "Hujan Lebat / Badai"}
    df['weathersit_label'] = df['weathersit'].map(weather_mapping)

    # Jika diperlukan, lakukan preprocessing tambahan berdasarkan analisis pada notebook
    return df


# Load dataset
df = load_data()

# Judul Dashboard
st.title("📊 Dashboard Analisis Bike Sharing")

# Sidebar untuk filter data
st.sidebar.header("🔍 Filter Data")

# Filter berdasarkan rentang tanggal
min_date = df['dteday'].min()
max_date = df['dteday'].max()
selected_date = st.sidebar.date_input("Pilih rentang tanggal", [min_date, max_date], min_value=min_date,
                                      max_value=max_date)

# Filter berdasarkan musim
season_options = df['season_label'].unique()
selected_seasons = st.sidebar.multiselect("Pilih musim", season_options, default=season_options)

# Filter berdasarkan kondisi cuaca
weather_options = df['weathersit_label'].unique()
selected_weathers = st.sidebar.multiselect("Pilih kondisi cuaca", weather_options, default=weather_options)

# Terapkan filter pada data
df_filtered = df[(df['dteday'] >= pd.to_datetime(selected_date[0])) &
                 (df['dteday'] <= pd.to_datetime(selected_date[1])) &
                 (df['season_label'].isin(selected_seasons)) &
                 (df['weathersit_label'].isin(selected_weathers))]

# Tampilkan beberapa baris data yang sudah difilter
st.write("### Data yang Ditampilkan Setelah Filter")
st.dataframe(df_filtered.head())

# Tampilkan metrik total pengguna
total_users = df_filtered['cnt'].sum()
st.metric("🚴 Total Pengguna", f"{total_users:,}".replace(",", "."))

# Visualisasi 1: Pengaruh Suhu terhadap Penggunaan Sepeda
# Asumsi: kolom 'temp' bersifat normalisasi, dikonversi ke nilai suhu aktual (misalnya dengan perkalian 41)
df_filtered['temp_actual'] = df_filtered['temp'] * 41
temp_bins = [0, 8.2, 16.4, 24.6, 32.8, 41]
temp_labels = ['Sangat Dingin (0-8°C)', 'Dingin (8-16°C)', 'Normal (16-24°C)', 'Hangat (24-32°C)', 'Panas (32-41°C)']
df_filtered['temp_group'] = pd.cut(df_filtered['temp_actual'], bins=temp_bins, labels=temp_labels)

st.subheader("📊 Pengaruh Suhu terhadap Penggunaan Sepeda")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='temp_group', y='cnt', data=df_filtered, estimator=sum, palette='coolwarm', ax=ax)
ax.set_xlabel('Kelompok Suhu')
ax.set_ylabel('Jumlah Pengguna')
ax.set_title('Jumlah Pengguna Berdasarkan Kelompok Suhu')
st.pyplot(fig)

# Visualisasi 2: Pengaruh Kelembaban terhadap Penggunaan Sepeda
hum_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
hum_labels = ['Sangat Kering', 'Kering', 'Normal', 'Lembab', 'Sangat Lembab']
df_filtered['hum_group'] = pd.cut(df_filtered['hum'], bins=hum_bins, labels=hum_labels)

st.subheader("📊 Pengaruh Kelembaban terhadap Penggunaan Sepeda")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='hum_group', y='cnt', data=df_filtered, estimator=sum, palette='viridis', ax=ax)
ax.set_xlabel('Kelompok Kelembaban')
ax.set_ylabel('Jumlah Pengguna')
ax.set_title('Jumlah Pengguna Berdasarkan Kelompok Kelembaban')
st.pyplot(fig)

# Visualisasi 3: Perbandingan Pengguna Casual vs Registered (Hari Kerja vs Hari Libur)
grouped_data = df_filtered.groupby('workingday')[['casual', 'registered']].sum().reset_index()
grouped_data['day_type'] = grouped_data['workingday'].map({0: 'Hari Libur', 1: 'Hari Kerja'})

st.subheader("📊 Perbandingan Pengguna Casual vs Registered")
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(grouped_data))
width = 0.4
ax.bar(x - width / 2, grouped_data['casual'], width, label='Casual', color='skyblue')
ax.bar(x + width / 2, grouped_data['registered'], width, label='Registered', color='salmon')
ax.set_xticks(x)
ax.set_xticklabels(grouped_data['day_type'])
ax.set_ylabel('Jumlah Pengguna')
ax.set_title('Pengguna Casual vs Registered berdasarkan Hari')
ax.legend()
st.pyplot(fig)

# Visualisasi 4: Tren Penggunaan Sepeda per Jam
hourly_trend = df_filtered.groupby('hr')['cnt'].mean().reset_index()

st.subheader("📊 Tren Penggunaan Sepeda per Jam")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x='hr', y='cnt', data=hourly_trend, marker='o', color='red', ax=ax)
ax.set_xlabel('Jam dalam Sehari')
ax.set_ylabel('Rata-rata Jumlah Pengguna')
ax.set_title('Tren Penggunaan Sepeda per Jam')
ax.set_xticks(range(0, 24))
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

# Visualisasi 5: Penggunaan Sepeda per Jam Berdasarkan Kondisi Cuaca
hourly_weather = df_filtered.groupby(['hr', 'weathersit_label'])['cnt'].mean().reset_index()
st.subheader("⛅ Penggunaan Sepeda per Jam Berdasarkan Kondisi Cuaca")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x='hr', y='cnt', data=hourly_weather, hue='weathersit_label', marker='o', palette='coolwarm', ax=ax)
ax.set_xlabel('Jam dalam Sehari')
ax.set_ylabel('Rata-rata Jumlah Pengguna')
ax.set_title('Penggunaan Sepeda per Jam berdasarkan Cuaca')
ax.set_xticks(range(0, 24))
ax.legend(title='Kondisi Cuaca')
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

st.markdown("---")
st.markdown("Dashboard ini dibuat berdasarkan analisis mendalam pada notebook dan data utama dari main_data.csv.")
