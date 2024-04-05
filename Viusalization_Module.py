import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define file paths for the additional 8 files
file_paths = [
    r"E:\Fault detection at rotatory Machinery dataset\0D.csv",
    r"E:\Fault detection at rotatory Machinery dataset\0E.csv",
    r"E:\Fault detection at rotatory Machinery dataset\1D.csv",
    r"E:\Fault detection at rotatory Machinery dataset\1E.csv",
    r"E:\Fault detection at rotatory Machinery dataset\2D.csv",
    r"E:\Fault detection at rotatory Machinery dataset\2E.csv",
    r"E:\Fault detection at rotatory Machinery dataset\3D.csv",
    r"E:\Fault detection at rotatory Machinery dataset\3E.csv",
    r"E:\Fault detection at rotatory Machinery dataset\4D.csv",
    r"E:\Fault detection at rotatory Machinery dataset\4E.csv",
]

# Loop through each dataset
for file_path in file_paths:
    # Extract filename from file path
    file_name = os.path.basename(file_path).split('.')[0]
    # os.path.basename is a function from os module.it takes file path as input and return file name without
    # directory path. split function is used to split the file name into parts using dot as delimiter
    # 0 selects the first part of the split and leaving only the base name.

    # Load data
    df = pd.read_csv(file_path, dtype=float)

    # Fill missing values with the mean
    df.fillna(df.mean(), inplace=True)

    # Downsample the data
    downsample_factor = 10 #reducing the number of samples.for every 10th value 1 value is selected
    df_downsampled = df.iloc[::downsample_factor, :]#This will take only the downsampled rows but contain all columns

    # Createing a new figure for the entire dataset
    fig, axes = plt.subplots(nrows=2, ncols=len(df_downsampled.columns), figsize=(15, 8))

    # Plot each signal and its corresponding FFT in subplots
    for j, col in enumerate(df_downsampled.columns):
        #This code loop through each column in a dataframe and assign index (j) and column name to each column

        # Plot the time-domain signal
        axes[0, j].plot(df_downsampled[col])
        #j indicates the column index and 0indicates the first row of subplot.Each data gegts its own subplot.
        axes[0, j].set_title(col)

        # Compute the FFT
        fft_values = np.fft.fft(df_downsampled[col])#computing the fft of the columns.
        fft_freq = np.fft.fftfreq(len(df_downsampled[col]))#Compute the frequency value as computed earlier,
        #base on the length of the data

        # Plot the frequency spectrum
        axes[1, j].plot(fft_freq, np.abs(fft_values)) #1,j tells where to draw the plot in the second row and jth column
        #np.abs helps to find out the magnitude of each frequency component in sinnal regardless of positive or negative
        axes[1, j].set_title(f"FFT of {col}")
        axes[1, j].set_xlabel('Frequency')
        axes[1, j].set_ylabel('Amplitude')
        axes[1, j].grid(True) #adding grid tos ubplot at the secondth row and jth column

    # Add title to the entire figure (group of subplots)
    fig.suptitle(f"Dataset: {file_name}")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
