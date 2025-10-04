# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 

# Name: Mohamed Hameem Sajith J
# reg : 212223240090


### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```python
# Import necessary Modules and Functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the Daily Delhi Climate dataset from Kaggle
# Make sure the 'DailyDelhiClimateTrain.csv' file is in the same folder
data = pd.read_csv('DailyDelhiClimateTrain.csv', parse_dates=['date'], index_col='date')

# We select the 'meantemp' column for our analysis.
X = data['meantemp']

# --- Create a single large figure to hold all plots ---
# The figure will have a 3x3 grid of subplots.
plt.figure(figsize=(18, 15))


# --- Plot 1: Original Data ---
plt.subplot(3, 3, 1) # (rows, columns, plot_number)
plt.plot(X)
plt.title('Original Mean Temperature Data')


# --- Plot 2: Partial Autocorrelation (Original) ---
ax2 = plt.subplot(3, 3, 2)
plot_acf(X, lags=40, ax=ax2)
ax2.set_title('Original Data ACF')


# --- Plot 3: Autocorrelation (Original) ---
ax3 = plt.subplot(3, 3, 3)
plot_pacf(X, lags=40, ax=ax3)
ax3.set_title('Original Data PACF')


# --- Fit and Simulate ARMA(1,1) Process ---
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params.get('ar.L1', 0)
theta1_arma11 = arma11_model.params.get('ma.L1', 0)
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
N = len(X)
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)


# --- Plot 4: SIMULATED ARMA(1,1) PROCESS ---
plt.subplot(3, 3, 4)
plt.plot(ARMA_1)
plt.title('SIMULATED ARMA(1,1) PROCESS')


# --- Plot 5: Partial Autocorrelation (ARMA 1,1) ---
ax5 = plt.subplot(3, 3, 5)
plot_pacf(ARMA_1, lags=40, ax=ax5)
ax5.set_title('Partial Autocorrelation')


# --- Plot 6: Autocorrelation (ARMA 1,1) ---
ax6 = plt.subplot(3, 3, 6)
plot_acf(ARMA_1, lags=40, ax=ax6)
ax6.set_title('Autocorrelation')


# --- Fit and Simulate ARMA(2,2) Process ---
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params.get('ar.L1', 0)
phi2_arma22 = arma22_model.params.get('ar.L2', 0)
theta1_arma22 = arma22_model.params.get('ma.L1', 0)
theta2_arma22 = arma22_model.params.get('ma.L2', 0)
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)


# --- Plot 7: SIMULATED ARMA(2,2) PROCESS ---
plt.subplot(3, 3, 7)
plt.plot(ARMA_2)
plt.title('SIMULATED ARMA(2,2) PROCESS')


# --- Plot 8: Partial Autocorrelation (ARMA 2,2) ---
ax8 = plt.subplot(3, 3, 8)
plot_pacf(ARMA_2, lags=40, ax=ax8)
ax8.set_title('Partial Autocorrelation')


# --- Plot 9: Autocorrelation (ARMA 2,2) ---
ax9 = plt.subplot(3, 3, 9)
plot_acf(ARMA_2, lags=40, ax=ax9)
ax9.set_title('Autocorrelation')


# --- Display the final consolidated plot ---
plt.tight_layout()
plt.show()
```
OUTPUT:
# ORGINAL :
<img width="626" height="530" alt="image" src="https://github.com/user-attachments/assets/ef181641-fe36-4fb8-87a4-11aa4486fa5f" />

# Partial Autocorrelation :
<img width="640" height="537" alt="image" src="https://github.com/user-attachments/assets/f3bcd219-07c7-4e6a-9bc2-a7563240d48d" />

# Autocorrelation :
<img width="642" height="519" alt="image" src="https://github.com/user-attachments/assets/3611d45d-3087-4fe9-a947-7b16fe7a15c3" />

# SIMULATED ARMA(1,1) PROCESS:

<img width="466" height="388" alt="image" src="https://github.com/user-attachments/assets/f4f2a38d-d167-42ee-ab28-7f792d7af458" />


# Partial Autocorrelation
<img width="481" height="402" alt="image" src="https://github.com/user-attachments/assets/c3997e88-887c-4f38-a860-13adcbae65d1" />

# Autocorrelation

<img width="488" height="395" alt="image" src="https://github.com/user-attachments/assets/e3951127-eada-4df5-a0e7-92511b3d1044" />


# SIMULATED ARMA(2,2) PROCESS:
<img width="459" height="408" alt="image" src="https://github.com/user-attachments/assets/e454ebfa-9301-4b5f-b0ab-f6a43107650d" />

# Partial Autocorrelation
<img width="469" height="396" alt="image" src="https://github.com/user-attachments/assets/c6e03d0e-09bb-4bd5-b6b8-f9ce501468bf" />



# Autocorrelation
<img width="480" height="394" alt="image" src="https://github.com/user-attachments/assets/d391aacb-ba73-4a89-a8ba-1c682bfd921f" />

# RESULT:
Thus, a python program is created to fir ARMA Model successfully.
