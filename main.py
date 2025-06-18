import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.ticker import FuncFormatter

# --- Ρυθμίσεις ---
ticker = 'DTE.DE'
# Ορίζουμε μια σταθερή ημερομηνία έναρξης για αναπαραγωγιμότητα
start_date = '2020-06-18'
# Η ημερομηνία τέλους θα είναι η τρέχουσα
end_date = pd.to_datetime('today')

print(f"Ανάκτηση δεδομένων για το σύμβολο {ticker} από {start_date} έως σήμερα...")

# --- 1. Ανάκτηση και Οπτικοποίηση Ιστορικών Δεδομένων ---
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    print(f"Σφάλμα: Δεν βρέθηκαν δεδομένα για το σύμβολο {ticker}.")
else:
    plt.style.use('seaborn-v0_8-grid')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data['Close'], label='Τιμή Κλεισίματος DTE.DE', color='darkblue')
    ax.set_title(f'Ιστορική Πορεία Μετοχής Deutsche Telekom ({ticker})', fontsize=16)
    ax.set_ylabel('Τιμή σε EUR (€)', fontsize=12)
    ax.set_xlabel('Ημερομηνία', fontsize=12)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'€{y:.2f}'))
    plt.show()

    # --- 2. Μοντέλο GARCH (Πρόβλεψη Μεταβλητότητας/Ρίσκου) ---
    returns = data['Close'].pct_change().dropna()
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    forecast = garch_fit.forecast(horizon=30)
    # Παίρνουμε την τελευταία πρόβλεψη και τη μετατρέπουμε σε ετήσια βάση
    predicted_volatility = np.sqrt(forecast.variance.iloc[-1])
    annualized_vol = predicted_volatility * np.sqrt(252) * 100

    print("\n--- Μοντέλο GARCH (Πρόβλεψη Κινδύνου/Μεταβλητότητας) ---")
    # Το .iloc[0] είναι για να πάρουμε τον αριθμό από τη σειρά του pandas
    print(f"Η προβλεπόμενη ετησιοποιημένη μεταβλητότητα για τις επόμενες 30 ημέρες είναι: {annualized_vol.iloc[0]:.2f}%")
    print("Ερμηνεία: Αυτό το ποσοστό αντιπροσωπεύει την αναμενόμενη διακύμανση της μετοχής. Μεγαλύτερο ποσοστό σημαίνει μεγαλύτερο ρίσκο.")


    # --- 3. Προσομοίωση Monte Carlo (Εύρος Πιθανών Τιμών) ---
    last_price = data['Close'][-1]
    num_simulations = 1000
    num_days = 252 # Ένα έτος συναλλαγών
    mu = returns.mean()
    sigma = returns.std()
    simulation_df = pd.DataFrame()

    for x in range(num_simulations):
        price_series = [last_price]
        for y in range(num_days):
            # Υπολογίζουμε την επόμενη τιμή με βάση την προηγούμενη
            price = price_series[-1] * (1 + np.random.normal(mu, sigma))
            price_series.append(price)
        simulation_df[f'Σενάριο {x+1}'] = price_series[1:]

    # Σχεδιάζουμε το γράφημα
    fig_mc, ax_mc = plt.subplots(figsize=(14, 7))
    ax_mc.plot(simulation_df)
    ax_mc.set_title(f'Προσομοίωση Monte Carlo για την DTE.DE (1000 σενάρια για 1 χρόνο)', fontsize=16)
    ax_mc.set_ylabel('Πιθανή Τιμή Μετοχής (€)', fontsize=12)
    ax_mc.set_xlabel('Μελλοντικές Ημέρες Συναλλαγών', fontsize=12)
    ax_mc.axhline(y=last_price, color='r', linestyle='--', label=f'Τρέχουσα Τιμή (€{last_price:.2f})')
    ax_mc.legend([ax_mc.get_legend_handles_labels()[0][0]], [ax_mc.get_legend_handles_labels()[1][0]]) # Fix for many labels
    ax_mc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'€{y:.2f}'))
    plt.show()

    # Υπολογίζουμε τα στατιστικά
    end_prices = simulation_df.iloc[-1]
    percentile_5 = end_prices.quantile(0.05)
    percentile_95 = end_prices.quantile(0.95)
    mean_price = end_prices.mean()

    print("\n--- Προσομοίωση Monte Carlo (Εύρος Πιθανών Τιμών σε 1 Έτος) ---")
    print(f"Μέση τιμή προσομοίωσης σε 1 έτος: €{mean_price:.2f}")
    print(f"Το 90% των σεναρίων έδειξε τιμή μεταξύ €{percentile_5:.2f} και €{percentile_95:.2f}.")
    print("Ερμηνεία: Το μοντέλο δεν μας λέει πού θα πάει η τιμή, αλλά μας δείχνει το τεράστιο εύρος των πιθανών εκβάσεων, οπτικοποιώντας την αβεβαιότητα.")


    # --- 4. Μοντέλο ARIMA (Βραχυπρόθεσμη Πρόβλεψη) ---
    model_arima = ARIMA(data['Close'], order=(5, 1, 0))
    model_arima_fit = model_arima.fit()
    forecast_arima = model_arima_fit.get_forecast(steps=30)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
    forecast_series = pd.Series(forecast_arima.predicted_mean.values, index=forecast_index)
    conf_int = forecast_arima.conf_int()
    conf_int.index = forecast_index

    # Σχεδιάζουμε το γράφημα
    fig_arima, ax_arima = plt.subplots(figsize=(14, 7))
    ax_arima.plot(data['Close'][-100:], label='Ιστορικές Τιμές')
    ax_arima.plot(forecast_series, label='Πρόβλεψη ARIMA', color='red')
    ax_arima.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.7, label='Διάστημα Εμπιστοσύνης (95%)')
    ax_arima.set_title('Πρόβλεψη ARIMA για 30 Ημέρες με Διάστημα Εμπιστοσύνης', fontsize=16)
    ax_arima.set_ylabel('Τιμή σε EUR (€)', fontsize=12)
    ax_arima.legend()
    ax_arima.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'€{y:.2f}'))
    plt.show()

    print("\n--- Μοντέλο ARIMA (Βραχυπρόθεσμη Πρόβλεψη) ---")
    print("Ερμηνεία: Δώστε προσοχή όχι στην κόκκινη γραμμή (πρόβλεψη), αλλά στην ροζ περιοχή (αβεβαιότητα). Όσο πιο μακριά πάμε στο μέλλον, τόσο πιο πλατιά γίνεται, δείχνοντας ότι η ακρίβεια της πρόβλεψης καταρρέει γρήγορα.")