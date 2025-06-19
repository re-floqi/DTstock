import yfinance as yf
import pandas as pd
from pandas.errors import PerformanceWarning
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from matplotlib.ticker import FuncFormatter
import warnings
import traceback

# --- Ρυθμίσεις ---
ticker = 'DTE.DE'
start_date = '2020-06-18'
end_date = pd.to_datetime('today')

# Αγνοούμε τις προειδοποιήσεις για καθαρότερο αποτέλεσμα
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=PerformanceWarning)
warnings.filterwarnings('ignore', message="y is poorly scaled*")

print(f"Ανάκτηση δεδομένων για το σύμβολο {ticker} από {start_date} έως σήμερα...")

try:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"Δεν βρέθηκαν δεδομένα για το σύμβολο {ticker}.")

    # --- ΔΗΜΙΟΥΡΓΙΑ ΚΑΙ ΑΠΟΘΗΚΕΥΣΗ 1ου ΓΡΑΦΗΜΑΤΟΣ ---
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data['Close'], label='Τιμή Κλεισίματος DTE.DE', color='darkblue')
    ax.set_title(f'Ιστορική Πορεία Μετοχής Deutsche Telekom ({ticker})', fontsize=16)
    ax.set_ylabel('Τιμή σε EUR (€)', fontsize=12)
    ax.set_xlabel('Ημερομηνία', fontsize=12)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'€{y:.2f}'))
    
    hist_filename = 'historical_chart.png'
    plt.savefig(hist_filename)
    plt.close(fig)
    print(f"\nΤο γράφημα της ιστορικής πορείας αποθηκεύτηκε ως '{hist_filename}'")

    # --- 2. GARCH Model ---
    returns = data['Close'].pct_change().dropna() * 100
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp='off')
    forecast = garch_fit.forecast(horizon=30)
    predicted_volatility = np.sqrt(forecast.variance.iloc[-1]) / 100
    annualized_vol = predicted_volatility * np.sqrt(252) * 100

    print("\n--- Μοντέλο GARCH (Πρόβλεψη Κινδύνου/Μεταβλητότητας) ---")
    print(f"Η προβλεπόμενη ετησιοποιημένη μεταβλητότητα για τις επόμενες 30 ημέρες είναι: {float(annualized_vol.iloc[0]):.2f}%")

    # --- 3. Προσομοίωση Monte Carlo για 4 Έτη ---
    returns_mc = data['Close'].pct_change().dropna()
    last_price = data['Close'].iloc[-1].item()
    
    # --- Η ΑΛΛΑΓΗ ΕΙΝΑΙ ΕΔΩ ---
    num_simulations = 10000
    
    num_days = 252 * 4
    mu = returns_mc.mean()
    sigma = returns_mc.std()
    
    print(f"\nΕκτέλεση βελτιστοποιημένης προσομοίωσης Monte Carlo για 4 έτη με {num_simulations} σενάρια...")

    random_returns = np.random.normal(mu, sigma, (num_days, num_simulations))
    price_paths = np.zeros((num_days + 1, num_simulations))
    price_paths[0] = last_price
    
    for t in range(1, num_days + 1):
        price_paths[t] = price_paths[t-1] * (1 + random_returns[t-1])
        
    simulation_df = pd.DataFrame(price_paths[1:], columns=[f'Σενάριο {i+1}' for i in range(num_simulations)])

    # --- ΔΗΜΙΟΥΡΓΙΑ ΚΑΙ ΑΠΟΘΗΚΕΥΣΗ 2ου ΓΡΑΦΗΜΑΤΟΣ ---
    fig_mc, ax_mc = plt.subplots(figsize=(14, 8))
    # Για 10,000 γραμμές, μειώνουμε κι άλλο το alpha για να φαίνεται καλύτερα
    ax_mc.plot(simulation_df, color='blue', alpha=0.01) 
    ax_mc.set_title(f'Προσομοίωση Monte Carlo για την DTE.DE ({num_simulations} σενάρια για 4 Χρόνια)', fontsize=16)
    ax_mc.set_ylabel('Πιθανή Τιμή Μετοχής (€)', fontsize=12)
    ax_mc.set_xlabel('Μελλοντικές Ημέρες Συναλλαγών', fontsize=12)
    ax_mc.axhline(y=last_price, color='r', linestyle='--', label=f'Τρέχουσα Τιμή (€{float(last_price):.2f})')
    handles, labels = ax_mc.get_legend_handles_labels()
    ax_mc.legend([handles[0]], [labels[0]])
    ax_mc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'€{y:.2f}'))

    mc_filename = 'monte_carlo_10k_4_years.png'
    plt.savefig(mc_filename)
    plt.close(fig_mc)
    print(f"Το γράφημα της προσομοίωσης Monte Carlo αποθηκεύτηκε ως '{mc_filename}'")

    # --- Εκτύπωση τελικών αποτελεσμάτων ---
    end_prices = simulation_df.iloc[-1]
    percentile_5 = end_prices.quantile(0.05)
    percentile_95 = end_prices.quantile(0.95)
    mean_price = end_prices.mean()

    print("\n--- Προσομοίωση Monte Carlo (Εύρος Πιθανών Τιμών σε 4 Έτη) ---")
    print(f"Τρέχουσα τιμή (ως βάση): €{float(last_price):.2f}")
    print(f"Μέση τιμή προσομοίωσης σε 4 έτη: €{float(mean_price):.2f}")
    print(f"Το 90% των σεναρίων έδειξε τιμή μεταξύ €{float(percentile_5):.2f} και €{float(percentile_95):.2f}.")

except Exception:
    import traceback
    print("\n--- AN ERROR OCCURRED ---")
    print("Full technical traceback below:")
    print("---------------------------------")
    print(traceback.format_exc())
    print("---------------------------------")