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
ticker_stock = 'DTE.DE'
ticker_index = '^GDAXI' # Ο δείκτης DAX
start_date = '2020-06-19'
end_date = pd.to_datetime('today')
# Ετήσιο επιτόκιο χωρίς κίνδυνο (π.χ. απόδοση γερμανικού ομολόγου), υποθέτουμε 2.5%
RISK_FREE_RATE = 0.025 

# Αγνοούμε τις προειδοποιήσεις για καθαρότερο αποτέλεσμα
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=PerformanceWarning)
warnings.filterwarnings('ignore', message="y is poorly scaled*")

print(f"Ανάκτηση δεδομένων για {ticker_stock} και {ticker_index} από {start_date} έως σήμερα...")

try:
    # Κατεβάζουμε ταυτόχρονα δεδομένα για μετοχή και δείκτη
    data = yf.download([ticker_stock, ticker_index], start=start_date, end=end_date)['Close']
    if data.empty:
        raise ValueError(f"Δεν βρέθηκαν δεδομένα.")

    # --- 1. Γράφημα Ιστορικής Πορείας ---
    plt.style.use('ggplot')
    fig_hist, ax_hist = plt.subplots(figsize=(14, 7))
    ax_hist.plot(data[ticker_stock], label=f'Τιμή Κλεισίματος {ticker_stock}', color='darkblue')
    ax_hist.set_title(f'Ιστορική Πορεία Μετοχής Deutsche Telekom ({ticker_stock})', fontsize=16)
    ax_hist.set_ylabel('Τιμή σε EUR (€)', fontsize=12)
    ax_hist.set_xlabel('Ημερομηνία', fontsize=12)
    ax_hist.legend()
    ax_hist.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'€{y:.2f}'))
    hist_filename = 'historical_chart.png'
    plt.savefig(hist_filename)
    plt.close(fig_hist)
    print(f"\nΤο γράφημα της ιστορικής πορείας αποθηκεύτηκε ως '{hist_filename}'")

    # --- 2. GARCH Model (Βραχυπρόθεσμο) ---
    returns_stock = data[ticker_stock].pct_change().dropna()
    garch_model = arch_model(returns_stock * 100, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp='off')
    forecast_30d = garch_fit.forecast(horizon=30)
    vol_30d = np.sqrt(forecast_30d.variance.iloc[-1]) / 100
    annualized_vol_30d = vol_30d * np.sqrt(252) * 100
    print("\n--- Μοντέλο GARCH (Πρόβλεψη Κινδύνου/Μεταβλητότητας) ---")
    print(f"Η προβλεπόμενη ετησιοποιημένη μεταβλητότητα για τις επόμενες 30 ημέρες είναι: {float(annualized_vol_30d.iloc[-1]):.2f}%")

    # --- 3. Προσομοίωση Monte Carlo για 4 Έτη ---
    last_price = data[ticker_stock].iloc[-1].item()
    num_simulations = 10000
    num_days = 252 * 4
    mu = returns_stock.mean()
    sigma = returns_stock.std()
    
    print(f"\nΕκτέλεση βελτιστοποιημένης προσομοίωσης Monte Carlo για 4 έτη με {num_simulations} σενάρια...")
    random_returns = np.random.normal(mu, sigma, (num_days, num_simulations))
    price_paths = np.zeros((num_days + 1, num_simulations))
    price_paths[0] = last_price
    for t in range(1, num_days + 1):
        price_paths[t] = price_paths[t-1] * (1 + random_returns[t-1])
    simulation_df = pd.DataFrame(price_paths[1:])

    fig_mc, ax_mc = plt.subplots(figsize=(14, 8))
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

    end_prices = simulation_df.iloc[-1]
    percentile_5 = end_prices.quantile(0.05)
    percentile_95 = end_prices.quantile(0.95)
    mean_price = end_prices.mean()
    print("\n--- Προσομοίωση Monte Carlo (Εύρος Πιθανών Τιμών σε 4 Έτη) ---")
    print(f"Τρέχουσα τιμή (ως βάση): €{float(last_price):.2f}")
    print(f"Μέση τιμή προσομοίωσης σε 4 έτη: €{float(mean_price):.2f}")
    print(f"Το 90% των σεναρίων έδειξε τιμή μεταξύ €{float(percentile_5):.2f} και €{float(percentile_95):.2f}.")

    # --- ΝΕΟ: Υπολογισμός Δείκτη Sharpe ---
    def calculate_sharpe_ratio(returns, risk_free_rate):
        # Υπολογίζουμε την περίσσεια απόδοσης (excess return)
        excess_returns = returns.mean() * 252 - risk_free_rate
        # Υπολογίζουμε την ετησιοποιημένη μεταβλητότητα (τον κίνδυνο)
        volatility = returns.std() * np.sqrt(252)
        # Υπολογίζουμε τον δείκτη Sharpe
        sharpe_ratio = excess_returns / volatility
        return sharpe_ratio

    returns_index = data[ticker_index].pct_change().dropna()

    sharpe_stock = calculate_sharpe_ratio(returns_stock, RISK_FREE_RATE)
    sharpe_index = calculate_sharpe_ratio(returns_index, RISK_FREE_RATE)

    print("\n--- Μοντέλο Αξιολόγησης Κινδύνου/Απόδοσης (Δείκτης Sharpe) ---")
    print(f"Δείκτης Sharpe για την μετοχή {ticker_stock}: {sharpe_stock:.2f}")
    print(f"Δείκτης Sharpe για τον δείκτη {ticker_index}: {sharpe_index:.2f}")
    if sharpe_stock > sharpe_index:
        print(f"\nΣυμπέρασμα: Η μετοχή {ticker_stock} είχε καλύτερη απόδοση προσαρμοσμένη στον κίνδυνο από τη συνολική αγορά κατά την εξεταζόμενη περίοδο.")
    else:
        print(f"\nΣυμπέρασμα: Η συνολική αγορά ({ticker_index}) είχε καλύτερη απόδοση προσαρμοσμένη στον κίνδυνο από τη μετοχή {ticker_stock} κατά την εξεταζόμενη περίοδο.")


except Exception:
    import traceback
    print("\n--- AN ERROR OCCURRED ---")
    print("Full technical traceback below:")
    print("---------------------------------")
    print(traceback.format_exc())
    print("---------------------------------")