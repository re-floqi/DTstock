import yfinance as yf
import pandas as pd
from pandas.errors import PerformanceWarning
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings
import traceback

# --- Ρυθμίσεις ---
ticker = 'DTE.DE'
# Κατεβάζουμε δεδομένα από παλιά για να έχουμε ιστορικό για το backtest
full_data_start_date = '2015-01-01' 
# Ημερομηνία έναρξης του backtest
backtest_start_date = '2021-01-04' # Πρώτη εργάσιμη του 2021
# Η σημερινή ημερομηνία
end_date = pd.to_datetime('today')

num_simulations = 10000
num_years = 4
num_days = 252 * num_years

# Αγνοούμε τις προειδοποιήσεις για καθαρότερο αποτέλεσμα
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=PerformanceWarning)

def run_monte_carlo(historical_data, start_price, days_to_sim, num_sims, plot_title, filename):
    """Συνάρτηση για την εκτέλεση και σχεδίαση της προσομοίωσης Monte Carlo."""
    print(f"\nΕκτέλεση προσομοίωσης Monte Carlo για {days_to_sim} ημέρες με {num_sims} σενάρια...")
    
    returns = historical_data['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()

    random_returns = np.random.normal(mu, sigma, (days_to_sim, num_sims))
    price_paths = np.zeros((days_to_sim + 1, num_sims))
    price_paths[0] = start_price
    
    for t in range(1, days_to_sim + 1):
        price_paths[t] = price_paths[t-1] * (1 + random_returns[t-1])
        
    simulation_df = pd.DataFrame(price_paths[1:])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(simulation_df, color='blue', alpha=0.01)
    ax.set_title(plot_title, fontsize=16)
    ax.set_ylabel('Πιθανή Τιμή Μετοχής (€)', fontsize=12)
    ax.set_xlabel('Μελλοντικές Ημέρες Συναλλαγών', fontsize=12)
    ax.axhline(y=start_price, color='r', linestyle='--', label=f'Τιμή Εκκίνησης (€{float(start_price):.2f})')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0]], [labels[0]])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'€{y:.2f}'))
    
    plt.savefig(filename)
    plt.close(fig)
    print(f"Το γράφημα της προσομοίωσης αποθηκεύτηκε ως '{filename}'")
    
    return simulation_df.iloc[-1]

try:
    print(f"Ανάκτηση δεδομένων για {ticker} από {full_data_start_date} έως σήμερα...")
    full_data = yf.download(ticker, start=full_data_start_date, end=end_date)
    if full_data.empty:
        raise ValueError("Δεν βρέθηκαν δεδομένα.")

    # --- ΦΑΣΗ 1: BACKTEST (2021 -> 2025) ---
    print("\n" + "="*40)
    print(" " * 8 + "ΦΑΣΗ 1: BACKTEST 2021-2025")
    print("="*40)
    
    # 1. Απομονώνουμε τα δεδομένα που ήταν γνωστά μέχρι την αρχή του 2021
    past_data = full_data.loc[:backtest_start_date]
    backtest_start_price = past_data['Close'].iloc[-1].item()
    print(f"Τιμή εκκίνησης για το backtest (την {backtest_start_date}): €{backtest_start_price:.2f}")

    # 2. Τρέχουμε την προσομοίωση με τα παλιά δεδομένα
    backtest_end_prices = run_monte_carlo(
        historical_data=past_data, 
        start_price=backtest_start_price, 
        days_to_sim=num_days, 
        num_sims=num_simulations,
        plot_title=f'Backtest Προσομοίωσης Monte Carlo ({num_simulations} σενάρια από 2021 έως 2025)',
        filename='backtest_2021_to_2025.png'
    )
    
    # 3. Βρίσκουμε τα αποτελέσματα της παλιάς πρόβλεψης
    backtest_p5 = backtest_end_prices.quantile(0.05)
    backtest_p95 = backtest_end_prices.quantile(0.95)
    backtest_mean = backtest_end_prices.mean()
    
    # 4. Βρίσκουμε την πραγματική τιμή σήμερα και συγκρίνουμε
    actual_price_today = full_data['Close'].iloc[-1].item()
    
    print("\n--- Αποτελέσματα Backtest ---")
    print(f"Η πρόβλεψη του 2021 για το 2025 ήταν ένα εύρος από €{backtest_p5:.2f} έως €{backtest_p95:.2f} (μέση τιμή: €{backtest_mean:.2f}).")
    print(f"Η ΠΡΑΓΜΑΤΙΚΗ τιμή σήμερα (Ιούνιος 2025) είναι: €{actual_price_today:.2f}")
    
    if backtest_p5 <= actual_price_today <= backtest_p95:
        print("\n--> ΑΞΙΟΛΟΓΗΣΗ: ΕΠΙΤΥΧΗΣ! Η πραγματική τιμή έπεσε μέσα στο 90% εύρος πιθανοτήτων του μοντέλου.")
    else:
        print("\n--> ΑΞΙΟΛΟΓΗΣΗ: ΑΠΟΤΥΧΙΑ. Η πραγματική τιμή έπεσε έξω από το 90% εύρος πιθανοτήτων του μοντέλου.")


    # --- ΦΑΣΗ 2: ΝΕΑ ΠΡΟΒΛΕΨΗ (2025 -> 2029) ---
    print("\n" + "="*40)
    print(" " * 5 + "ΦΑΣΗ 2: ΝΕΑ ΠΡΟΒΛΕΨΗ 2025-2029")
    print("="*40)

    # 1. Χρησιμοποιούμε ΟΛΑ τα δεδομένα μέχρι σήμερα
    current_start_price = actual_price_today
    print(f"Τιμή εκκίνησης για τη νέα πρόβλεψη (σήμερα): €{current_start_price:.2f}")

    # 2. Τρέχουμε τη νέα προσομοίωση
    forecast_end_prices = run_monte_carlo(
        historical_data=full_data, # Χρησιμοποιούμε όλα τα δεδομένα για να "εκπαιδεύσουμε" το μοντέλο
        start_price=current_start_price, 
        days_to_sim=num_days, 
        num_sims=num_simulations,
        plot_title=f'Νέα Προσομοίωση Monte Carlo ({num_simulations} σενάρια από 2025 έως 2029)',
        filename='forecast_2025_to_2029.png'
    )
    
    # 3. Βρίσκουμε τα αποτελέσματα της νέας πρόβλεψης
    forecast_p5 = forecast_end_prices.quantile(0.05)
    forecast_p95 = forecast_end_prices.quantile(0.95)
    forecast_mean = forecast_end_prices.mean()
    
    print("\n--- Αποτελέσματα Νέας Πρόβλεψης για το 2029 ---")
    print(f"Η πρόβλεψη για το 2029 είναι ένα εύρος από €{forecast_p5:.2f} έως €{forecast_p95:.2f} (μέση τιμή: €{forecast_mean:.2f}).")
    print("Αυτό αντιπροσωπεύει το στατιστικά πιθανό εύρος τιμών σε 4 χρόνια από σήμερα, με βάση το σύνολο της ιστορικής συμπεριφοράς της μετοχής.")

except Exception:
    import traceback
    print("\n--- AN ERROR OCCURRED ---")
    print("Full technical traceback below:")
    print("---------------------------------")
    print(traceback.format_exc())
    print("---------------------------------")