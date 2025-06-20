import streamlit as st
import yfinance as yf
import pandas as pd
from pandas.errors import PerformanceWarning
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from matplotlib.ticker import FuncFormatter
import warnings
import traceback
from datetime import datetime, timedelta
import math

# --- Βασικές Ρυθμίσεις Σελίδας & Προειδοποιήσεων ---
st.set_page_config(layout="wide", page_title="Financial Analysis Toolkit")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=PerformanceWarning)
warnings.filterwarnings('ignore', message="y is poorly scaled*")

if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = ''

# --- Συνάρτηση για Caching Δεδομένων ---
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty: return None
        return data
    except Exception:
        return None

# --- UI Στοιχεία στην Πλαϊνή Μπάρα ---
st.sidebar.header('Επιλογές Ανάλυσης')
ticker = st.sidebar.text_input('Σύμβολο Μετοχής (Ticker)', 'DTE.DE').upper()
analysis_type = st.sidebar.selectbox(
    'Επιλέξτε Ανάλυση',
    [
        'Υπολογιστής Επένδυσης (4-Έτη)',
        'Επισκόπηση & Βραχυπρόθεσμος Κίνδυνος (GARCH)',
        'Ανάλυση Θεμελιωδών Δεδομένων',
        'Προσομοίωση Monte Carlo (4-Έτη)', 
        'Αξιολόγηση Κινδύνου/Απόδοσης (Δείκτης Sharpe)',
        'Ιστορικός Έλεγχος (Backtesting 2021-2025)'
    ]
)
if analysis_type == 'Ανάλυση Θεμελιωδών Δεδομένων':
    years_to_show = st.sidebar.slider('Ιστορικός Ορίζοντας (Έτη)', min_value=3, max_value=15, value=10, key='fundamental_years')

# Το κεντρικό κουμπί ελέγχει μόνο τις "γενικές" αναλύσεις
run_general_analysis_button = st.sidebar.button('Εκτέλεση Γενικής Ανάλυσης')

st.title(f'Financial Analysis Toolkit: {ticker}')

# --- Κεντρική Λογική της Εφαρμογής ---

# --- Απομονωμένη Λογική για τον Υπολογιστή Επένδυσης ---
if analysis_type == 'Υπολογιστής Επένδυσης (4-Έτη)':
    st.header(f'Υπολογιστής Προβολής Επένδυσης για {ticker}')
    st.info("Ορίστε τις παραμέτρους της επένδυσής σας παρακάτω και πατήστε 'Υπολόγισε' για να δείτε την προβολή.")

    st.subheader("1. Ορίστε το Σενάριο Επένδυσής σας")
    col1, col2 = st.columns(2)
    with col1:
        investment_amount = st.number_input("Αρχικό Ποσό Επένδυσης (€)", min_value=50.0, value=1000.0, step=50.0)
    with col2:
        offer_enabled = st.checkbox("Ενεργοποίηση προσφοράς 'Αγορά 2, πάρε 1 Δώρο'", value=True)
    
    st.subheader("2. Ορίστε τις Παραδοχές για τα Μερίσματα")
    col3, col4 = st.columns(2)
    with col3:
        dividend_per_share = st.number_input("Αναμενόμενο Ετήσιο Μέρισμα ανά Μετοχή (€)", value=0.90, step=0.01, format="%.2f")
    with col4:
        dividend_growth = st.number_input("Ετήσιος Ρυθμός Αύξησης Μερίσματος (%)", value=3.0, step=0.5, format="%.1f")
    
    calculate_investment_button = st.button("Υπολόγισε την Προβολή της Επένδυσης")

    if calculate_investment_button:
        with st.spinner('Εκτέλεση προσομοίωσης και υπολογισμών...'):
            start_date_full = (pd.to_datetime('today') - timedelta(days=365*16)).strftime('%Y-%m-%d')
            end_date_full = pd.to_datetime('today')
            data = load_data(ticker, start_date_full, end_date_full)
            
            if data is not None:
                last_price = data['Close'].iloc[-1].item()
                
                shares_bought = investment_amount / last_price
                bonus_shares = math.floor(shares_bought / 2) if offer_enabled else 0
                total_shares = shares_bought + bonus_shares
                effective_cost_per_share = investment_amount / total_shares if total_shares > 0 else 0

                total_dividends = 0
                current_dividend = dividend_per_share
                growth_rate = dividend_growth / 100
                for _ in range(4):
                    total_dividends += total_shares * current_dividend
                    current_dividend *= (1 + growth_rate)
                
                returns_mc = data['Close'].pct_change().dropna()
                mu, sigma = returns_mc.mean(), returns_mc.std()
                num_simulations, num_days = 10000, 252 * 4
                random_returns = np.random.normal(mu, sigma, (num_days, num_simulations))
                price_paths = np.zeros((num_days + 1, num_simulations))
                price_paths[0] = last_price
                for t in range(1, num_days + 1):
                    price_paths[t] = price_paths[t-1] * (1 + random_returns[t-1])
                
                end_prices = pd.Series(price_paths[-1])
                p5, p95, mean_price = end_prices.quantile(0.05), end_prices.quantile(0.95), end_prices.mean()

                st.divider()
                st.subheader("Ανάλυση Αρχικής Επένδυσης")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Επένδυση", f"€{investment_amount:.2f}")
                c2.metric("Τρέχουσα Τιμή Μετοχής", f"€{last_price:.2f}")
                c3.metric("Σύνολο Μετοχών (με δώρο)", f"{total_shares:.2f}")
                c4.metric("Πραγματικό Κόστος / Μετοχή", f"€{effective_cost_per_share:.2f}", delta=f"{-((1 - (effective_cost_per_share/last_price))*100):.1f}% έκπτωση" if offer_enabled and last_price > 0 else None, delta_color="inverse")

                st.divider()
                st.subheader("Προβολή Συνολικής Αξίας σε 4 Έτη")
                st.info("Οι παρακάτω τιμές είναι το άθροισμα της προβλεπόμενης αξίας των μετοχών (βάσει Monte Carlo) και του συνόλου των εκτιμώμενων μερισμάτων (βάσει των παραδοχών σας).")
                
                col1, col2, col3 = st.columns(3, gap="large")
                with col1:
                    st.markdown("<h4 style='text-align: center; color: #E84C3D;'>Απαισιόδοξο Σενάριο (5%)</h4>", unsafe_allow_html=True)
                    st.metric("Συνολική Αξία", f"€{(p5 * total_shares) + total_dividends:.2f}")
                    st.write(f"Αξία Μετοχών: €{p5 * total_shares:.2f}")
                    st.write(f"Σύνολο Μερισμάτων: €{total_dividends:.2f}")
                with col2:
                    st.markdown("<h4 style='text-align: center; color: orange;'>Μέσο Σενάριο</h4>", unsafe_allow_html=True)
                    st.metric("Συνολική Αξία", f"€{(mean_price * total_shares) + total_dividends:.2f}")
                    st.write(f"Αξία Μετοχών: €{mean_price * total_shares:.2f}")
                    st.write(f"Σύνολο Μερισμάτων: €{total_dividends:.2f}")
                with col3:
                    st.markdown("<h4 style='text-align: center; color: #2ECC70;'>Αισιόδοξο Σενάριο (95%)</h4>", unsafe_allow_html=True)
                    st.metric("Συνολική Αξία", f"€{(p95 * total_shares) + total_dividends:.2f}")
                    st.write(f"Αξία Μετοχών: €{p95 * total_shares:.2f}")
                    st.write(f"Σύνολο Μερισμάτων: €{total_dividends:.2f}")
            else:
                st.error(f"Δεν ήταν δυνατή η ανάκτηση δεδομένων για το σύμβολο '{ticker}'.")

# --- Λογική για όλες τις υπόλοιπες αναλύσεις ---
else:
    st.info(f"Επιλέχθηκε η ανάλυση: '{analysis_type}'. Πατήστε το κουμπί 'Εκτέλεση Γενικής Ανάλυσης' στην πλαϊνή μπάρα για να ξεκινήσει.")
    
    if run_general_analysis_button:
        start_date_full = (pd.to_datetime('today') - timedelta(days=365*16)).strftime('%Y-%m-%d')
        end_date_full = pd.to_datetime('today')
        
        # --- Ανάλυση Θεμελιωδών Δεδομένων ---
        if analysis_type == 'Ανάλυση Θεμελιωδών Δεδομένων':
            st.header(f'Θεμελιώδη Στοιχεία για {ticker}')
            with st.spinner('Άντληση live δεικτών και ιστορικών δεδομένων...'):
                try:
                    tkr = yf.Ticker(ticker)
                    info = tkr.info
                    st.subheader('Βασικοί Δείκτες Αποτίμησης & Υγείας (Live)')
                    def get_info_metric(info_dict, key, format_str='{:.2f}'):
                        val = info_dict.get(key);
                        if val is None or val == 0: return "N/A"
                        if isinstance(val, (int, float)):
                            if format_str == 'human':
                                if abs(val) > 1_000_000_000_000: return f"€{val/1_000_000_000_000:.2f}T"
                                if abs(val) > 1_000_000_000: return f"€{val/1_000_000_000:.2f}B"
                                if abs(val) > 1_000_000: return f"€{val/1_000_000:.2f}M"
                                return f"€{val}"
                            if format_str.endswith('%'): return f"{val*100:.2f}%"
                            return format_str.format(val)
                        return val
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Κεφαλαιοποίηση", get_info_metric(info, 'marketCap', 'human'))
                    col2.metric("P/E (Trailing)", get_info_metric(info, 'trailingPE'))
                    col3.metric("Μερισματική Απόδοση", get_info_metric(info, 'dividendYield', '%'))
                    col4.metric("Δείκτης Χρέους/Ιδ.Κεφ.", get_info_metric(info, 'debtToEquity'))

                    st.subheader("Ιστορική Πορεία Βασικών Μεγεθών (2010-2024)")
                    st.info("Το παρακάτω γράφημα χρησιμοποιεί τον σταθερό πίνακα δεδομένων για την DTE.DE, συνδυασμένο με την ιστορική τιμή της μετοχής.")
                    user_data = {
                        'Έτος': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
                        'Έσοδα (€ δις)': [58.6, 58.1, 60.1, 62.6, 69.2, 73.0, 74.9, 75.6, 80.5, 100.0, 108.0, 114.0, 111.0, 115.8, 112.0],
                        'Καθαρά Κέρδη (€ δις)': [0.55, -5.2, 0.93, 2.9, 3.2, 2.6, 3.4, 2.1, 3.8, 4.1, 4.1, 8.0, 17.7, 20.4, 9.0],
                        'Μέρισμα (€ ανά μετοχή)': [0.70, 0.70, 0.70, 0.50, 0.50, 0.55, 0.60, 0.65, 0.70, 0.60, 0.60, 0.64, 0.70, 0.77, 0.90]
                    }
                    user_df = pd.DataFrame(user_data).set_index('Έτος')
                    price_data = load_data(ticker, '2010-01-01', end_date_full)
                    annual_price = price_data['Close'].resample('A').mean().squeeze()
                    annual_price.index = annual_price.index.year
                    user_df['Price'] = user_df.index.map(annual_price)
                    df_plot = user_df.dropna()
                    
                    fig, ax1 = plt.subplots(figsize=(14, 8))
                    plt.style.use('ggplot')
                    ax1.set_title(f'Σύγκριση Οικονομικών Στοιχείων & Τιμής Μετοχής για {ticker}', fontsize=16)
                    ax1.set_xlabel('Έτος', fontsize=12)
                    ax1.set_ylabel('Ποσό σε Δισ. EUR (€)', fontsize=12)
                    ax1.bar(df_plot.index - 0.2, df_plot['Έσοδα (€ δις)'], width=0.4, label='Κύκλος Εργασιών', align='center', color='royalblue')
                    ax1.bar(df_plot.index + 0.2, df_plot['Καθαρά Κέρδη (€ δις)'], width=0.4, label='Καθαρά Κέρδη', align='center', color='cyan')
                    ax2 = ax1.twinx()
                    ax3 = ax1.twinx()
                    ax3.spines['right'].set_position(('outward', 60))
                    ax2.set_ylabel('Μέση Ετήσια Τιμή Μετοχής (€)', fontsize=12, color='darkred')
                    ax2.plot(df_plot.index, df_plot['Price'].values, color='darkred', marker='o', linestyle='--', label='Μέση Τιμή Μετοχής')
                    ax2.tick_params(axis='y', labelcolor='darkred')
                    ax3.set_ylabel('Μέρισμα ανά Μετοχή (€)', fontsize=12, color='green')
                    ax3.plot(df_plot.index, df_plot['Μέρισμα (€ ανά μετοχή)'], color='green', marker='D', linestyle=':', label='Μέρισμα ανά Μετοχή')
                    ax3.tick_params(axis='y', labelcolor='green')
                    lines, labels = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    lines3, labels3 = ax3.get_legend_handles_labels()
                    ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')
                    fig.tight_layout()
                    st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Παρουσιάστηκε ένα σφάλμα: {e}")

        # --- GARCH ---
        elif analysis_type == 'Επισκόπηση & Βραχυπρόθεσμος Κίνδυνος (GARCH)':
            data = load_data(ticker, start_date_full, end_date_full)
            st.header(f'Ιστορική Πορεία & Πρόβλεψη Κινδύνου για {ticker}')
            fig, ax = plt.subplots(figsize=(14, 7))
            plt.style.use('ggplot')
            ax.plot(data['Close'], label=f'Τιμή Κλεισίματος {ticker}', color='darkblue')
            st.pyplot(fig)
            with st.spinner('Υπολογισμός GARCH...'):
                returns = data['Close'].pct_change().dropna() * 100
                garch_model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
                garch_fit = garch_model.fit(disp='off')
                forecast = garch_fit.forecast(horizon=30)
                predicted_volatility = np.sqrt(forecast.variance.iloc[-1]) / 100
                annualized_vol = predicted_volatility * np.sqrt(252) * 100
                st.subheader('Πρόβλεψη Κινδύνου (Επόμενες 30 Ημέρες)')
                st.metric(label="Προβλεπόμενη Ετησιοποιημένη Μεταβλητότητα", value=f"{float(annualized_vol.iloc[-1]):.2f}%")
        
        # --- MONTE CARLO ---
        elif analysis_type == 'Προσομοίωση Monte Carlo (4-Έτη)':
            data = load_data(ticker, start_date_full, end_date_full)
            st.header(f'Προσομοίωση Monte Carlo για {ticker} (4-Έτη)')
            with st.spinner('Εκτέλεση 10,000 σεναρίων Monte Carlo...'):
                returns_mc = data['Close'].pct_change().dropna()
                last_price = data['Close'].iloc[-1].item()
                num_simulations, num_days = 10000, 252 * 4
                mu, sigma = returns_mc.mean(), returns_mc.std()
                random_returns = np.random.normal(mu, sigma, (num_days, num_simulations))
                price_paths = np.zeros((num_days + 1, num_simulations))
                price_paths[0] = last_price
                for t in range(1, num_days + 1):
                    price_paths[t] = price_paths[t-1] * (1 + random_returns[t-1])
                simulation_df = pd.DataFrame(price_paths[1:])
                fig_mc, ax_mc = plt.subplots(figsize=(14, 8))
                plt.style.use('ggplot')
                ax_mc.plot(simulation_df, color='lightblue', alpha=0.01)
                sample_indices = np.random.choice(simulation_df.columns, 50, replace=False)
                ax_mc.plot(simulation_df[sample_indices], alpha=0.5)
                ax_mc.set_title(f'Προσομοίωση Monte Carlo ({num_simulations} σενάρια για 4 Χρόνια)', fontsize=16)
                ax_mc.set_ylabel('Πιθανή Τιμή Μετοχής', fontsize=12)
                ax_mc.axhline(y=last_price, color='r', linestyle='--', label=f'Τρέχουσα Τιμή (€{last_price:.2f})')
                ax_mc.legend([ax_mc.get_legend_handles_labels()[0][-1]], [ax_mc.get_legend_handles_labels()[1][-1]])
                st.pyplot(fig_mc)
                end_prices = simulation_df.iloc[-1]
                p5, p95, mean_price = end_prices.quantile(0.05), end_prices.quantile(0.95), end_prices.mean()
                st.subheader(f'Αποτελέσματα Προσομοίωσης για το {end_date_full.year + 4}')
                col1, col2, col3 = st.columns(3)
                col1.metric("Απαισιόδοξο Σεν. (5%)", f"€{p5:.2f}")
                col2.metric("Μέση Τιμή Σεναρίων", f"€{mean_price:.2f}")
                col3.metric("Αισιόδοξο Σεν. (95%)", f"€{p95:.2f}")

        # --- SHARPE RATIO ---
        elif analysis_type == 'Αξιολόγηση Κινδύνου/Απόδοσης (Δείκτης Sharpe)':
            st.header(f'Σύγκριση Δείκτη Sharpe: {ticker} vs. DAX Index')
            index_ticker = '^GDAXI'
            with st.spinner(f'Ανάκτηση δεδομένων για τον δείκτη {index_ticker}...'):
                all_tickers = [ticker, index_ticker]
                data_sharpe = load_data(all_tickers, '2020-01-01', end_date_full)
            if data_sharpe is None or ticker not in data_sharpe['Close'].columns or index_ticker not in data_sharpe['Close'].columns:
                st.error("Δεν ήταν δυνατή η ανάκτηση των δεδομένων για τη σύγκριση.")
            else:
                data_sharpe = data_sharpe['Close']
                RISK_FREE_RATE = 0.025
                def calculate_sharpe(returns):
                    excess_returns = returns.mean() * 252 - RISK_FREE_RATE
                    volatility = returns.std() * np.sqrt(252)
                    return excess_returns / volatility if volatility != 0 else 0
                sharpe_stock = calculate_sharpe(data_sharpe[ticker].pct_change().dropna())
                sharpe_index = calculate_sharpe(data_sharpe[index_ticker].pct_change().dropna())
                col1, col2 = st.columns(2)
                col1.metric(f"Δείκτης Sharpe για {ticker}", f"{sharpe_stock:.2f}")
                col2.metric(f"Δείκτης Sharpe για DAX", f"{sharpe_index:.2f}")
                if sharpe_stock > sharpe_index:
                    st.success(f"Συμπέρασμα: Η μετοχή {ticker} είχε καλύτερη απόδοση προσαρμοσμένη στον κίνδυνο.")
                else:
                    st.warning(f"Συμπέρασμα: Η συνολική αγορά (DAX) είχε καλύτερη απόδοση προσαρμοσμένη στον κίνδυνο.")

        # --- BACKTESTING ---
        elif analysis_type == 'Ιστορικός Έλεγχος (Backtesting 2021-2025)':
            data = load_data(ticker, start_date_full, end_date_full)
            st.header(f'Backtesting Προσομοίωσης για {ticker} (2021-2025)')
            backtest_start_date = '2021-01-04'
            past_data = data.loc[:backtest_start_date]
            if past_data.empty or len(past_data) < 252:
                st.error("Δεν υπάρχουν αρκετά ιστορικά δεδομένα για το backtest.")
            else:
                backtest_start_price = past_data['Close'].iloc[-1].item()
                actual_price_today = data['Close'].iloc[-1].item()
                st.write(f"Εκτελούμε προσομοίωση με βάση τα δεδομένα που ήταν γνωστά την {backtest_start_date} (τιμή εκκίνησης: €{backtest_start_price:.2f})")
                with st.spinner('Εκτέλεση backtesting...'):
                    returns_past = past_data['Close'].pct_change().dropna()
                    mu, sigma = returns_past.mean(), returns_past.std()
                    num_days = (end_date_full - pd.to_datetime(backtest_start_date)).days
                    random_returns = np.random.normal(mu, sigma, (int(num_days), 10000))
                    price_paths = np.zeros((int(num_days) + 1, 10000))
                    price_paths[0] = backtest_start_price
                    for t in range(1, int(num_days) + 1):
                        price_paths[t] = price_paths[t-1] * (1 + random_returns[t-1])
                    backtest_end_prices = pd.Series(price_paths[-1])
                    p5, p95 = backtest_end_prices.quantile(0.05), backtest_end_prices.quantile(0.95)
                st.subheader("Αποτελέσματα Backtest")
                st.write(f"Η πρόβλεψη του 2021 για σήμερα ήταν ένα εύρος από **€{p5:.2f}** έως **€{p95:.2f}**.")
                st.write(f"Η πραγματική τιμή σήμερα είναι **€{actual_price_today:.2f}**.")
                if p5 <= actual_price_today <= p95:
                    st.success("--> ΑΞΙΟΛΟΓΗΣΗ: ΕΠΙΤΥΧΗΣ! Η πραγματική τιμή έπεσε μέσα στο 90% εύρος πιθανοτήτων του μοντέλου.")
                else:
                    st.error("--> ΑΞΙΟΛΟΓΗΣΗ: ΑΠΟΤΥΧΙΑ. Η πραγματική τιμή έπεσε έξω από το 90% εύρος πιθανοτήτων του μοντέλου.")