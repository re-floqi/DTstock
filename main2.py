import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
 
# Alternative method if Prophet is not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not available. Using alternative forecasting methods.")
    PROPHET_AVAILABLE = False
 
print("=== DTE.DE (Deutsche Telekom AG) Stock Analysis ===\n")
 
# Get the last 5 years of data
ticker_symbol = 'DTE.DE'
try:
    ticker_data = yf.Ticker(ticker_symbol)
    ticker_df_5y = ticker_data.history(period='5y')
   
    if ticker_df_5y.empty:
        print("No data found for DTE.DE. Trying alternative ticker...")
        # Try alternative ticker format
        ticker_symbol = 'DTE.F'  # Frankfurt exchange
        ticker_data = yf.Ticker(ticker_symbol)
        ticker_df_5y = ticker_data.history(period='5y')
   
    if ticker_df_5y.empty:
        raise Exception("No data available for Deutsche Telekom")
       
except Exception as e:
    print(f"Error fetching data: {e}")
    print("This might be due to network issues or ticker symbol changes.")
    exit()
 
print(f"Successfully fetched {len(ticker_df_5y)} days of data for {ticker_symbol}")
print(f"Date range: {ticker_df_5y.index[0].date()} to {ticker_df_5y.index[-1].date()}")
 
# Display basic statistics
print(f"\n=== 5-Year Price Summary ===")
print(f"Current Price: €{ticker_df_5y['Close'][-1]:.2f}")
print(f"5-Year High: €{ticker_df_5y['Close'].max():.2f}")
print(f"5-Year Low: €{ticker_df_5y['Close'].min():.2f}")
print(f"Average Price: €{ticker_df_5y['Close'].mean():.2f}")
print(f"Price Change: {((ticker_df_5y['Close'][-1] / ticker_df_5y['Close'][0] - 1) * 100):.1f}%")
 
# Display recent closing prices
print(f"\n=== Recent Closing Prices ===")
print(ticker_df_5y['Close'].tail(10))
 
# Method 1: Linear Regression Prediction
print(f"\n=== Method 1: Linear Regression Forecast ===")
 
# Prepare data for linear regression
dates_numeric = np.arange(len(ticker_df_5y)).reshape(-1, 1)
prices = ticker_df_5y['Close'].values
 
# Fit linear regression
lr_model = LinearRegression()
lr_model.fit(dates_numeric, prices)
 
# Predict next 4 years (approximately 1460 days, accounting for weekends/holidays ~1040 trading days)
future_days = 1040
future_dates_numeric = np.arange(len(ticker_df_5y), len(ticker_df_5y) + future_days).reshape(-1, 1)
lr_predictions = lr_model.predict(future_dates_numeric)
 
# Calculate R² score
lr_r2 = r2_score(prices, lr_model.predict(dates_numeric))
print(f"Linear Regression R² Score: {lr_r2:.3f}")
 
# Generate future dates
last_date = ticker_df_5y.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')
 
print(f"Linear Trend Prediction for next 4 years:")
print(f"Year 1 End: €{lr_predictions[260]:.2f}")  # ~1 year of trading days
print(f"Year 2 End: €{lr_predictions[520]:.2f}")  # ~2 years
print(f"Year 3 End: €{lr_predictions[780]:.2f}")  # ~3 years
print(f"Year 4 End: €{lr_predictions[-1]:.2f}")   # 4 years
 
# Method 2: Polynomial Regression (2nd degree)
print(f"\n=== Method 2: Polynomial Regression Forecast ===")
 
poly_features = PolynomialFeatures(degree=2)
dates_poly = poly_features.fit_transform(dates_numeric)
future_dates_poly = poly_features.transform(future_dates_numeric)
 
poly_model = LinearRegression()
poly_model.fit(dates_poly, prices)
poly_predictions = poly_model.predict(future_dates_poly)
 
poly_r2 = r2_score(prices, poly_model.predict(dates_poly))
print(f"Polynomial Regression R² Score: {poly_r2:.3f}")
 
print(f"Polynomial Trend Prediction for next 4 years:")
print(f"Year 1 End: €{poly_predictions[260]:.2f}")
print(f"Year 2 End: €{poly_predictions[520]:.2f}")
print(f"Year 3 End: €{poly_predictions[780]:.2f}")
print(f"Year 4 End: €{poly_predictions[-1]:.2f}")
 
# Method 3: Moving Average Trend
print(f"\n=== Method 3: Moving Average Trend Forecast ===")
 
# Calculate different moving averages
ma_50 = ticker_df_5y['Close'].rolling(window=50).mean()
ma_200 = ticker_df_5y['Close'].rolling(window=200).mean()
 
# Simple trend projection based on recent slope
recent_period = 100  # Last 100 days
recent_slope = (ticker_df_5y['Close'][-1] - ticker_df_5y['Close'][-recent_period]) / recent_period
current_price = ticker_df_5y['Close'][-1]
 
ma_predictions = []
for days in range(1, future_days + 1):
    predicted_price = current_price + (recent_slope * days)
    ma_predictions.append(predicted_price)
 
ma_predictions = np.array(ma_predictions)
 
print(f"Moving Average Trend Prediction for next 4 years:")
print(f"Year 1 End: €{ma_predictions[260]:.2f}")
print(f"Year 2 End: €{ma_predictions[520]:.2f}")
print(f"Year 3 End: €{ma_predictions[780]:.2f}")
print(f"Year 4 End: €{ma_predictions[-1]:.2f}")
 
# Method 4: Prophet (if available)
if PROPHET_AVAILABLE:
    print(f"\n=== Method 4: Prophet Forecast ===")
   
    try:
        # Prepare data for Prophet
        df_prophet = ticker_df_5y['Close'].reset_index()
        df_prophet.columns = ['ds', 'y']
       
        # Remove timezone information from the datetime column
        if df_prophet['ds'].dt.tz is not None:
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
       
        # Initialize and fit Prophet model
        prophet_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        prophet_model.fit(df_prophet)
       
        # Create future dataframe
        future_prophet = prophet_model.make_future_dataframe(periods=future_days, freq='B')
        forecast = prophet_model.predict(future_prophet)
       
        # Get future predictions only
        # Convert the last date to timezone-naive for comparison
        last_date = ticker_df_5y.index[-1]
        if hasattr(last_date, 'tz_localize'):
            last_date = last_date.tz_localize(None)
       
        future_forecast = forecast[forecast['ds'] > last_date]
       
        print(f"Prophet Forecast for next 4 years:")
        print(f"Year 1 End: €{future_forecast.iloc[260]['yhat']:.2f}")
        print(f"Year 2 End: €{future_forecast.iloc[520]['yhat']:.2f}")
        print(f"Year 3 End: €{future_forecast.iloc[780]['yhat']:.2f}")
        print(f"Year 4 End: €{future_forecast.iloc[-1]['yhat']:.2f}")
       
        prophet_predictions = future_forecast['yhat'].values
       
    except Exception as e:
        print(f"Prophet forecast failed: {e}")
        PROPHET_AVAILABLE = False
 
# Create comprehensive visualization
plt.figure(figsize=(15, 10))
 
# Historical data
plt.subplot(2, 2, 1)
plt.plot(ticker_df_5y.index, ticker_df_5y['Close'], label='Historical Price', linewidth=2)
plt.plot(ticker_df_5y.index, ma_50, label='MA 50', alpha=0.7)
plt.plot(ticker_df_5y.index, ma_200, label='MA 200', alpha=0.7)
plt.title('DTE.DE Historical Prices (5 Years)')
plt.xlabel('Date')
plt.ylabel('Price (€)')
plt.legend()
plt.grid(True, alpha=0.3)
 
# All predictions comparison
plt.subplot(2, 2, 2)
# Show last year of historical data for context
historical_context = ticker_df_5y['Close'][-252:]
context_dates = historical_context.index
 
plt.plot(context_dates, historical_context, label='Historical', linewidth=2, color='black')
plt.plot(future_dates, lr_predictions, label='Linear Regression', alpha=0.8)
plt.plot(future_dates, poly_predictions, label='Polynomial Regression', alpha=0.8)
plt.plot(future_dates, ma_predictions, label='Moving Average Trend', alpha=0.8)
 
if PROPHET_AVAILABLE:
    plt.plot(future_dates, prophet_predictions, label='Prophet', alpha=0.8)
 
plt.title('4-Year Price Predictions Comparison')
plt.xlabel('Date')
plt.ylabel('Price (€)')
plt.legend()
plt.grid(True, alpha=0.3)
 
# Price distribution
plt.subplot(2, 2, 3)
plt.hist(ticker_df_5y['Close'], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(ticker_df_5y['Close'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(ticker_df_5y['Close'].median(), color='green', linestyle='--', label='Median')
plt.title('Price Distribution (5 Years)')
plt.xlabel('Price (€)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
 
# Volume analysis
plt.subplot(2, 2, 4)
plt.plot(ticker_df_5y.index, ticker_df_5y['Volume'], alpha=0.7)
plt.title('Trading Volume (5 Years)')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.show()
 
# Summary of predictions
print(f"\n=== 4-Year Prediction Summary ===")
methods = ['Linear Regression', 'Polynomial Regression', 'Moving Average Trend']
final_predictions = [lr_predictions[-1], poly_predictions[-1], ma_predictions[-1]]
 
if PROPHET_AVAILABLE:
    methods.append('Prophet')
    final_predictions.append(prophet_predictions[-1])
 
for method, prediction in zip(methods, final_predictions):
    change_pct = ((prediction / ticker_df_5y['Close'][-1]) - 1) * 100
    print(f"{method:20}: €{prediction:6.2f} ({change_pct:+5.1f}%)")
 
avg_prediction = np.mean(final_predictions)
avg_change_pct = ((avg_prediction / ticker_df_5y['Close'][-1]) - 1) * 100
print(f"{'Average Prediction':20}: €{avg_prediction:6.2f} ({avg_change_pct:+5.1f}%)")
 
print(f"\n=== Important Disclaimers ===")
print("• Stock predictions are highly uncertain and should not be used for investment decisions")
print("• Past performance does not guarantee future results")
print("• Multiple factors affect stock prices that models cannot capture")
print("• Consider consulting financial advisors for investment decisions")
print("• These are statistical projections, not investment recommendations")