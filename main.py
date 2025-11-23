import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "^NSEI"      
TRADING_DAYS = 252    
CURRENCY_SYMBOL = "₹" 

def fetch_data(ticker, period="1y"):
    """Fetches historical data for the given ticker."""
    print(f"Fetching data for {ticker}...")
    
    data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False, multi_level_index=False)
    
    if data.empty:
        raise ValueError("No data found. Check ticker symbol or internet connection.")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    return data

def run_simulation(data, time_horizon_years=1, num_simulations=100):
    """Runs Monte Carlo simulation using Geometric Brownian Motion."""
    
    try:
        prices = data['Adj Close']
    except KeyError:
        prices = data['Close'] # Fallback
        
    log_returns = np.log(prices / prices.shift(1))
    
    S0 = prices.iloc[-1]  
    mu = log_returns.mean() * TRADING_DAYS  
    sigma = log_returns.std() * np.sqrt(TRADING_DAYS) 
    
    print(f"--- Market Stats ---")
    print(f"Start Price: {CURRENCY_SYMBOL}{S0:,.2f}")
    print(f"Drift (Exp. Return): {mu:.2%}")
    print(f"Volatility (Risk): {sigma:.2%}")

    time_steps = int(TRADING_DAYS * time_horizon_years)
    dt = time_horizon_years / time_steps
    simulation_results = np.zeros((time_steps, num_simulations))
    simulation_results[0] = S0

    for t in range(1, time_steps):
        Z = np.random.normal(0, 1, num_simulations)
        drift_term = (mu - 0.5 * sigma**2) * dt
        shock_term = sigma * np.sqrt(dt) * Z
        simulation_results[t] = simulation_results[t-1] * np.exp(drift_term + shock_term)
        
    return simulation_results, S0, sigma

def plot_results(simulation_results, S0, sigma, ticker):
    """Visualizes the simulation paths."""
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_results, lw=1, alpha=0.3) 
    plt.axhline(y=S0, color='r', linestyle='--', label='Start Price')
    plt.title(f'Geometric Brownian Motion: Projected Paths for {ticker}\n(Volatility: {sigma:.1%})')
    plt.xlabel('Days (Future)')
    plt.ylabel(f'Price ({CURRENCY_SYMBOL})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def main():
    try:
        data = fetch_data(TICKER)
        results, S0, sigma = run_simulation(data)
        
        final_prices = results[-1]
        print(f"\n--- Simulation Results (1 Year) ---")
        print(f"Worst Case (Min): {CURRENCY_SYMBOL}{np.min(final_prices):,.2f}")
        print(f"Best Case (Max):  {CURRENCY_SYMBOL}{np.max(final_prices):,.2f}")
        print(f"Average Outcome:  {CURRENCY_SYMBOL}{np.mean(final_prices):,.2f}")
        
        var_95 = S0 - np.percentile(final_prices, 5)
        print(f"VaR (95% Confidence): Potential Loss > {CURRENCY_SYMBOL}{var_95:,.2f}")
        
        plot_results(results, S0, sigma, TICKER)
        
    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()