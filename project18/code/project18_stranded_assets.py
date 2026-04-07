"""
===============================================================================
PROJECT 18: Stranded Assets Valuation Under Climate Scenarios (Monte Carlo)
===============================================================================
RESEARCH QUESTION:
    What is the potential loss in value for fossil fuel companies under
    different climate transition scenarios (orderly, disorderly, hot house)?
METHOD:
    Monte Carlo simulation of asset valuation under NGFS climate scenarios.
    Model carbon price paths, demand shocks, and stranding probability.
DATA:
    Yahoo Finance for current valuations, NGFS scenario parameters
===============================================================================
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

np.random.seed(42)

# =============================================================================
# STEP 1: Define NGFS scenarios and download company data
# =============================================================================
print("STEP 1: Setting up NGFS climate scenarios and downloading data...")

# NGFS scenario parameters (calibrated to published NGFS Phase IV)
scenarios = {
    'Orderly (Net Zero 2050)': {
        'carbon_price_2030': 130, 'carbon_price_2050': 250,
        'demand_decline_annual': -0.035, 'stranding_prob': 0.25,
        'volatility': 0.20, 'color': '#2ecc71'
    },
    'Disorderly (Delayed Transition)': {
        'carbon_price_2030': 50, 'carbon_price_2050': 350,
        'demand_decline_annual': -0.015, 'stranding_prob': 0.40,
        'volatility': 0.35, 'color': '#f39c12'
    },
    'Hot House World (Current Policies)': {
        'carbon_price_2030': 20, 'carbon_price_2050': 30,
        'demand_decline_annual': -0.005, 'stranding_prob': 0.10,
        'volatility': 0.25, 'color': '#e74c3c'
    }
}

# Download fossil fuel company data
tickers = ['XOM','CVX','COP','BP','SHEL','TTE','EOG','PXD','SLB','HAL']
prices = {}
for t in tickers:
    df = yf.download(t, start='2020-01-01', end='2025-12-31', auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if not df.empty:
        prices[t] = df['Close'].iloc[-1]
        print(f"  {t}: ${df['Close'].iloc[-1]:.2f}")

company_values = pd.Series(prices)
company_values.to_csv('data/current_valuations.csv')

# =============================================================================
# STEP 2: Monte Carlo simulation
# =============================================================================
print("\nSTEP 2: Running Monte Carlo simulations (10,000 paths per scenario)...")

n_sims = 10000
horizon = 25  # years to 2050
years = np.arange(2025, 2050 + 1)

results = {}
for scenario_name, params in scenarios.items():
    print(f"\n  {scenario_name}:")
    
    # Carbon price path (geometric Brownian motion)
    cp_start = 50  # Current carbon price ~$50/ton
    cp_end = params['carbon_price_2050']
    cp_drift = np.log(cp_end / cp_start) / horizon
    
    # Simulate portfolio value paths
    portfolio_paths = np.zeros((n_sims, horizon + 1))
    portfolio_paths[:, 0] = 100  # Normalized to 100
    
    stranding_events = np.zeros(n_sims)
    carbon_paths = np.zeros((n_sims, horizon + 1))
    carbon_paths[:, 0] = cp_start
    
    for t in range(1, horizon + 1):
        # Carbon price evolution
        z_carbon = np.random.normal(0, 1, n_sims)
        carbon_paths[:, t] = carbon_paths[:, t-1] * np.exp(
            (cp_drift - 0.5 * 0.3**2) + 0.3 * z_carbon)
        
        # Demand shock based on carbon price
        carbon_impact = -0.001 * (carbon_paths[:, t] - cp_start)
        
        # Portfolio return
        drift = params['demand_decline_annual'] + carbon_impact
        z_port = np.random.normal(0, 1, n_sims)
        returns = drift + params['volatility'] * z_port
        
        # Stranding events (sudden value destruction)
        strand_shock = np.random.uniform(0, 1, n_sims) < (params['stranding_prob'] / horizon)
        returns[strand_shock] -= np.random.uniform(0.2, 0.5, strand_shock.sum())
        stranding_events += strand_shock
        
        portfolio_paths[:, t] = portfolio_paths[:, t-1] * (1 + returns)
    
    # Ensure no negative values
    portfolio_paths = np.maximum(portfolio_paths, 0.01)
    
    # Statistics
    final_values = portfolio_paths[:, -1]
    results[scenario_name] = {
        'paths': portfolio_paths,
        'carbon_paths': carbon_paths,
        'mean_final': final_values.mean(),
        'median_final': np.median(final_values),
        'p5': np.percentile(final_values, 5),
        'p95': np.percentile(final_values, 95),
        'prob_loss_50pct': (final_values < 50).mean() * 100,
        'prob_loss_90pct': (final_values < 10).mean() * 100,
        'avg_stranding': stranding_events.mean(),
        'var_95': 100 - np.percentile(final_values, 5),
    }
    
    print(f"    Mean final value: {final_values.mean():.1f} (from 100)")
    print(f"    Prob >50% loss:   {(final_values < 50).mean()*100:.1f}%")
    print(f"    95% VaR:          {100 - np.percentile(final_values, 5):.1f}")

# Save results table
summary = pd.DataFrame({s: {k: v for k, v in r.items() if k not in ['paths','carbon_paths']} 
                         for s, r in results.items()}).T
summary.to_csv('output/tables/scenario_results.csv')

# =============================================================================
# STEP 3: Visualizations
# =============================================================================
print("\nSTEP 3: Creating visualizations...")

# Fig 1: Fan charts of portfolio value paths
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for idx, (scenario_name, data) in enumerate(results.items()):
    ax = axes[idx]
    paths = data['paths']
    color = scenarios[scenario_name]['color']
    
    # Plot percentile bands
    for p_low, p_high, alpha in [(5,95,0.15),(10,90,0.2),(25,75,0.3)]:
        low = np.percentile(paths, p_low, axis=0)
        high = np.percentile(paths, p_high, axis=0)
        ax.fill_between(years, low, high, alpha=alpha, color=color)
    
    ax.plot(years, np.median(paths, axis=0), color=color, linewidth=2, label='Median')
    ax.plot(years, np.mean(paths, axis=0), color=color, linewidth=2, linestyle='--', label='Mean')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.3)
    ax.set_title(scenario_name, fontweight='bold', fontsize=11)
    ax.set_xlabel('Year')
    if idx == 0: ax.set_ylabel('Portfolio Value (base=100)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(300, np.percentile(paths, 95, axis=0).max() * 1.1))

plt.suptitle('Fossil Fuel Portfolio Value Under NGFS Climate Scenarios', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('output/figures/fig1_scenario_fan_charts.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Final value distributions
fig, ax = plt.subplots(figsize=(12, 6))
for scenario_name, data in results.items():
    color = scenarios[scenario_name]['color']
    ax.hist(data['paths'][:, -1], bins=80, alpha=0.4, color=color, label=scenario_name, density=True)
    ax.axvline(data['mean_final'], color=color, linestyle='--', linewidth=2)

ax.axvline(100, color='black', linestyle=':', label='Initial Value')
ax.set_title('Distribution of Final Portfolio Values (2050)', fontweight='bold')
ax.set_xlabel('Portfolio Value (base=100)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig('output/figures/fig2_final_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Carbon price paths
fig, ax = plt.subplots(figsize=(12, 6))
for scenario_name, data in results.items():
    color = scenarios[scenario_name]['color']
    median_cp = np.median(data['carbon_paths'], axis=0)
    p5 = np.percentile(data['carbon_paths'], 5, axis=0)
    p95 = np.percentile(data['carbon_paths'], 95, axis=0)
    ax.plot(years, median_cp, color=color, linewidth=2, label=scenario_name)
    ax.fill_between(years, p5, p95, alpha=0.15, color=color)

ax.set_title('Carbon Price Paths Under NGFS Scenarios', fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('Carbon Price ($/ton)')
ax.legend()
plt.tight_layout()
plt.savefig('output/figures/fig3_carbon_prices.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 4: Risk metrics comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['prob_loss_50pct', 'var_95', 'mean_final']
titles = ['Probability of >50% Loss (%)', '95% Value at Risk', 'Mean Final Value']
for idx, (metric, title) in enumerate(zip(metrics, titles)):
    vals = [results[s][metric] for s in results]
    colors = [scenarios[s]['color'] for s in results]
    axes[idx].bar(range(len(results)), vals, color=colors)
    axes[idx].set_xticks(range(len(results)))
    axes[idx].set_xticklabels(['Orderly','Disorderly','Hot House'], rotation=20)
    axes[idx].set_title(title, fontweight='bold')

plt.tight_layout()
plt.savefig('output/figures/fig4_risk_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

print("  COMPLETE!")
