# Stranded Assets Valuation Under Climate Scenarios

## Research Question
What is the potential value loss for fossil fuel portfolios under NGFS climate transition scenarios?

## Methodology
**Language:** Python  
**Methods:** Monte Carlo simulation, NGFS scenarios

## Data
Yahoo Finance for current valuations, NGFS scenario parameters

## Key Findings
Orderly transition: 84% probability of >50% loss; Hot House: 48% probability; carbon price path is key driver.

## How to Run
```bash
pip install -r requirements.txt
python code/project18_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*
