# risk-analytics-engine
Portfolio risk analytics engine in Python: returns, covariance, VaR/ES (historical/parametric/MC), stress testing, and VaR backtesting.

> **[EN]** A modular Python library for portfolio risk analytics: return computation, VaR/ES estimation (historical, parametric, Cornish-Fisher, Monte Carlo), stress testing, and VaR backtesting via the Kupiec test.
>
> **[IT]** Libreria Python modulare per l'analisi del rischio di portafoglio: calcolo dei rendimenti, stima di VaR/ES (storico, parametrico, Cornish-Fisher, Monte Carlo), stress testing e backtesting del VaR tramite il test di Kupiec.

---

## Modules / Moduli

| Module | Description (EN) | Descrizione (IT) |
|---|---|---|
| `returns.py` | Log/simple return computation and horizon aggregation | Calcolo rendimenti logaritmici/semplici e aggregazione multi-periodo |
| `portfolio.py` | Weight validation and portfolio return series | Validazione dei pesi e costruzione dei rendimenti di portafoglio |
| `var.py` | Historical VaR/ES, Parametric VaR, Cornish-Fisher VaR | VaR storico/ES, VaR parametrico, VaR corretto per skewness e kurtosi |
| `monte_carlo.py` | Multivariate Monte Carlo VaR via Cholesky decomposition | VaR Monte Carlo multivariato con decomposizione di Cholesky |
| `stress.py` | Mean/volatility shock application to return series | Applicazione di shock di media e volatilità ai rendimenti |
| `backtesting.py` | Kupiec unconditional coverage test for VaR validation | Test di Kupiec per la validazione statistica del modello VaR |

---

## Requirements / Requisiti

Python **3.10+** and the following packages:

```txt
numpy
pandas
scipy
```

---

## Project Structure / Struttura del Progetto
