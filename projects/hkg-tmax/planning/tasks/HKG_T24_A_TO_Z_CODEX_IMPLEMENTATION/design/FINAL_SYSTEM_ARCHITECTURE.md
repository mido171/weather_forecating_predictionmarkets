# Final System Architecture

```text
Exact-vintage official HKO forecast vintages
        +
GFS deterministic trajectories
        +
GEFS member distribution
        +
IFS/IFS ensemble challengers
        +
AI/CWA/ARWF challengers
        +
pre-cutoff station-network state
        +
causally available HKO target memory
        +
diagnostic-to-safe physics students
        ↓
family-specific expert forecasts and expected-error models
        ↓
static + context-dynamic expected-error router with abstention
        ↓
bounded specialist corrections with benefit gates
        ↓
distributional residual calibration
        ↓
conditional-median Tmax point forecast + calibrated probabilities
```

The official HKO forecast remains the central anchor. NWP supplies explicit forward atmospheric evolution. Stations correct local microclimate mismatch. Long history supplies climatology and regime priors. ML predicts residuals, expected expert loss, regime benefit, and uncertainty; it does not indiscriminately relearn the atmosphere from all columns at once.
