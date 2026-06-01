"""
 (Durbin & Koopman, 2012, ekvation 2.3):
    y_t     = μ_t + ε_t,       ε_t ~ N(0, σ²_ε) 
    μ_{t+1} = μ_t + η_t,      η_t ~ N(0, σ²_η)

Matriserna sätts explicit i LocalLevel-klassen:
    Z = 1   (design)
    T = 1   (transition)
    R = 1   (selection)
    H = σ²_ε  (obs_cov, skattas via MLE)
    Q = σ²_η  (state_cov, skattas via MLE)

Skattning sker i tre steg:
  1. model.fit() kör MLE via L-BFGS-optimering och maximerar
     log-likelihood för att hitta σ²_ε och σ²_η.

  2. Kalman-smoothern (results.smoothed_state) ger den bästa
     skattningen av μ_t givet ALLA observationer y_1,...,y_n.
     Används för skattning (prediktion) och visualisering. 
     Kalman smoothern används också för utvärderingsmåtten.

  3. Kalman-filtret används för one-step-ahead prediktioner och används för att beräkna durbin watson.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.tsa.stattools import acf as sm_acf
except ImportError:
    raise ImportError("Installera statsmodels: pip install statsmodels")


class LocalLevel(MLEModel):

    # Endast σ²_η skattas via MLE; σ²_ε är hårdkodad till 0.1² = 0.01
    start_params = [1.0]
    param_names = ["sigma2.level"]
    SIGMA2_EPS = 0.1**2  # = 0.01, fast observationsvarians

    def __init__(self, endog):
        super().__init__(endog, k_states=1)

        # Z = 1
        self["design", 0, 0] = 1.0

        # T = 1
        self["transition", 0, 0] = 1.0

        # R = 1
        self["selection", 0, 0] = 1.0

        # H = σ²_ε: hårdkodad
        self["obs_cov", 0, 0] = self.SIGMA2_EPS

        # Diffus initiering, inget antagande om startvärde
        self.initialize_approximate_diffuse()

    def transform_params(self, params):
        # Kvadrering säkerställer att variansen alltid är positiv
        return params**2

    def untransform_params(self, params):
        return params**0.5

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        # Q = σ²_η: processbruset (hur mycket nivån rör sig)
        self["state_cov", 0, 0] = params[0]


#får in alla filer på samma ställe som detta skript
basror_22W102  = Path(__file__).parent / "22W102.csv"
basror_17XX01U = Path(__file__).parent / "17XX01U.csv"
basror_G1101   = Path("/home/angelica/Hämtningar/obs2.csv")

#nedan ser vi att observationsrör 22W102=rör A, G1101=rör B, 17XX01U=rör C
STATIONS = [
    (basror_22W102, "Rör A"),
    (basror_G1101, "Rör B"),
    (basror_17XX01U, "Rör C"),

]

#Tar in datan korrekt
def load_base_station(filepath: Path) -> pd.Series:
    with open(filepath, encoding="utf-8-sig") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","

    has_header = any(
        word in first_line.lower()
        for word in ["date", "datum", "level", "nivå", "niva"]
    )

    df = pd.read_csv(
        filepath,
        sep=sep,
        header=0 if has_header else None,
        names=None if has_header else ["date", "level"],
        encoding="utf-8-sig",
    )


    df.columns = [c.strip().lower() for c in df.columns]
    date_col = [c for c in df.columns if "dat" in c][0]
    level_col = [c for c in df.columns if c != date_col][0]

    # Försök först med automatisk tolkning; om det ger orimliga datum
    # (t.ex. YY/MM/DD tolkas fel) → prova explicit %y/%m/%d
    try:
        parsed = pd.to_datetime(df[date_col], format="%y/%m/%d")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(df[date_col], format="mixed", dayfirst=False)
    df[date_col] = parsed
    df[level_col] = (
        df[level_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df[level_col] = pd.to_numeric(df[level_col], errors="coerce")
    df = df.set_index(date_col).sort_index()
    return df[level_col].rename("level")


def detect_frequency(series: pd.Series) -> str:
    """
    Väljer tidsfrekvens automatiskt baserat på datadensitet.
    Tre nivåer: veckovis, varannan vecka, eller månadsvis.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return "7D"

    # Antal dagar från första till sista mätningen
    span_days = (valid.index.max() - valid.index.min()).days

    # Genomsnittligt antal dagar mellan mätningar
    avg_gap = span_days / max(len(valid) - 1, 1)

    # om avg_gap är >21 då kör vi på 1 månad mellan alla observationer eftersom avståndet mellan observationerna behöver vara konstand
    if avg_gap > 21:
        return "MS"   
    # om avg_gap är >10 då kör vi på varannan vecka mellan alla observationer i den tidserien
    elif avg_gap > 10:
        return "14D"  
    else:
    # Annars kör vi veckovis mellan alla observationer 
        return "7D"    


# Här resamplar vi så vi får samma avstånd mellan flera observationer om det finns flera mätvärden ex inom en vecka tar vi istället medelvärdet
def prepare_series(series: pd.Series, freq: str) -> pd.Series:
    resampled = series.resample(freq).mean()
    return resampled





def fit_model(y: pd.Series) -> dict:
    #modelstrukturen skapas
    model = LocalLevel(y)
    #lbfgs optimeringen testar olika värden på σ²_ε och σ²_η för att hitta de som maximerar log-likelihood    
    # maxiter=2000 ger optimeraren upp efter 2000 försök, disp=False döljer utskriften 
    results = model.fit(method="lbfgs", maxiter=2000, disp=False)
    #kalman smoothern ser all data, [0, :] tar det första tillståndet ,det finns bara 1, och alla observationer
    smoothed_level = results.smoothed_state[0, :]
    #variansen för smoothed nivån
    smoothed_level_var = results.smoothed_state_cov[0, 0, :]

    # σ²_ε är hårdkodad i modellen
    sigma2_eps = LocalLevel.SIGMA2_EPS

    # Beräknar 95%-konfidensintervall för den smoothed nivån
    z95 = sp_stats.norm.ppf(0.975) 
    pred_std = np.sqrt(np.maximum(smoothed_level_var + sigma2_eps, 0))
    pred_ci = np.column_stack([
        smoothed_level - z95 * pred_std,
        smoothed_level + z95 * pred_std,
    ])

    return {
        "results":          results,       
        "index":            y.index,        
        "observed_base":    y.values,       
        "level_smoothed":   smoothed_level, 
        "pred_base":        smoothed_level, 
        "pred_ci":          pred_ci,        
    }



"""
Utvärderings mått: MAE, RMSE, Durbin-Watson.
"""

def evaluate_model(out: dict, station_id: str = "") -> dict:

    obs = out["observed_base"]
    pred = out["pred_base"]

    # Mask: tidpunkter med faktisk observation OCH giltig prediktion
    valid = ~np.isnan(obs) & ~np.isnan(pred)
    o = obs[valid]
    p = pred[valid]
    resid = o - p

    # Burn-in: hoppa över de första tidstegen (diffus initiering)
    burn = min(10, len(resid) - 10)
    resid_eval = resid[burn:]
    o_eval = o[burn:]


    metrics = {
        "MAE":            float(np.mean(np.abs(resid_eval))),
        "RMSE":           float(np.sqrt(np.mean(resid_eval**2))),
    }

    #Durbin-Watson 
    obs_mask = ~np.isnan(obs)
    try:
        std_innov = out["results"].filter_results.standardized_forecasts_error[0]
        si_valid = std_innov[obs_mask][burn:]
        si_valid = si_valid[~np.isnan(si_valid)]
        if len(si_valid) > 10:
            metrics["Durbin-Watson"] = float(durbin_watson(si_valid))
    except Exception:
        pass


    print(f"\n  DIAGNOSTIK: {station_id}")
    print(f"  {'-'*40}")
    print(f"  {'σ²_ε (obs-brus)':<20} {LocalLevel.SIGMA2_EPS:.6f} (fast)")
    print(f"  {'σ²_η (process-brus)':<20} {out['results'].params.iloc[0]:.6f}")
    print(f"  {'MAE':<20} {metrics['MAE']:.4f}")
    print(f"  {'RMSE':<20} {metrics['RMSE']:.4f}")
    print(f"  {'Durbin-Watson':<20} {metrics.get('Durbin-Watson', float('nan')):.4f}")
    print(f"  {'-'*40}")

    return metrics



#visualisering
def plot_results(
    out: dict,
    station_id: str = "22W102",
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "State Space Model (Local Level)\n"
        f"Observationsrör: {station_id}",
        fontsize=14, fontweight="bold",
    )

    idx      = out["index"]
    obs      = out["observed_base"]
    pred     = out["pred_base"].copy()
    ci       = out["pred_ci"].copy()
    smoothed = out["level_smoothed"].copy()

    # Första prediktionen är 0 p.g.a. diffus initialisering — dölj den
    pred[0] = np.nan
    ci[0, :] = np.nan
    smoothed[0] = np.nan

    # Konfidensintervall (interpolera över eventuella NaN så bandet blir kontinuerligt)
    ci_lower = pd.Series(ci[:, 0], index=idx).interpolate(method="index").values
    ci_upper = pd.Series(ci[:, 1], index=idx).interpolate(method="index").values
    ax.fill_between(
        idx, ci_lower, ci_upper,
        color="steelblue", alpha=0.20,
        label="95% konfidensintervall",
    )

<<<<<<< HEAD
        # y_t — observerad nivå
    obs_valid = ~np.isnan(obs)
    obs_dates = idx[obs_valid]
    obs_vals = obs[obs_valid]

    # Dra linje bara mellan observationer som ligger nära varandra
    # Beräkna max tillåtet gap (3x mediangapet)
    if len(obs_dates) > 1:
        gaps = np.diff(obs_dates).astype("timedelta64[D]").astype(float)
        max_gap = np.median(gaps) * 3

        # Sätt NaN i serien där gapet är för stort → bryter linjen
        obs_plot = obs_vals.copy()

        for i in range(1, len(gaps)):
            if gaps[i] > max_gap:
                obs_plot[i] = np.nan

    else:
        obs_plot = obs_vals

    # Observationer som blå prickar
    ax.plot(
    obs_dates,
    obs_plot,
    "o",
    color="navy",
    markersize=3,
    alpha=0.7,
    label=r"$y_t$ (observation)",
    )

    # Skattade värden som röd heldragen linje
    ax.plot(
    idx,
    smoothed,
    linestyle="-",
    linewidth=1.4,
    color="red",
    alpha=0.9,
    label=r"$\mu_t$ (latent nivå), skattad serie",
)


=======
    # μ_t — smoothad latent nivå
    ax.plot(
        idx, smoothed,
        linestyle="-", linewidth=1.4, color="crimson",
        label=r"$\mu_t$ (latent nivå) predikterad serie",
    )

    # y_t — observerad nivå (enbart prickar)
    obs_valid = ~np.isnan(obs)
    ax.plot(
        idx[obs_valid], obs[obs_valid],
        "o", color="navy", markersize=3, alpha=0.75,
        label=r"$y_t$ (observation)",
    )
>>>>>>> 75cdf97bb43dc20330df4851808db080306a8799

    ax.set_xlabel("Datum")
    ax.set_ylabel("Nivå (m ö.h.)")
    ax.legend(loc="upper left", fontsize=8)

    # Datumaxel
    n_months = (idx[-1] - idx[0]).days / 30
    interval = max(4, int(n_months / 8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    ax.grid(True, alpha=0.3)

    # Fast ±0.5 marginal på y-axeln
    all_vals = pd.concat([
        pd.Series(obs[~np.isnan(obs)]),
        pd.Series(smoothed[~np.isnan(smoothed)]),
    ]).dropna()
    if len(all_vals) > 0:
        ax.set_ylim(all_vals.min() - 0.5, all_vals.max() + 0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig




def plot_acf_residuals(out: dict, station_id: str, lags: int = 20) -> plt.Figure:
    """
    ACF-plot för residualerna (observation − one-step-ahead prediktion).
    Lodräta staplar, streckade CI-linjer, grid.
    """
    obs  = out["observed_base"]
    pred = out["pred_base"].flatten()[:len(obs)]
    residualer = obs - pred

    residualer_series = pd.Series(residualer).dropna()
    residualer_series = residualer_series[~np.isnan(residualer_series.values)]

    if len(residualer_series) < 10:
        print(f"  För få residualer för ACF-plot")
        return None

    max_lags = min(lags, len(residualer_series) // 2 - 1)
    acf_vals = sm_acf(residualer_series, nlags=max_lags, fft=True)
    ci_bound = 1.96 / np.sqrt(len(residualer_series))
    lags_arr = np.arange(len(acf_vals))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.vlines(lags_arr, 0, acf_vals, colors="steelblue", linewidth=1.5)
    ax.plot(lags_arr, acf_vals, "o", color="steelblue", markersize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline( ci_bound, linestyle="--", color="steelblue", linewidth=0.9)
    ax.axhline(-ci_bound, linestyle="--", color="steelblue", linewidth=0.9)


    ax.set_xlabel("Lag")
    ax.set_ylabel("Autokorrelation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_histogram_residuals(out: dict, station_id: str) -> plt.Figure:
    """
    Histogram av residualerna med överlagrad normalfördelningskurva.
    """
    obs = out["observed_base"]
    pred = out["pred_base"].flatten()[:len(obs)]
    residualer = pd.Series(obs - pred).dropna().values

    if len(residualer) < 10:
        print("  För få residualer för histogram")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(residualer, bins="auto", density=True, alpha=0.7,
            color="steelblue", edgecolor="white", label="Residualer")

    mu, sigma = residualer.mean(), residualer.std()
    x = np.linspace(residualer.min(), residualer.max(), 200)
    ax.plot(x, sp_stats.norm.pdf(x, mu, sigma),
            color="red", lw=2, label=f"N({mu:.4f}, {sigma:.4f}²)")


    ax.set_xlabel("Residual (m)")
    ax.set_ylabel("Densitet")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_residuals_over_time(out: dict, station_id: str) -> plt.Figure:
    """
    Residualer över tid:
    residual = observation - one-step-ahead prediction
    """

    obs = out["observed_base"]

    # one-step-ahead prediction från Kalman-filtret
    pred = out["pred_base"].flatten()[:len(obs)]

    residuals = obs - pred

    # Tar bort NaN
    valid = ~np.isnan(residuals)

    dates = out["index"][valid]
    residuals = residuals[valid]

    if len(residuals) < 5:
        print("  För få residualer för residualplot")
        return None

    fig, ax = plt.subplots(figsize=(14, 6))


    # Residualserie
    ax.plot(
        dates,
        residuals,
        "o-",
        color="steelblue",
        linewidth=1.2,
        markersize=3,
        alpha=0.8,
    )

    # Horisontell linje vid 0
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Datum")
    ax.set_ylabel("Residual")

    # Datumformat
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    n_months = (dates[-1] - dates[0]).days / 30
    interval = max(4, int(n_months / 8))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


if __name__ == "__main__":

    #skapar en tom dictionary där alla utvärderingsmått kommer sparas
    all_metrics = {}

    #loopar över alla stationer i stations
    for station_path, display_name in STATIONS:
        file_id = display_name.lower().replace(" ", "_").replace("ö", "o")
        plot_path = f"ssm_results_{file_id}.png"
        base = load_base_station(station_path)



        #Kollar hur ofta vi har observationer och resamplar
        freq = detect_frequency(base)
        print(f"  Detekterad frekvens: {freq}")
        y = prepare_series(base, freq)
   

        # Anpassar modellen
        print("\n=== Anpassar State Space Model (Local Level) ===")
        #fit_model() skapar local level modellen
        out = fit_model(y)

        # Utvärdering
        metrics = evaluate_model(out, display_name)
        all_metrics[display_name] = metrics

        # Plotta och spara
        fig = plot_results(out, display_name)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ACF-plot för residualer
        fig_acf = plot_acf_residuals(out, display_name, lags=10)
        if fig_acf is not None:
            acf_path = f"acf_residuals_{file_id}.png"
            fig_acf.savefig(acf_path, dpi=150, bbox_inches="tight")
            print(f"  ACF-plot sparad: {acf_path}")
            plt.close(fig_acf)

        # Histogram av residualer
        fig_hist = plot_histogram_residuals(out, display_name)
        if fig_hist is not None:
            hist_path = f"hist_residuals_{file_id}.png"
            fig_hist.savefig(hist_path, dpi=150, bbox_inches="tight")
            print(f"  Histogram sparad: {hist_path}")
            plt.close(fig_hist)


        # Residualer över tid
        fig_resid = plot_residuals_over_time(out, display_name)

        if fig_resid is not None:
            resid_path = f"residuals_over_time_{file_id}.png"

            fig_resid.savefig(
                resid_path,
                dpi=150,
                bbox_inches="tight"
            )

            print(f"  Residualplot sparad: {resid_path}")

            plt.close(fig_resid)

    # ── Sammanfattning ──
    if all_metrics:
        print("\n\n" + "=" * 55)
        print(" SAMMANFATTNING AV ALLA STATIONER")
        print("=" * 55)
        header = f"  {'Station':<12} {'MAE':>8} {'RMSE':>8} {'DW':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for sid, m in all_metrics.items():
            print(
                f"  {sid:<12} "
                f"{m.get('MAE', float('nan')):>8.4f} "
                f"{m.get('RMSE', float('nan')):>8.4f} "
                f"{m.get('Durbin-Watson', float('nan')):>8.4f}"
            )
        print("=" * 55)






    # ══════════════════════════════════════════════════════════════════
    # SÄSONGSMODELL: Local Level + deterministisk säsong (nseason=26)
    # ══════════════════════════════════════════════════════════════════

    class LocalLevelSeasonal(MLEModel):
        """
        Lokal nivå + säsongskomponent (sum-to-zero, nseason perioder).
        Tillståndsvektor: [µ, γ_0, γ_1, …, γ_{nseason-2}]
        y_t = µ_t + γ_{0,t} + ε_t
        """

        param_names = ["sigma2.level", "sigma2.season"]
        start_params = [1.0, 0.01]
        SIGMA2_EPS = 0.1**2  # fast observationsvarians

        def __init__(self, endog, nseason=26):
            self.nseason = nseason
            k_states = 1 + (nseason - 1)  # µ + säsongstillstånd
            super().__init__(endog, k_states=k_states, k_posdef=2)

            # Z: observation = µ + γ_0
            self["design"] = np.zeros((1, k_states))
            self["design", 0, 0] = 1.0  # µ
            self["design", 0, 1] = 1.0  # γ_0

            # T: transition
            # µ: random walk
            self["transition"] = np.zeros((k_states, k_states))
            self["transition", 0, 0] = 1.0

            # Säsong: shift-operator med sum-to-zero
            # γ_{t+1} = -γ_0 - γ_1 - … - γ_{s-2}, sedan shift
            for j in range(1, nseason - 1):
                self["transition", 1, j] = -1.0
            for j in range(2, nseason - 1):
                self["transition", j, j - 1] = 1.0

            # R: selection (processbrus påverkar µ och γ_0)
            self["selection"] = np.zeros((k_states, 2))
            self["selection", 0, 0] = 1.0  # η_µ
            self["selection", 1, 1] = 1.0  # η_γ

            # H: obs_cov
            self["obs_cov", 0, 0] = self.SIGMA2_EPS

            self.initialize_approximate_diffuse()

        def transform_params(self, params):
            return params**2

        def untransform_params(self, params):
            return params**0.5

        def update(self, params, **kwargs):
            params = super().update(params, **kwargs)
            # Q: state_cov (2×2 diagonal)
            self["state_cov", 0, 0] = params[0]  # σ²_η_level
            self["state_cov", 1, 1] = params[1]  # σ²_η_season

    def fit_model_seasonal(y: pd.Series, freq: str = "14D") -> dict:
        """Anpassar LocalLevelSeasonal och returnerar resultat-dict."""
        # nseason = antal observationer per år (en hel säsongscykel)
        if freq == "7D":
            nseason = 52
        elif freq == "14D":
            nseason = 26
        else:  # "MS"
            nseason = 12
        model = LocalLevelSeasonal(y, nseason=nseason)
        results = model.fit(method="lbfgs", maxiter=3000, disp=False)

        smoothed = results.smoothed_state
        level = smoothed[0, :]
        season = smoothed[1, :]
        pred = level + season

        # Korrekt varians: Var(µ+γ) = Var(µ) + Var(γ) + 2·Cov(µ,γ)
        var_level = results.smoothed_state_cov[0, 0, :]
        var_season = results.smoothed_state_cov[1, 1, :]
        cov_ls = results.smoothed_state_cov[0, 1, :]
        smoothed_var = var_level + var_season + 2 * cov_ls

        sigma2_eps = LocalLevelSeasonal.SIGMA2_EPS
        z95 = sp_stats.norm.ppf(0.975)
        pred_std = np.sqrt(np.maximum(smoothed_var + sigma2_eps, 0))
        pred_ci = np.column_stack([
            pred - z95 * pred_std,
            pred + z95 * pred_std,
        ])

        return {
            "results":        results,
            "index":          y.index,
            "observed_base":  y.values,
            "level_smoothed": level,
            "season":         season,
            "pred_base":      pred,
            "pred_ci":        pred_ci,
            "nseason":        nseason,
        }

    def evaluate_seasonal(out: dict, station_id: str) -> dict:
        """Utvärdering av säsongsmodellen: MAE, RMSE, Durbin-Watson."""
        obs = out["observed_base"]
        pred = out["pred_base"]
        valid = ~np.isnan(obs) & ~np.isnan(pred)
        resid = obs[valid] - pred[valid]

        burn = min(10, len(resid) - 10)
        resid_eval = resid[burn:]

        metrics = {
            "MAE":  float(np.mean(np.abs(resid_eval))),
            "RMSE": float(np.sqrt(np.mean(resid_eval**2))),
        }

        obs_mask = ~np.isnan(obs)
        try:
            std_innov = out["results"].filter_results.standardized_forecasts_error[0]
            si_valid = std_innov[obs_mask][burn:]
            si_valid = si_valid[~np.isnan(si_valid)]
            if len(si_valid) > 10:
                metrics["Durbin-Watson"] = float(durbin_watson(si_valid))
        except Exception:
            pass

        print(f"\n  DIAGNOSTIK SÄSONG: {station_id}")
        print(f"  {'-'*40}")
        print(f"  {'σ²_η level':<20} {out['results'].params.iloc[0]:.6f}")
        print(f"  {'σ²_η season':<20} {out['results'].params.iloc[1]:.6f}")
        print(f"  {'MAE':<20} {metrics['MAE']:.4f}")
        print(f"  {'RMSE':<20} {metrics['RMSE']:.4f}")
        print(f"  {'Durbin-Watson':<20} {metrics.get('Durbin-Watson', float('nan')):.4f}")
        print(f"  {'-'*40}")
        return metrics

    def plot_seasonal_results(out: dict, station_id: str) -> plt.Figure:
        """Tidsserie-plot för säsongsmodellen (nivå+säsong), samma stil som plot_results."""
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle(
            f"SSM med säsong: {station_id}",
            fontsize=14, fontweight="bold",
        )
        idx = out["index"]
        obs = out["observed_base"]
        pred = out["pred_base"].copy()
        ci = out["pred_ci"].copy()

        # Dölj bara första tidsteget (diffus init)
        pred[0] = np.nan
        ci[0, :] = np.nan

        # Konfidensintervall (interpolera så bandet blir kontinuerligt)
        ci_lower = pd.Series(ci[:, 0], index=idx).interpolate(method="index").values
        ci_upper = pd.Series(ci[:, 1], index=idx).interpolate(method="index").values
        ax.fill_between(
            idx, ci_lower, ci_upper,
            color="steelblue", alpha=0.20,
            label="95% konfidensintervall",
        )

        # Observationer — bryt linje vid stora gap
        obs_valid = ~np.isnan(obs)
        obs_dates = idx[obs_valid]
        obs_vals = obs[obs_valid]

        if len(obs_dates) > 1:
            gaps = np.diff(obs_dates).astype("timedelta64[D]").astype(float)
            max_gap = np.median(gaps) * 3
            obs_plot = obs_vals.copy()
            for i in range(1, len(gaps)):
                if gaps[i] > max_gap:
                    obs_plot[i] = np.nan
        else:
            obs_plot = obs_vals

        ax.plot(obs_dates, obs_plot, "o-", color="navy",
                linewidth=1.0, markersize=2, alpha=0.7,
                label=r"$y_t$ (observation)")

        # Smoothed prediktion (µ+γ) som svart linje
        ax.plot(idx, pred, linestyle="-", linewidth=1.4, color="black", alpha=0.9,
                label=r"$\mu_t + \gamma_t$ (latent nivå + säsong)")

        ax.set_ylabel("Nivå (m ö.h.)")
        ax.set_xlabel("Datum")
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        n_months = (idx[-1] - idx[0]).days / 30
        interval = max(4, int(n_months / 8))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
        ax.grid(alpha=0.3)

        # y-axel inkluderar obs + CI
        all_vals = np.concatenate([obs[~np.isnan(obs)], ci_lower[~np.isnan(ci_lower)], ci_upper[~np.isnan(ci_upper)]])
        if len(all_vals) > 0:
            margin = (all_vals.max() - all_vals.min()) * 0.05
            margin = max(margin, 0.1)
            ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)

        plt.tight_layout()
        return fig

    # ── Kör säsongsmodellen för alla stationer ──
    print("\n\n" + "=" * 55)
    print(" SÄSONGSMODELL (Local Level + Seasonal)")
    print("=" * 55)

    all_metrics_season = {}

    for station_path, display_name in STATIONS:
        file_id = display_name.lower().replace(" ", "_").replace("ö", "o")
        base = load_base_station(station_path)
        freq = detect_frequency(base)
        y = prepare_series(base, freq)

        print(f"\n── Säsongsmodell: {display_name} (freq={freq}, nseason={'52' if freq=='7D' else '26' if freq=='14D' else '12'}) ──")
        out_s = fit_model_seasonal(y, freq=freq)

        metrics_s = evaluate_seasonal(out_s, display_name)
        all_metrics_season[display_name] = metrics_s

        # Tidsserie-plot
        fig_s = plot_seasonal_results(out_s, display_name)
        fig_s.savefig(f"ssm_seson_{file_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  Sparad: ssm_seson_{file_id}.png")

        # Histogram av residualer
        out_s_hist = {
            "observed_base": out_s["observed_base"],
            "pred_base": out_s["pred_base"],
            "index": out_s["index"],
        }
        fig_hist_s = plot_histogram_residuals(out_s_hist, display_name)
        if fig_hist_s is not None:
            hist_path_s = f"hist_residuals_seson_{file_id}.png"
            fig_hist_s.savefig(hist_path_s, dpi=150, bbox_inches="tight")
            print(f"  Histogram sparad: {hist_path_s}")
            plt.close(fig_hist_s)

    # Sammanfattning säsongsmodell
    if all_metrics_season:
        print("\n\n" + "=" * 55)
        print(" SAMMANFATTNING SÄSONGSMODELL")
        print("=" * 55)
        header = f"  {'Station':<12} {'MAE':>8} {'RMSE':>8} {'DW':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for sid, m in all_metrics_season.items():
            print(
                f"  {sid:<12} "
                f"{m.get('MAE', float('nan')):>8.4f} "
                f"{m.get('RMSE', float('nan')):>8.4f} "
                f"{m.get('Durbin-Watson', float('nan')):>8.4f}"
            )
        print("=" * 55)


