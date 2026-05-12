#-----------------------------------Visuella metoden kod: utvärderingsmått - Kandidatuppsats-----------------------------------------------

#---------------------------------------------------- Installerar paket --------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as sp_stats

#---------------------------------------------------- Figurinställningar --------------------------------------------------------------

#gör alla vanliga figurtitlar feta automatiskt
plt.rcParams["axes.titleweight"] = "bold"

#gör även överordnade figurtitlar feta automatiskt
plt.rcParams["figure.titleweight"] = "bold"


#---------------------------------------------------- Interaktiv inläsning av csv-fil --------------------------------------------------------------

#skapar en funktion som frågar efter sökvägen till csv-filen tills en giltig fil anges
def funktion_las_in_visuell_fil():
    
    while True:
        #ber användaren skriva in full sökväg till csv-filen
        sokvag_text = input(
            "\nAnge full sökväg till csv-filen för visuella metoden: "
        ).strip()
        
        #tar bort eventuella citationstecken som råkat följa med vid inklistring
        sokvag_text = sokvag_text.strip('"').strip("'")
        
        #gör om texten till ett Path-objekt
        data_path = Path(sokvag_text)
        
        #kontrollerar att filen finns
        if not data_path.exists():
            print("Filen kunde inte hittas. Kontrollera sökvägen och försök igen.")
            continue
        
        #kontrollerar att sökvägen verkligen pekar på en fil
        if not data_path.is_file():
            print("Sökvägen pekar inte på en fil. Försök igen.")
            continue
        
        #om allt ser bra ut returneras sökvägen
        return data_path


#---------------------------------------------------- Laddar in datat på korrekt sätt --------------------------------------------------------------

#läser in sökvägen interaktivt
data_path = funktion_las_in_visuell_fil()

#anger vilket observationsrör figurerna gäller
#ändra manuellt till "Rör A", "Rör B" eller "Rör C"
observationsror_namn = "Rör C"

#skapar ett filnamnsvänligt id som används när figurer sparas
filnamn_id = (
    observationsror_namn
    .lower()
    .replace(" ", "_")
    .replace("ö", "o")
    .replace("å", "a")
    .replace("ä", "a")
)

#läser in csv-filen
#header=None används eftersom vi vill behandla filen som rå data
#names anger tydligt vilken kolumn som är vad
visuell_df = pd.read_csv(
    data_path,
    sep=";",
    decimal=",",
    header=None,
    names=["datum_obs", "obs_varde", "datum_matchad", "matchad_varde"]
)

#byter ut texten #SAKNAS! mot riktiga saknade värden från de matchade värdena
visuell_df = visuell_df.replace("#SAKNAS!", pd.NA)

#tar bort eventuella extra blanksteg i textkolumner
for kolumn in ["datum_obs", "obs_varde", "datum_matchad", "matchad_varde"]:
    visuell_df[kolumn] = visuell_df[kolumn].astype("string").str.strip()


#--------------------------------------- Konverterar datumkolumner och värdekolumner --------------------------------------------------------------

#konverterar observationsrörets datum till datumformat
visuell_df["datum_obs"] = pd.to_datetime(
    visuell_df["datum_obs"],
    errors="coerce"
)

#konverterar den matchade seriens datum till datumformat
visuell_df["datum_matchad"] = pd.to_datetime(
    visuell_df["datum_matchad"],
    errors="coerce"
)

#konverterar observationsvärden till numeriska värden
visuell_df["obs_varde"] = pd.to_numeric(
    visuell_df["obs_varde"],
    errors="coerce"
)

#konverterar matchade värden till numeriska värden
visuell_df["matchad_varde"] = pd.to_numeric(
    visuell_df["matchad_varde"],
    errors="coerce"
)


#---------------------------------------------- Skapar två separata tabeller --------------------------------------------------------------

#skapar en tabell för observationsröret från kolumn A och B
observations_df = visuell_df[["datum_obs", "obs_varde"]].copy()

#byter till tydliga kolumnnamn
observations_df.columns = ["Datum", "Obsvarde"]

#tar bort rader där datum eller observationsvärde saknas
observations_df = observations_df.dropna(subset=["Datum", "Obsvarde"]).copy()

#sorterar i tidsordning
observations_df = observations_df.sort_values("Datum").reset_index(drop=True)

#skapar en tabell för den visuellt matchade serien från kolumn C och D
matchad_df = visuell_df[["datum_matchad", "matchad_varde"]].copy()

#byter till tydliga kolumnnamn
matchad_df.columns = ["Datum", "Prediktionsvarde"]

#tar bort rader där datum eller matchat värde saknas
matchad_df = matchad_df.dropna(subset=["Datum", "Prediktionsvarde"]).copy()

#om det finns flera matchade värden samma datum beräknas ett dagligt medelvärde
matchad_df = (
    matchad_df
    .groupby("Datum", as_index=False)["Prediktionsvarde"]
    .mean()
)

#sorterar i tidsordning
matchad_df = matchad_df.sort_values("Datum").reset_index(drop=True)


#------------------------------------ Slår ihop tabellerna på gemensamma datum --------------------------------------------------------------

#anger hur nära i tid ett matchat värde får ligga observationen
#14 dagar används eftersom observationerna inte alltid har exakt samma datum som den matchade serien
tolerans_dagar = 14

#sorterar observationerna
observations_for_merge = observations_df.sort_values("Datum").copy()

#sorterar den visuellt matchade serien
matchad_for_merge = matchad_df.sort_values("Datum").copy()

#byter namn på datumkolumnen i den matchade serien
matchad_for_merge = matchad_for_merge.rename(columns={"Datum": "Datum_matchad"})

#kopplar varje observation till närmaste visuellt matchade värde inom vald tolerans
gemensamma_datapar = pd.merge_asof(
    observations_for_merge,
    matchad_for_merge,
    left_on="Datum",
    right_on="Datum_matchad",
    direction="nearest",
    tolerance=pd.Timedelta(days=tolerans_dagar)
)

#räknar hur många dagar det är mellan observation och matchat värde
gemensamma_datapar["dagar_till_matchad"] = (
    gemensamma_datapar["Datum"] - gemensamma_datapar["Datum_matchad"]
).abs().dt.days

#tar bort observationer som inte fick något matchat värde inom toleransen
gemensamma_datapar = gemensamma_datapar.dropna(
    subset=["Datum", "Obsvarde", "Prediktionsvarde"]
).copy()

#sorterar igen
gemensamma_datapar = gemensamma_datapar.sort_values("Datum").reset_index(drop=True)

print("\nAntal gemensamma datapar efter matchning med närmaste datum:", len(gemensamma_datapar))

print("\nSammanfattning av antal dagar mellan observationsdatum och matchat datum:")
print(gemensamma_datapar["dagar_till_matchad"].describe())

print("\nFörsta gemensamma dataparen:")
print(gemensamma_datapar.head(10))

#sorterar igen för säkerhets skull
gemensamma_datapar = gemensamma_datapar.sort_values("Datum").reset_index(drop=True)

#skriver ut antal gemensamma datapar
print("\nAntal gemensamma datapar:", len(gemensamma_datapar))

#skriver ut de första raderna
print("\nFörsta gemensamma dataparen:")
print(gemensamma_datapar.head(10))


#---------------------------------------------------- Skapar residualer --------------------------------------------------------------

#beräknar residualen som observation minus visuellt matchat värde
gemensamma_datapar["Residual"] = (
    gemensamma_datapar["Obsvarde"] - gemensamma_datapar["Prediktionsvarde"]
)

#skriver ut de första raderna så att vi ser att residualen blev rätt
print("\nFörsta raderna med residual:")
print(gemensamma_datapar.head(10))


#---------------------------------------------------- Utvärderingsmått --------------------------------------------------------------

#skapar en funktion som beräknar utvärderingsmått för den visuella metoden
def funktion_utvarderingsmatt_visuell(gemensamma_datapar: pd.DataFrame):

    #tar bort rader där observation eller matchat värde saknas
    df = gemensamma_datapar.dropna(subset=["Obsvarde", "Prediktionsvarde", "Residual"]).copy()

    #avbryter om inga datapunkter finns
    if len(df) == 0:
        return None

    #plockar ut residualerna
    residualer = df["Residual"]

    #antal datapunkter
    n = len(df)

    #---------------------------------------------------- Mean Absolute Error (MAE) -----------------------------------------------------

    #genomsnittlig absolut avvikelse mellan observation och matchat värde
    mae = np.mean(np.abs(residualer))

    #---------------------------------------------------- Root Mean Squared Error (RMSE) -----------------------------------------------------

    #medelvärde av kvadrerade residualer
    mse = np.mean(residualer ** 2)

    #rot ur mse för att få rmse
    rmse = np.sqrt(mse)

    #---------------------------------------------------- Mean Absolute Percentage Error (MAPE) -----------------------------------------------------

    #tar bort eventuella rader där observationsvärdet är 0 för att undvika division med 0
    df_mape = df[df["Obsvarde"] != 0].copy()

    #beräknar mape om sådana rader finns
    if len(df_mape) > 0:
        mape = np.mean(
            np.abs((df_mape["Obsvarde"] - df_mape["Prediktionsvarde"]) / df_mape["Obsvarde"])
        ) * 100
    else:
        mape = np.nan

    #---------------------------------------------------- Residualstandardavvikelse -----------------------------------------------------

    #standardavvikelse för residualerna
    residual_std = residualer.std(ddof=1)

    #---------------------------------------------------- Durbin-Watson -----------------------------------------------------

    #skillnaden mellan varje residual och föregående residual
    diff_residualer = np.diff(residualer)

    #skydd mot division med 0 om residualerna mot förmodan alla skulle vara exakt 0
    if np.sum(residualer ** 2) > 0:
        durbin_watson = np.sum(diff_residualer ** 2) / np.sum(residualer ** 2)
    else:
        durbin_watson = np.nan

    #---------------------------------------------------- Log-likelihood -----------------------------------------------------

    #variansen för residualerna
    sigma2 = np.var(residualer)

    #om residualvariansen är större än 0 kan log-likelihood beräknas
    if sigma2 > 0:
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    else:
        log_likelihood = np.nan

    #---------------------------------------------------- AIC och BIC -----------------------------------------------------

    #vi använder samma antal modellparametrar som i akvifärkoden för jämförbarhet
    k = 2

    #beräknar AIC och BIC om log-likelihood finns
    if pd.notna(log_likelihood):
        AIC = 2 * k - 2 * log_likelihood
        BIC = np.log(n) * k - 2 * log_likelihood
    else:
        AIC = np.nan
        BIC = np.nan

    #---------------------------------------------------- Returnerar alla mått -----------------------------------------------------

    return {
        "antal_gemensamma_datapunkter": n,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_procent": mape,
        "residual_standardavvikelse": residual_std,
        "durbin_watson": durbin_watson,
        "AIC": AIC,
        "BIC": BIC
    }


#---------------------------------------------------- Beräknar och skriver ut utvärderingsmått --------------------------------------------------------------

#beräknar alla mått för den visuella metoden
utvarderingsmatt_resultat = funktion_utvarderingsmatt_visuell(gemensamma_datapar)

#kontrollerar att måtten kunde beräknas
if utvarderingsmatt_resultat is None:
    print("Utvärderingsmått kunde inte beräknas.")
else:
    #gör om resultatet till en dataframe så att det blir lättare att läsa
    utvarderingsmatt_df = pd.DataFrame(
        list(utvarderingsmatt_resultat.items()),
        columns=["utvarderingsmatt", "varde"]
    )

    #skriver ut måtten
    print("\nUtvärderingsmått för visuella metoden:")
    print(utvarderingsmatt_df)


#---------------------------------------------------- Hjälpfunktion för datumaxel --------------------------------------------------------------

#skapar en funktion som gör datumaxeln
def funktion_formatera_datumaxel(ax, datum_min, datum_max):

    #sätter datumformatet till år-månad
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    #räknar ungefärligt antal månader i figuren
    n_months = max((datum_max - datum_min).days / 30, 1)

    #väljer lagom avstånd mellan datummarkeringarna
    interval = max(4, int(n_months / 8))

    #sätter datumintervall
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))

    #roterar datumetiketterna
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)


#---------------------------------------------------- Visualisering: huvudfigur --------------------------------------------------------------

#skapar en funktion som plottar observationer och visuellt matchad serie över tid
#huvudfiguren använder de två separata serierna, inte bara gemensamma datapar
def funktion_plotta_huvudfigur_visuell(
    observations_df: pd.DataFrame,
    matchad_df: pd.DataFrame,
    observationsror_namn: str
):

    #tar bort rader där observation saknas
    obs = observations_df.dropna(subset=["Datum", "Obsvarde"]).copy()

    #tar bort rader där matchat värde saknas
    matchad = matchad_df.dropna(subset=["Datum", "Prediktionsvarde"]).copy()

    #avbryter om någon av serierna saknar värden
    if len(obs) == 0 or len(matchad) == 0:
        print("Det finns inte tillräckligt med värden för att plotta huvudfiguren.")
        return None

    #sorterar båda serierna i tidsordning
    obs = obs.sort_values("Datum").reset_index(drop=True)
    matchad = matchad.sort_values("Datum").reset_index(drop=True)

    #tar fram observationsrörets observerade period
    obs_start = obs["Datum"].min()
    obs_slut = obs["Datum"].max()

    #begränsar den visuellt matchade serien till observationsrörets observerade period
    #detta gör att den röda linjen inte ritas från exempelvis 1960-talet
    matchad = matchad[
        (matchad["Datum"] >= obs_start) &
        (matchad["Datum"] <= obs_slut)
    ].copy()

    #avbryter om det inte finns några matchade värden inom observationsperioden
    if len(matchad) == 0:
        print("Det finns inga matchade värden inom observationsrörets observerade period.")
        return None

    #skapar figur 
    fig, ax = plt.subplots(figsize=(14, 6))

    #sätter rubrik 
    fig.suptitle(
        "Visuella metoden\n"
        f"Observationsrör: {observationsror_namn}",
        fontsize=14,
        fontweight="bold"
    )

    #ritar den visuellt matchade serien som svart linje
    ax.plot(
        matchad["Datum"],
        matchad["Prediktionsvarde"],
        linestyle="-",
        linewidth=1.4,
        color="red",
        label="Visuellt matchad serie"
    )

    #ritar observationerna som blå punkter
    ax.plot(
        obs["Datum"],
        obs["Obsvarde"],
        "o",
        color="navy",
        markersize=3,
        alpha=0.75,
        label=f"Observation ({observationsror_namn})"
    )

    #sätter axelrubriker
    ax.set_xlabel("Datum")
    ax.set_ylabel("Nivå (m ö.h.)")

    #lägger legend uppe till vänster
    ax.legend(loc="upper left", fontsize=8)

    #formaterar datumaxeln utifrån observationsrörets observerade period
    funktion_formatera_datumaxel(ax, obs_start, obs_slut)

    #sätter x-axeln så att figuren bara visar observationsperioden
    ax.set_xlim(obs_start, obs_slut)

    #lägger till diskretare rutnät
    ax.grid(True, alpha=0.3)

    #sätter rimlig y-axel utifrån både observationer och matchad serie
    alla_varden = pd.concat([obs["Obsvarde"], matchad["Prediktionsvarde"]]).dropna()

    if len(alla_varden) > 0:
        ymin = alla_varden.min() - 0.5
        ymax = alla_varden.max() + 0.5
        ax.set_ylim(ymin, ymax)

    #gör layouten snyggare
    plt.tight_layout()

    #returnerar figuren så att den kan sparas
    return fig


#---------------------------------------------------- Visualisering: residualer över tid --------------------------------------------------------------

#skapar en funktion som plottar residualerna över tid
def funktion_plotta_residualer_over_tid(
    gemensamma_datapar: pd.DataFrame,
    observationsror_namn: str
):

    #tar bort rader där residual saknas
    df = gemensamma_datapar.dropna(subset=["Datum", "Residual"]).copy()

    #avbryter om inga residualer finns
    if len(df) == 0:
        print("Inga residualer finns att plotta över tid.")
        return None

    #sorterar i tidsordning
    df = df.sort_values("Datum").reset_index(drop=True)

    #skapar figur och axel
    fig, ax = plt.subplots(figsize=(14, 6))

    #sätter rubrik
    fig.suptitle(
        "Residualer över tid — Visuella metoden\n"
        f"Observationsrör: {observationsror_namn}",
        fontsize=14,
        fontweight="bold"
    )

    #skapar en kopia av residualerna för själva figuren
    residualer_for_plot = df["Residual"].astype(float).copy()

    #räknar antal dagar mellan residualpunkterna
    datumskillnad = df["Datum"].diff().dt.days

    #bryter linjen om det är mer än 90 dagar mellan två punkter
    #detta påverkar bara hur figuren ritas, inte utvärderingsmåtten
    residualer_for_plot[datumskillnad > 90] = np.nan

    #ritar residualerna över tid
    ax.plot(
        df["Datum"],
        residualer_for_plot,
        "o-",
        color="steelblue",
        markersize=3,
        linewidth=1.2,
        alpha=0.8
    )

    #lägger till en horisontell linje vid 0
    ax.axhline(0, color="black", linewidth=0.8)

    #lägger till axeltexter
    ax.set_xlabel("Datum")
    ax.set_ylabel("Residual (m)")

    #formaterar datumaxeln
    funktion_formatera_datumaxel(ax, df["Datum"].min(), df["Datum"].max())

    #rutnät
    ax.grid(True, alpha=0.3)

    #layout
    plt.tight_layout()

    return fig


#---------------------------------------------------- Visualisering: ACF för residualer --------------------------------------------------------------

#skapar en funktion som beräknar och plottar ACF för residualerna
def funktion_plotta_acf_residualer(
    gemensamma_datapar: pd.DataFrame,
    observationsror_namn: str,
    max_lag=10
):

    #tar bort rader där residual saknas
    df = gemensamma_datapar.dropna(subset=["Residual"]).copy()

    #avbryter om för få residualer finns
    if len(df) < 3:
        print("För få residualer finns för att beräkna ACF.")
        return None

    #sorterar i tidsordning
    df = df.sort_values("Datum").reset_index(drop=True)

    #plockar ut residualerna
    residualer = df["Residual"]

    #antal möjliga laggar
    max_lag = min(max_lag, len(residualer) - 1)

    #beräknar ACF-värden manuellt
    laggar = list(range(max_lag + 1))
    acf_varden = [1.0]

    for lag in range(1, max_lag + 1):
        acf_varden.append(residualer.autocorr(lag=lag))

    #ungefärlig 95 %-gräns
    konfidensgrans = 1.96 / np.sqrt(len(residualer))

    #skapar figur och axel
    fig, ax = plt.subplots(figsize=(8, 5))

    #ritar acf som linjer från 0
    ax.vlines(
        laggar,
        0,
        acf_varden,
        colors="steelblue",
        linewidth=1.5
    )

    #ritar punkter
    ax.plot(
        laggar,
        acf_varden,
        "o",
        color="steelblue",
        markersize=5
    )

    #hjälplinjer
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(konfidensgrans, linestyle="--", color="steelblue", linewidth=0.9)
    ax.axhline(-konfidensgrans, linestyle="--", color="steelblue", linewidth=0.9)

    #rubriker och axlar
    ax.set_title(
        f"ACF för residualer — Visuella metoden ({observationsror_namn})",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autokorrelation")

    #rutnät
    ax.grid(True, alpha=0.3)

    #layout
    plt.tight_layout()

    #skriver också ut tabell
    acf_tabell = pd.DataFrame({
        "lag": laggar,
        "ACF": acf_varden
    })

    print("\nACF för residualer:")
    print(acf_tabell)

    return fig


#---------------------------------------------------- Visualisering: histogram över residualer --------------------------------------------------------------

#skapar en funktion som plottar histogram över residualerna
def funktion_plotta_histogram_residualer(
    gemensamma_datapar: pd.DataFrame,
    observationsror_namn: str
):

    #tar bort rader där residual saknas
    df = gemensamma_datapar.dropna(subset=["Residual"]).copy()

    #plockar ut residualerna
    residualer = df["Residual"].to_numpy(dtype=float)

    #avbryter om för få residualer finns
    if len(residualer) < 3:
        print("För få residualer finns för histogram.")
        return None

    #skapar figur och axel
    fig, ax = plt.subplots(figsize=(8, 5))

    #ritar histogram
    ax.hist(
        residualer,
        bins="auto",
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        label="Residualer"
    )

    #beräknar medelvärde och standardavvikelse
    mu = residualer.mean()
    sigma = residualer.std(ddof=1)

    #ritar normalfördelningskurva om standardavvikelsen är större än 0
    if sigma > 0:
        x = np.linspace(residualer.min(), residualer.max(), 200)
        ax.plot(
            x,
            sp_stats.norm.pdf(x, mu, sigma),
            color="red",
            linewidth=2,
            label=f"N({mu:.4f}, {sigma:.4f}²)"
        )

    #rubrik och axlar
    ax.set_title(
        f"Histogram av residualer — Visuella metoden ({observationsror_namn})",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Residual (m)")
    ax.set_ylabel("Densitet")

    #legend och rutnät
    ax.legend()
    ax.grid(True, alpha=0.3)

    #layout
    plt.tight_layout()

    return fig

#---------------------------------------------------- Kör visualiseringarna --------------------------------------------------------------

#ritar och sparar huvudfiguren
fig_huvud = funktion_plotta_huvudfigur_visuell(
    observations_df=observations_df,
    matchad_df=matchad_df,
    observationsror_namn=observationsror_namn
)

if fig_huvud is not None:
    fig_huvud.savefig(
        f"visuell_metod_huvudfigur_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: visuell_metod_huvudfigur_{filnamn_id}.png")


#ritar och sparar residualer över tid
fig_residualer = funktion_plotta_residualer_over_tid(
    gemensamma_datapar=gemensamma_datapar,
    observationsror_namn=observationsror_namn
)

if fig_residualer is not None:
    fig_residualer.savefig(
        f"visuell_metod_residualer_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: visuell_metod_residualer_{filnamn_id}.png")


#ritar och sparar ACF för residualerna
fig_acf = funktion_plotta_acf_residualer(
    gemensamma_datapar=gemensamma_datapar,
    observationsror_namn=observationsror_namn,
    max_lag=10
)

if fig_acf is not None:
    fig_acf.savefig(
        f"visuell_metod_acf_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: visuell_metod_acf_{filnamn_id}.png")


#ritar och sparar histogram över residualerna
fig_hist = funktion_plotta_histogram_residualer(
    gemensamma_datapar=gemensamma_datapar,
    observationsror_namn=observationsror_namn
)

if fig_hist is not None:
    fig_hist.savefig(
        f"visuell_metod_histogram_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: visuell_metod_histogram_{filnamn_id}.png")


#visar alla figurer när koden körs
plt.show()
