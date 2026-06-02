# ----------------------------------- Enkel regressionsmodell - Kandidatuppsats -----------------------------------

# ---------------------------------------------------- Importerar paket --------------------------------------------------------------

import math
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import truststore
from scipy.stats import norm, t
from sgu_client import SGUClient

truststore.inject_into_ssl()


# ---------------------------------------------------- Kontrollerar miljö -----------------------------------------------------

print("Python:", sys.version)
print("pandas:", pd.__version__)

plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.titleweight"] = "bold"


# ---------------------------------------------------- Interaktiv inläsning av observationsrör -----------------------------------------------------

def funktion_las_in_observationsfil():
    """Läser in sökvägen till observationsrörets csv-fil."""

    while True:
        sokvag_text = input(
            "\nAnge full sökväg till observationsrörets csv-fil "
            "(utan citattecken eller parenteser): "
        ).strip()

        sokvag_text = sokvag_text.strip('"').strip("'")
        data_path = Path(sokvag_text)

        if not data_path.exists():
            print("Filen kunde inte hittas. Kontrollera sökvägen och försök igen.")
            continue

        if not data_path.is_file():
            print("Sökvägen pekar inte på en fil. Försök igen.")
            continue

        return data_path


def funktion_hamta_observationsror_id(data_path: Path):
    """Hämtar observationsrörets id från filnamnet eller från användaren."""

    forslag_id = data_path.stem.strip()

    print(f"\nFöreslaget observationsrör-id baserat på filnamnet: {forslag_id}")

    anvandarens_id = input(
        "Tryck Enter för att använda detta id, eller skriv in ett annat observationsrör-id: "
    ).strip()

    if anvandarens_id == "":
        return forslag_id

    return anvandarens_id


def funktion_las_in_observationsdata(data_path: Path):
    """Läser in observationsrörets csv-fil och rensar datum och nivåvärden."""

    observations_df = pd.read_csv(
        data_path,
        sep=";",
        decimal=",",
        header=None,
        names=["datum", "gw_moh"],
    )

    observations_df["datum"] = pd.to_datetime(
        observations_df["datum"],
        errors="coerce"
    )

    observations_df["gw_moh"] = pd.to_numeric(
        observations_df["gw_moh"],
        errors="coerce"
    )

    observations_df = observations_df.dropna(
        subset=["datum", "gw_moh"]
    ).copy()

    observations_df = observations_df.sort_values("datum").reset_index(drop=True)

    if len(observations_df) < 3:
        raise ValueError("För få giltiga observationer finns i filen efter rensning.")

    return observations_df


def funktion_visa_oversikt_for_observationsror(
    observations_df: pd.DataFrame,
    observationsror_id: str
):
    """Visar enkel datasammanfattning och översiktsfigurer."""

    observations_df["skillnad_dagar"] = observations_df["datum"].diff().dt.days

    print("\nSammanfattning av antal dagar mellan mätningar:")
    print(observations_df["skillnad_dagar"].describe())

    observations_start = observations_df["datum"].min()
    observations_slut = observations_df["datum"].max()
    antal_ar = (observations_slut - observations_start).days / 365.25

    print("\nSammanfattning av observationsserien:")
    print("Första datum:", observations_start.date())
    print("Sista datum:", observations_slut.date())
    print("Antal observationer:", len(observations_df))
    print("Ungefärlig serielängd i år:", round(antal_ar, 2))

    plt.figure()
    plt.plot(
        observations_df["datum"],
        observations_df["gw_moh"],
        marker="o",
        linestyle="-"
    )
    plt.title(f"Observationsrör {observationsror_id} (m.ö.h)")
    plt.xlabel("Datum")
    plt.ylabel("Grundvattennivå (m.ö.h)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(observations_df["skillnad_dagar"].dropna(), bins=20)
    plt.xlabel("Dagar mellan mätningar")
    plt.ylabel("Antal")
    plt.title("Histogram över mätintervall")
    plt.grid(True)
    plt.show()


def funktion_hamta_aggregeringsperiod():
    """Låter användaren välja aggregeringsperiod för tidsblock."""

    while True:
        anvandarens_period = input(
            "\nAnge aggregeringsperiod utifrån histogrammet\n"
            "samt sammanfattningen av antal dagar mellan mätningar ovan.\n"
            "\nExempel på godkänt format: 7D, 14D, 30D.\n"
            "Det står för 7 dagar, 14 dagar respektive 30 dagar.\n"
            "\nSkriv vald aggregeringsperiod här: "
        ).strip()

        if anvandarens_period == "":
            print("Du måste ange en aggregeringsperiod.")
            continue

        try:
            pd.to_timedelta(anvandarens_period)
            return anvandarens_period
        except Exception:
            print("Ogiltigt format. Använd till exempel 7D, 14D eller 30D.")


# ---------------------------------------------------- Startar analysen -----------------------------------------------------

data_path = funktion_las_in_observationsfil()
observationsror_id = funktion_hamta_observationsror_id(data_path)

# Ändra manuellt till "Rör A", "Rör B" eller "Rör C"
observationsror_namn = "Rör A"

filnamn_id = (
    observationsror_namn
    .lower()
    .replace(" ", "_")
    .replace("ö", "o")
    .replace("å", "a")
    .replace("ä", "a")
)

observations_df = funktion_las_in_observationsdata(data_path)

funktion_visa_oversikt_for_observationsror(
    observations_df=observations_df,
    observationsror_id=observationsror_id
)

aggregeringsperiod = funktion_hamta_aggregeringsperiod()
konfidensniva = 0.95

print("\nValda parametrar:")
print("Observationsrör:", observationsror_id)
print("Aggregeringsperiod:", aggregeringsperiod)
print("Konfidensnivå:", konfidensniva)

observations_tidsserie = observations_df.set_index("datum")["gw_moh"].sort_index()
observations_tidsserie.name = f"{observationsror_id}_moh"


# ---------------------------------------------------- Observerad period -----------------------------------------------------

tidsserie_start = observations_tidsserie.index.min().normalize()
tidsserie_slut = observations_tidsserie.index.max().normalize()

tidsserie_min = tidsserie_start.strftime("%Y-%m-%d")
tidsserie_max = tidsserie_slut.strftime("%Y-%m-%d")

print("\nPeriod som används i analysen:")
print("Första datum i observationsserien:", tidsserie_min)
print("Sista datum i observationsserien:", tidsserie_max)
print("Modellen anpassas på:", "hela observerade perioden")


# ---------------------------------------------------- Koordinater och geografiskt urval -----------------------------------------------------

def funktion_hamta_observation_longitud():
    """Läser in observationsrörets longitud."""

    while True:
        anvandarens_longitud = input(
            "\nAnge observationsrörets longitud i decimalgrader.\n"
            "Exempel: 18.1083\n"
            "\nSkriv longitud här: "
        ).strip()

        anvandarens_longitud = anvandarens_longitud.replace(",", ".")

        try:
            observation_longitud = float(anvandarens_longitud)

            if observation_longitud < -180 or observation_longitud > 180:
                print("Longitud måste ligga mellan -180 och 180.")
                continue

            return observation_longitud

        except Exception:
            print("Ogiltigt värde. Ange longitud som ett tal, till exempel 18.1083.")


def funktion_hamta_observation_latitud():
    """Läser in observationsrörets latitud."""

    while True:
        anvandarens_latitud = input(
            "\nAnge observationsrörets latitud i decimalgrader.\n"
            "Exempel: 59.3108\n"
            "\nSkriv latitud här: "
        ).strip()

        anvandarens_latitud = anvandarens_latitud.replace(",", ".")

        try:
            observation_latitud = float(anvandarens_latitud)

            if observation_latitud < -90 or observation_latitud > 90:
                print("Latitud måste ligga mellan -90 och 90.")
                continue

            return observation_latitud

        except Exception:
            print("Ogiltigt värde. Ange latitud som ett tal, till exempel 59.3108.")


observation_longitud = funktion_hamta_observation_longitud()
observation_latitud = funktion_hamta_observation_latitud()

# Fast sökradie för potentiella referensrör.
sokradie_km = 50

# Omvandlar sökradien från kilometer till grader.
grad_lat = sokradie_km / 111.32
grad_lon = sokradie_km / (
    111.32 * math.cos(math.radians(observation_latitud))
)

bbox = [
    observation_longitud - grad_lon,
    observation_latitud - grad_lat,
    observation_longitud + grad_lon,
    observation_latitud + grad_lat
]


# ---------------------------------------------------- Hämtar potentiella referensrör från SGU -----------------------------------------------------

with SGUClient() as client:
    referens_stationer = client.levels.observed.get_stations(
        bbox=bbox,
        limit=500
    )

ref_station_id = []

for st in referens_stationer.features:
    station_id = getattr(st.properties, "station_id", None)
    station_name = getattr(st.properties, "station_name", None)

    if station_id is None:
        station_id = station_name

    ref_station_id.append({
        "referens_id": station_id,
        "referens_namn": station_name
    })

sgu_stationer_df = pd.DataFrame(ref_station_id)

print("Antal referensstationer inom sökområdet:", len(sgu_stationer_df))


# ---------------------------------------------------- Tidsaggregering -----------------------------------------------------

def funktion_till_tidsblock(
    tidsserie: pd.Series,
    aggregeringsperiod: str,
    origin: pd.Timestamp
) -> pd.Series:
    """Aggregerar en tidsserie till gemensamma tidsblock."""

    period = pd.to_timedelta(aggregeringsperiod)

    tidsserie = tidsserie.dropna().sort_index()

    if getattr(tidsserie.index, "tz", None) is not None:
        tidsserie = tidsserie.copy()
        tidsserie.index = tidsserie.index.tz_convert(None)

    df = tidsserie.to_frame("varde")

    # Tilldelar varje mätning det tidsblock vars mittpunkt ligger närmast.
    df["tidsblock"] = np.rint((df.index - origin) / period).astype("int64")

    # Om flera värden finns i samma tidsblock används medelvärdet.
    medel_per_tidsblock = df.groupby("tidsblock")["varde"].mean()

    datum_for_tidsblock = origin + (
        medel_per_tidsblock.index.to_numpy() * period
    )

    tidsblock_tidsserie = pd.Series(
        medel_per_tidsblock.to_numpy(),
        index=datum_for_tidsblock,
        name=tidsserie.name
    )

    return tidsblock_tidsserie.sort_index()


observations_tidsserie_aggr = funktion_till_tidsblock(
    observations_tidsserie,
    aggregeringsperiod=aggregeringsperiod,
    origin=tidsserie_start
)

print("Antal tidsblock med observationer:", len(observations_tidsserie_aggr))
print(observations_tidsserie_aggr.head())

observations_tidsblock_i_period = observations_tidsserie_aggr.loc[
    tidsserie_min:tidsserie_max
].dropna()

antal_observationstidsblock_i_period = len(observations_tidsblock_i_period)

print(
    "Antal observationstidsblock i hela perioden:",
    antal_observationstidsblock_i_period
)


# ---------------------------------------------------- Linjär regression -----------------------------------------------------

def funktion_regressionsmatt(matchade_datapar: pd.DataFrame):
    """Beräknar enkel linjär regression och centrala modellmått."""

    matchade_datapar = matchade_datapar.dropna(
        subset=["observationsvarde", "referensvarde"]
    ).copy()

    if len(matchade_datapar) < 3:
        return None

    x = matchade_datapar["referensvarde"].to_numpy(dtype=float)
    y = matchade_datapar["observationsvarde"].to_numpy(dtype=float)

    n = len(matchade_datapar)

    x_medel = x.mean()
    y_medel = y.mean()

    sxx = np.sum((x - x_medel) ** 2)
    sxy = np.sum((x - x_medel) * (y - y_medel))

    if sxx <= 0:
        return None

    beta_1 = sxy / sxx
    beta_0 = y_medel - beta_1 * x_medel

    y_hat = beta_0 + beta_1 * x
    residualer = y - y_hat

    sse = np.sum(residualer ** 2)
    sst = np.sum((y - y_medel) ** 2)

    if sst > 0:
        r_kvadrat = 1 - (sse / sst)
    else:
        r_kvadrat = np.nan

    residualstandardfel = np.sqrt(sse / (n - 2))

    return {
        "n": n,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "residualstandardfel": residualstandardfel,
        "r_kvadrat": r_kvadrat,
        "referens_min": matchade_datapar["referensvarde"].min(),
        "referens_max": matchade_datapar["referensvarde"].max()
    }


# ---------------------------------------------------- Osäkerhetsmått -----------------------------------------------------

def funktion_osakerhetsmatt(
    antal_datapar: int,
    residualstandardfel: float,
    konfidensniva: float
):
    """Beräknar osäkerhetsmåttet som används för rangordning av referensrör."""

    if antal_datapar < 2:
        return None

    sannolikhet_t = (1 - konfidensniva) / 2
    students_t_varde = -t.ppf(sannolikhet_t, df=antal_datapar - 1)

    osakerhetsmatt = (
        students_t_varde
        * residualstandardfel
        * np.sqrt(1 + (1 / antal_datapar))
    )

    return {
        "students_t_varde": students_t_varde,
        "osakerhetsmatt": osakerhetsmatt
    }


# ---------------------------------------------------- Resultat för matchade datapar -----------------------------------------------------

def funktion_berakna_resultat_for_matchade_datapar(
    matchade_datapar: pd.DataFrame,
    beta_0: float,
    beta_1: float,
    osakerhetsmatt: float
):
    """Beräknar skattade värden, intervall och residualer för matchade datapar."""

    df = matchade_datapar.dropna(
        subset=["observationsvarde", "referensvarde"]
    ).copy()

    if len(df) == 0:
        return None

    df["skattat_varde"] = beta_0 + beta_1 * df["referensvarde"]

    df["nedre_intervallgrans"] = df["skattat_varde"] - osakerhetsmatt
    df["ovre_intervallgrans"] = df["skattat_varde"] + osakerhetsmatt

    df["residual"] = df["observationsvarde"] - df["skattat_varde"]

    df["inom_osakerhetsintervall"] = (
        (df["observationsvarde"] >= df["nedre_intervallgrans"]) &
        (df["observationsvarde"] <= df["ovre_intervallgrans"])
    )

    return df


# ---------------------------------------------------- Skattad serie -----------------------------------------------------

def funktion_skapa_skattad_serie(
    referens_tidsserie: pd.Series,
    beta_0: float,
    beta_1: float,
    osakerhetsmatt: float,
    tidsserie_min: str,
    tidsserie_max: str
):
    """Skapar skattad serie för observationsröret utifrån valt referensrör."""

    df = referens_tidsserie.loc[tidsserie_min:tidsserie_max].dropna().to_frame(
        "referensvarde"
    ).copy()

    if len(df) == 0:
        return None

    df["skattat_varde"] = beta_0 + beta_1 * df["referensvarde"]

    df["nedre_intervallgrans"] = df["skattat_varde"] - osakerhetsmatt
    df["ovre_intervallgrans"] = df["skattat_varde"] + osakerhetsmatt

    return df


# ---------------------------------------------------- Urvalskrav för referensrör -----------------------------------------------------

min_datapar = 15
min_varaktighet_dagar = 365
min_unika_manader = 7

min_tackningsgrad_referensserie = 0.85
max_startforskjutning_referensserie_dagar = 365
max_slutforskjutning_referensserie_dagar = 365

referens_tidsserier_aggr = {}
referens_tidsserier_raw = {}
resultat_referensror = []
matchade_datapar_per_referensror = {}

observationsperiod_dagar = max((tidsserie_slut - tidsserie_start).days, 1)


# ---------------------------------------------------- Matchar observationsrör och referensrör -----------------------------------------------------

with SGUClient() as client:

    for station_id in sgu_stationer_df["referens_id"].dropna().astype(str).unique():

        if str(station_id) == str(observationsror_id):
            continue

        try:
            measurements = client.levels.observed.get_measurements_by_name(
                station_id=station_id,
                limit=200000
            )

            referens_tidsserie = measurements.to_series()

            if getattr(referens_tidsserie.index, "tz", None) is not None:
                referens_tidsserie.index = referens_tidsserie.index.tz_convert(None)

            referens_tidsserie_aggr = funktion_till_tidsblock(
                referens_tidsserie,
                aggregeringsperiod=aggregeringsperiod,
                origin=tidsserie_start
            )

            referens_tidsserie_aggr.name = station_id

            referens_tidsserie_i_period = referens_tidsserie_aggr.loc[
                tidsserie_min:tidsserie_max
            ].dropna()

            if len(referens_tidsserie_i_period) < 2:
                continue

            referensserie_start = referens_tidsserie_i_period.index.min().normalize()
            referensserie_slut = referens_tidsserie_i_period.index.max().normalize()

            referensserie_varaktighet_dagar = (
                referensserie_slut - referensserie_start
            ).days

            tackningsgrad_referensserie = (
                referensserie_varaktighet_dagar / observationsperiod_dagar
            )

            startforskjutning_referensserie_dagar = (
                referensserie_start - tidsserie_start
            ).days

            slutforskjutning_referensserie_dagar = (
                tidsserie_slut - referensserie_slut
            ).days

            if (
                tackningsgrad_referensserie < min_tackningsgrad_referensserie
                or startforskjutning_referensserie_dagar > max_startforskjutning_referensserie_dagar
                or slutforskjutning_referensserie_dagar > max_slutforskjutning_referensserie_dagar
            ):
                continue

            matchade_datapar = pd.concat(
                [observations_tidsserie_aggr, referens_tidsserie_aggr],
                axis=1,
                join="inner"
            ).dropna()

            matchade_datapar.columns = ["observationsvarde", "referensvarde"]

            matchade_datapar["tidsblock"] = np.rint(
                (matchade_datapar.index - tidsserie_start)
                / pd.to_timedelta(aggregeringsperiod)
            ).astype("int64")

            matchade_datapar_period = matchade_datapar.loc[
                tidsserie_min:tidsserie_max
            ].copy()

            antal_datapar = len(matchade_datapar_period)

            if antal_datapar < min_datapar:
                continue

            datapar_start = matchade_datapar_period.index.min().normalize()
            datapar_slut = matchade_datapar_period.index.max().normalize()

            datapar_varaktighet_dagar = (datapar_slut - datapar_start).days

            tackningsgrad_datapar = datapar_varaktighet_dagar / observationsperiod_dagar

            startforskjutning_datapar_dagar = (
                datapar_start - tidsserie_start
            ).days

            slutforskjutning_datapar_dagar = (
                tidsserie_slut - datapar_slut
            ).days

            antal_unika_manader = (
                matchade_datapar_period.index.to_period("M").nunique()
            )

            if antal_observationstidsblock_i_period > 0:
                andel_observationstidsblock_med_datapar = (
                    antal_datapar / antal_observationstidsblock_i_period
                )
            else:
                andel_observationstidsblock_med_datapar = 0

            if (
                datapar_varaktighet_dagar < min_varaktighet_dagar
                or antal_unika_manader < min_unika_manader
            ):
                continue

            regressionsmatt = funktion_regressionsmatt(matchade_datapar_period)

            if regressionsmatt is None:
                continue

            osakerhetsresultat = funktion_osakerhetsmatt(
                antal_datapar=regressionsmatt["n"],
                residualstandardfel=regressionsmatt["residualstandardfel"],
                konfidensniva=konfidensniva
            )

            if osakerhetsresultat is None:
                continue

            students_t_varde = osakerhetsresultat["students_t_varde"]
            osakerhetsmatt = osakerhetsresultat["osakerhetsmatt"]

            matchande_rad = (
                sgu_stationer_df["referens_id"].astype(str) == str(station_id)
            )

            if matchande_rad.any():
                referens_namn = sgu_stationer_df.loc[
                    matchande_rad,
                    "referens_namn"
                ].iloc[0]
            else:
                referens_namn = station_id

            resultat_referensror.append({
                "observationsror_id": observationsror_id,
                "referens_id": station_id,
                "referens_namn": referens_namn,
                "osakerhetsmatt": osakerhetsmatt,
                "students_t_varde": students_t_varde,
                "n": regressionsmatt["n"],
                "beta_0": regressionsmatt["beta_0"],
                "beta_1": regressionsmatt["beta_1"],
                "residualstandardfel": regressionsmatt["residualstandardfel"],
                "r_kvadrat": regressionsmatt["r_kvadrat"],
                "referens_min": regressionsmatt["referens_min"],
                "referens_max": regressionsmatt["referens_max"],
                "datapar_varaktighet_dagar": datapar_varaktighet_dagar,
                "tackningsgrad_datapar": tackningsgrad_datapar,
                "startforskjutning_datapar_dagar": startforskjutning_datapar_dagar,
                "slutforskjutning_datapar_dagar": slutforskjutning_datapar_dagar,
                "antal_unika_manader": antal_unika_manader,
                "andel_observationstidsblock_med_datapar": andel_observationstidsblock_med_datapar,
                "tackningsgrad_referensserie": tackningsgrad_referensserie,
                "startforskjutning_referensserie_dagar": startforskjutning_referensserie_dagar,
                "slutforskjutning_referensserie_dagar": slutforskjutning_referensserie_dagar,
                "sista_matchade_datum": matchade_datapar_period.index.max(),
                "aggregeringsperiod": aggregeringsperiod,
                "konfidensniva": konfidensniva,
                "avstand_m": np.nan
            })

            matchade_datapar_per_referensror[station_id] = matchade_datapar.copy()
            referens_tidsserier_aggr[station_id] = referens_tidsserie_aggr
            referens_tidsserier_raw[station_id] = referens_tidsserie.copy()

        except Exception:
            continue


# ---------------------------------------------------- Rangordnar referensrör -----------------------------------------------------

print("Antal referensrör efter filtrering:", len(referens_tidsserier_aggr))

if len(resultat_referensror) == 0:
    raise ValueError("Inga referensrör kunde rangordnas.")

resultat_referensror_df = pd.DataFrame(resultat_referensror)

resultat_referensror_df = resultat_referensror_df.sort_values(
    by="osakerhetsmatt",
    ascending=True
).reset_index(drop=True)

topp_25_referensror_df = resultat_referensror_df[
    [
        "referens_id",
        "referens_namn",
        "osakerhetsmatt",
        "r_kvadrat",
        "n",
        "antal_unika_manader",
        "tackningsgrad_referensserie"
    ]
].head(25).copy()

topp_25_referensror_df.index = range(1, len(topp_25_referensror_df) + 1)

print("\nTop 25 referensrör enligt osäkerhetsmått:")
print(topp_25_referensror_df.to_string())


# ---------------------------------------------------- Manuellt val av referensrör -----------------------------------------------------

valt_referensror_id = input(
    "\nAnge referens_id för det referensrör som ska användas efter hydrogeologisk kontroll: "
).strip()

if str(valt_referensror_id) not in resultat_referensror_df["referens_id"].astype(str).values:
    raise ValueError(
        "Det manuellt valda referensröret finns inte bland de godkända kandidaterna."
    )

vald_referensrad = resultat_referensror_df.loc[
    resultat_referensror_df["referens_id"].astype(str) == str(valt_referensror_id)
].iloc[0]

if pd.notna(vald_referensrad["referens_namn"]):
    valt_referensror_namn = str(vald_referensrad["referens_namn"])
else:
    valt_referensror_namn = str(valt_referensror_id)

valda_matchade_datapar = matchade_datapar_per_referensror[
    str(valt_referensror_id)
].copy()

valda_periodens_datapar = valda_matchade_datapar.loc[
    tidsserie_min:tidsserie_max
].copy()


# ---------------------------------------------------- Resultat med valt referensrör -----------------------------------------------------

periodens_resultat_df = funktion_berakna_resultat_for_matchade_datapar(
    matchade_datapar=valda_periodens_datapar,
    beta_0=vald_referensrad["beta_0"],
    beta_1=vald_referensrad["beta_1"],
    osakerhetsmatt=vald_referensrad["osakerhetsmatt"]
)

if periodens_resultat_df is None:
    raise ValueError("Resultatet kunde inte beräknas för det valda referensröret.")

vald_referens_tidsserie_raw = referens_tidsserier_raw[
    str(valt_referensror_id)
].copy()

skattad_tidsserie_df = funktion_skapa_skattad_serie(
    referens_tidsserie=vald_referens_tidsserie_raw,
    beta_0=vald_referensrad["beta_0"],
    beta_1=vald_referensrad["beta_1"],
    osakerhetsmatt=vald_referensrad["osakerhetsmatt"],
    tidsserie_min=tidsserie_min,
    tidsserie_max=tidsserie_max
)

if skattad_tidsserie_df is None:
    raise ValueError("Den skattade tidsserien kunde inte skapas.")

print("\nSkattad serie:")
print(
    "Antal värden i observationsröret:",
    len(observations_tidsserie.loc[tidsserie_min:tidsserie_max].dropna())
)
print(
    "Antal värden i skattad serie:",
    len(skattad_tidsserie_df)
)

print("\nManuellt valt referensrör:")
print(
    vald_referensrad[
        [
            "referens_id",
            "referens_namn",
            "osakerhetsmatt",
            "beta_0",
            "beta_1",
            "residualstandardfel",
            "r_kvadrat",
            "n"
        ]
    ]
)


# ---------------------------------------------------- Utvärderingsmått -----------------------------------------------------

def funktion_utvarderingsmatt(resultat_df: pd.DataFrame):
    """Beräknar MSE, RMSE och Durbin-Watson för residualerna."""

    df = resultat_df.dropna(
        subset=["observationsvarde", "skattat_varde"]
    ).copy()

    if len(df) == 0:
        return None

    residualer = df["observationsvarde"] - df["skattat_varde"]

    mse = np.mean(residualer ** 2)
    rmse = np.sqrt(mse)

    if np.sum(residualer ** 2) > 0:
        durbin_watson = np.sum(np.diff(residualer) ** 2) / np.sum(residualer ** 2)
    else:
        durbin_watson = np.nan

    return {
        "antal_datapunkter": len(residualer),
        "MSE": mse,
        "RMSE": rmse,
        "Durbin_Watson": durbin_watson
    }


utvarderingsmatt_resultat = funktion_utvarderingsmatt(periodens_resultat_df)

if utvarderingsmatt_resultat is None:
    print("Utvärderingsmått kunde inte beräknas.")
else:
    utvarderingsmatt_df = pd.DataFrame(
        list(utvarderingsmatt_resultat.items()),
        columns=["utvarderingsmatt", "varde"]
    )

    print("\nUtvärderingsmått för hela den observerade perioden:")
    print(utvarderingsmatt_df)


# ---------------------------------------------------- Hjälpfunktion för datumaxel -----------------------------------------------------

def funktion_formatera_datumaxel(ax, datum_min, datum_max):
    """Formaterar datumaxeln för tidsseriefigurer."""

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    n_months = max((datum_max - datum_min).days / 30, 1)
    interval = max(4, int(n_months / 8))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)


# ---------------------------------------------------- Huvudfigur -----------------------------------------------------

def funktion_plotta_skattad_tidsserie(
    skattad_tidsserie_df: pd.DataFrame,
    observations_tidsserie: pd.Series,
    referens_namn: str,
    observationsror_namn: str,
    visa_osakerhetsintervall: bool = True
):
    """Plottar skattad serie, observationsvärden och osäkerhetsintervall."""

    if visa_osakerhetsintervall:
        df = skattad_tidsserie_df.dropna(
            subset=[
                "skattat_varde",
                "nedre_intervallgrans",
                "ovre_intervallgrans"
            ]
        ).copy()
    else:
        df = skattad_tidsserie_df.dropna(
            subset=["skattat_varde"]
        ).copy()

    if len(df) == 0:
        print("Det finns inga värden att plotta för den skattade tidsserien.")
        return None

    df = df.sort_index()

    obs = observations_tidsserie.loc[df.index.min():df.index.max()].dropna().copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    fig.suptitle(
        "Enkel regressionsmodell\n"
        f"Observationsrör: {observationsror_namn}",
        fontsize=14,
        fontweight="bold"
    )

    if visa_osakerhetsintervall:
        ax.fill_between(
            df.index,
            df["nedre_intervallgrans"],
            df["ovre_intervallgrans"],
            color="steelblue",
            alpha=0.20,
            label="95% osäkerhetsintervall"
        )

    ax.plot(
        df.index,
        df["skattat_varde"],
        linestyle="-",
        linewidth=1.4,
        color="crimson",
        label=f"Skattad serie ({referens_namn})"
    )

    ax.plot(
        obs.index,
        obs.values,
        "o",
        color="navy",
        markersize=3,
        alpha=0.75,
        label=f"Observation ({observationsror_namn})"
    )

    ax.set_xlabel("Datum")
    ax.set_ylabel("Nivå (m ö.h.)")
    ax.legend(loc="upper left", fontsize=8)

    funktion_formatera_datumaxel(ax, df.index.min(), df.index.max())

    ax.grid(True, alpha=0.3)

    alla_varden = pd.concat([
        pd.Series(df["skattat_varde"].values),
        pd.Series(obs.values)
    ]).dropna()

    if len(alla_varden) > 0:
        ymin = alla_varden.min() - 0.5
        ymax = alla_varden.max() + 0.5
        ax.set_ylim(ymin, ymax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


# ---------------------------------------------------- Histogram över residualer -----------------------------------------------------

def funktion_plotta_residualhistogram(
    resultat_df: pd.DataFrame,
    observationsror_namn: str
):
    """Plottar histogram över residualerna."""

    df = resultat_df.dropna(subset=["residual"]).copy()
    residualer = df["residual"].to_numpy(dtype=float)

    if len(residualer) < 3:
        print("Det finns för få residualer att plotta histogram.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        residualer,
        bins="auto",
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        label="Residualer"
    )

    mu = residualer.mean()
    sigma = residualer.std(ddof=1)

    if sigma > 0:
        x = np.linspace(residualer.min(), residualer.max(), 200)
        ax.plot(
            x,
            norm.pdf(x, mu, sigma),
            color="red",
            linewidth=2,
            label=f"N({mu:.4f}, {sigma:.4f}²)"
        )

    ax.set_title(
        f"Histogram av residualer — Enkel regressionsmodell ({observationsror_namn})",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Residual (m)")
    ax.set_ylabel("Densitet")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


# ---------------------------------------------------- Residualer över tid -----------------------------------------------------

def funktion_plotta_residualer_over_tid(
    resultat_df: pd.DataFrame,
    observationsror_namn: str
):
    """Plottar residualerna över tid."""

    df = resultat_df.dropna(subset=["residual"]).copy()

    if len(df) == 0:
        print("Det finns inga residualer att plotta.")
        return None

    df = df.sort_index()

    fig, ax = plt.subplots(figsize=(14, 6))

    fig.suptitle(
        "Residualer över tid — Enkel regressionsmodell\n"
        f"Observationsrör: {observationsror_namn}",
        fontsize=14,
        fontweight="bold"
    )

    ax.plot(
        df.index,
        df["residual"],
        "o-",
        color="steelblue",
        markersize=3,
        linewidth=1.2,
        alpha=0.8
    )

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Datum")
    ax.set_ylabel("Residual (m)")

    funktion_formatera_datumaxel(ax, df.index.min(), df.index.max())

    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


# ---------------------------------------------------- SAC för residualer -----------------------------------------------------

def funktion_plotta_sac_residualer(
    resultat_df: pd.DataFrame,
    observationsror_namn: str,
    max_lag: int = 10
):
    """Plottar seriell autokorrelation, SAC, för residualerna."""

    df = resultat_df.dropna(subset=["residual"]).copy()

    if len(df) < 3:
        print("För få residualer för att plotta SAC.")
        return None

    residualer = df["residual"].to_numpy(dtype=float)
    residualer_cent = residualer - residualer.mean()

    max_lag = min(max_lag, len(residualer_cent) - 1)

    if max_lag < 1:
        print("För få residualer för att plotta SAC.")
        return None

    namnare = np.sum(residualer_cent ** 2)

    if namnare == 0:
        print("Residualerna har ingen variation, SAC kan inte beräknas.")
        return None

    lag_lista = np.arange(0, max_lag + 1)
    sac_varden = [1.0]

    for lag in range(1, max_lag + 1):
        taljare = np.sum(residualer_cent[lag:] * residualer_cent[:-lag])
        sac_lag = taljare / namnare
        sac_varden.append(sac_lag)

    konfidensgrans = 1.96 / np.sqrt(len(residualer_cent))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.vlines(
        lag_lista,
        0,
        sac_varden,
        colors="steelblue",
        linewidth=1.5
    )

    ax.plot(
        lag_lista,
        sac_varden,
        "o",
        color="steelblue",
        markersize=5
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(konfidensgrans, linestyle="--", color="steelblue", linewidth=0.9)
    ax.axhline(-konfidensgrans, linestyle="--", color="steelblue", linewidth=0.9)

    ax.set_title(
        f"SAC för residualer — Enkel regressionsmodell ({observationsror_namn})",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Lag")
    ax.set_ylabel("Seriell autokorrelation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


# ---------------------------------------------------- Kör visualiseringar -----------------------------------------------------

visa_osakerhetsintervall = True

fig_huvud = funktion_plotta_skattad_tidsserie(
    skattad_tidsserie_df=skattad_tidsserie_df,
    observations_tidsserie=observations_tidsserie,
    referens_namn=valt_referensror_namn,
    observationsror_namn=observationsror_namn,
    visa_osakerhetsintervall=visa_osakerhetsintervall
)

if fig_huvud is not None:
    fig_huvud.savefig(
        f"enkel_regression_huvudfigur_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: enkel_regression_huvudfigur_{filnamn_id}.png")

fig_hist = funktion_plotta_residualhistogram(
    resultat_df=periodens_resultat_df,
    observationsror_namn=observationsror_namn
)

if fig_hist is not None:
    fig_hist.savefig(
        f"enkel_regression_histogram_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: enkel_regression_histogram_{filnamn_id}.png")

fig_residualer = funktion_plotta_residualer_over_tid(
    resultat_df=periodens_resultat_df,
    observationsror_namn=observationsror_namn
)

if fig_residualer is not None:
    fig_residualer.savefig(
        f"enkel_regression_residualer_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: enkel_regression_residualer_{filnamn_id}.png")

fig_sac = funktion_plotta_sac_residualer(
    resultat_df=periodens_resultat_df,
    observationsror_namn=observationsror_namn,
    max_lag=10
)

if fig_sac is not None:
    fig_sac.savefig(
        f"enkel_regression_sac_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: enkel_regression_sac_{filnamn_id}.png")

plt.show()
