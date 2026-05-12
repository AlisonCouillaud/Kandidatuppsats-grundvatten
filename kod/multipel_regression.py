# ----------------------------------- Multipel regressionsmodell - Kandidatuppsats -----------------------------------

# ---------------------------------------------------- Importerar paket --------------------------------------------------------------

import math
import sys
from itertools import combinations
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

# Ändra manuellt till "Rör A", "Rör B" eller "Rör C".
observationsror_namn = "Rör C"

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


# ---------------------------------------------------- Enkel linjär regression för rangordning av kandidater -----------------------------------------------------

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


# ---------------------------------------------------- Osäkerhetsmått för enkel rangordning -----------------------------------------------------

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


# ---------------------------------------------------- Multipel regressionsmodell -----------------------------------------------------

metodnamn_multipel = "Multipel regressionsmodell"

# Antal av de bäst rankade enskilda referensrören som får föreslås som kandidater.
urval_fran_topp = 10

# Sätt minsta antal till 2 om modellen måste vara multipel.
min_antal_referensror_att_testa = 2

# Högsta antal referensrör som får testas i samma modell.
max_antal_referensror_att_testa = 5

# False betyder att koden testar flera antal och väljer bäst enligt LOOCV-RMSE.
# True betyder att koden bara testar det antal som står i manuellt_antal_referensror.
valj_antal_manuellt = False
manuellt_antal_referensror = 2

# Samma grundkrav som i den enkla regressionsmodellen.
min_datapar_multipel = min_datapar

# Extra skydd mot att den multipla modellen blir för stor i förhållande till datamängden.
# Antal parametrar = intercept + antal referensrör.
min_datapar_per_parameter = 8

max_antal_tillatet_av_datamangd = max(
    1,
    int(np.floor(antal_observationstidsblock_i_period / min_datapar_per_parameter)) - 1
)

max_antal_referensror_att_testa = min(
    max_antal_referensror_att_testa,
    max_antal_tillatet_av_datamangd
)

print("\nInställningar för multipel regressionsmodell:")
print("Urval från topp:", urval_fran_topp)
print("Minsta antal referensrör att testa:", min_antal_referensror_att_testa)
print("Största antal referensrör att testa:", max_antal_referensror_att_testa)
print("Minsta antal datapar:", min_datapar_multipel)
print("Datapar per parameter-krav:", min_datapar_per_parameter)


# ---------------------------------------------------- Datapar för flera referensrör -----------------------------------------------------

def funktion_skapa_multipla_datapar(
    valda_referensror_id: list,
    observations_tidsserie_aggr: pd.Series,
    referens_tidsserier_aggr: dict,
    tidsserie_min: str,
    tidsserie_max: str
):
    """Skapar en gemensam tabell för observationsröret och flera referensrör."""

    df = observations_tidsserie_aggr.rename("observationsvarde").to_frame()

    for i, referens_id in enumerate(valda_referensror_id, start=1):
        referens_id = str(referens_id)

        referensserie = referens_tidsserier_aggr[referens_id].copy()
        referensserie.name = f"referens_{i}"

        df = df.join(referensserie, how="inner")

    df = df.loc[tidsserie_min:tidsserie_max].copy()
    df = df.dropna()

    return df


# ---------------------------------------------------- Anpassar multipel regression -----------------------------------------------------

def funktion_anpassa_multipel_regression(
    datapar_df: pd.DataFrame,
    konfidensniva: float
):
    """Anpassar en multipel regressionsmodell med flera referensrör."""

    referenskolumner = [
        kolumn for kolumn in datapar_df.columns
        if kolumn.startswith("referens_")
    ]

    antal_referensror = len(referenskolumner)
    antal_parametrar = antal_referensror + 1

    df = datapar_df.dropna(
        subset=["observationsvarde"] + referenskolumner
    ).copy()

    n = len(df)

    if n < min_datapar_multipel:
        return None

    if n / antal_parametrar < min_datapar_per_parameter:
        return None

    datapar_start = df.index.min().normalize()
    datapar_slut = df.index.max().normalize()

    datapar_varaktighet_dagar = (datapar_slut - datapar_start).days
    antal_unika_manader = df.index.to_period("M").nunique()

    if antal_observationstidsblock_i_period > 0:
        andel_observationstidsblock_med_datapar = (
            n / antal_observationstidsblock_i_period
        )
    else:
        andel_observationstidsblock_med_datapar = 0

    if (
        datapar_varaktighet_dagar < min_varaktighet_dagar
        or antal_unika_manader < min_unika_manader
    ):
        return None

    y = df["observationsvarde"].to_numpy(dtype=float)
    x_referens = df[referenskolumner].to_numpy(dtype=float)

    # Första kolumnen är intercept. Övriga kolumner är referensrörens nivåer.
    x_design = np.column_stack([np.ones(n), x_referens])

    # Om referensrören är nästan identiska kan designmatrisen inte skattas stabilt.
    rang = np.linalg.matrix_rank(x_design)

    if rang < x_design.shape[1]:
        return None

    beta, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)

    y_hat = x_design @ beta
    residualer = y - y_hat

    sse = np.sum(residualer ** 2)
    sst = np.sum((y - y.mean()) ** 2)

    if sst > 0:
        r_kvadrat = 1 - (sse / sst)
    else:
        r_kvadrat = np.nan

    frihetsgrader = n - antal_parametrar

    if frihetsgrader <= 0:
        return None

    residualstandardfel = np.sqrt(sse / frihetsgrader)

    mse_train = np.mean(residualer ** 2)
    rmse_train = np.sqrt(mse_train)

    # LOOCV-RMSE används för att minska risken att välja en modell bara för att
    # den råkar passa träningsdatan bäst.
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    hat_diag = np.sum(x_design * (x_design @ xtx_inv), axis=1)

    if np.any(np.abs(1 - hat_diag) < 1e-8):
        return None

    loo_residualer = residualer / (1 - hat_diag)
    rmse_loocv = np.sqrt(np.mean(loo_residualer ** 2))

    students_t_varde = t.ppf(
        (1 + konfidensniva) / 2,
        df=frihetsgrader
    )

    # Förenklat osäkerhetsmått för den multipla modellen.
    # Här används residualstandardfelet från den multipla regressionen och en
    # justering för antal skattade parametrar.
    osakerhetsmatt = (
        students_t_varde
        * residualstandardfel
        * np.sqrt(1 + (antal_parametrar / n))
    )

    resultat_df = df.copy()
    resultat_df["prediktionsvarde"] = y_hat
    resultat_df["nedre_intervallgrans"] = resultat_df["prediktionsvarde"] - osakerhetsmatt
    resultat_df["ovre_intervallgrans"] = resultat_df["prediktionsvarde"] + osakerhetsmatt
    resultat_df["residual"] = residualer

    resultat_df["inom_osakerhetsintervall"] = (
        (resultat_df["observationsvarde"] >= resultat_df["nedre_intervallgrans"])
        & (resultat_df["observationsvarde"] <= resultat_df["ovre_intervallgrans"])
    )

    return {
        "n": n,
        "antal_referensror": antal_referensror,
        "antal_parametrar": antal_parametrar,
        "beta": beta,
        "residualstandardfel": residualstandardfel,
        "MSE_train": mse_train,
        "RMSE_train": rmse_train,
        "RMSE_LOOCV": rmse_loocv,
        "r_kvadrat": r_kvadrat,
        "students_t_varde": students_t_varde,
        "osakerhetsmatt": osakerhetsmatt,
        "datapar_varaktighet_dagar": datapar_varaktighet_dagar,
        "antal_unika_manader": antal_unika_manader,
        "andel_observationstidsblock_med_datapar": andel_observationstidsblock_med_datapar,
        "resultat_df": resultat_df
    }


# ---------------------------------------------------- Predikterad multipel tidsserie -----------------------------------------------------

def funktion_skapa_predikterad_tidsserie_multipel(
    valda_referensror_id: list,
    referens_tidsserier_aggr: dict,
    beta: np.ndarray,
    osakerhetsmatt: float,
    tidsserie_min: str,
    tidsserie_max: str
):
    """Skapar predikterad serie för observationsröret från flera referensrör."""

    df = None

    for i, referens_id in enumerate(valda_referensror_id, start=1):
        referens_id = str(referens_id)

        serie = referens_tidsserier_aggr[referens_id].loc[
            tidsserie_min:tidsserie_max
        ].dropna().copy()

        serie.name = f"referens_{i}"

        if df is None:
            df = serie.to_frame()
        else:
            df = df.join(serie, how="inner")

    if df is None or len(df) == 0:
        return None

    df = df.dropna().copy()

    if len(df) == 0:
        return None

    referenskolumner = [
        kolumn for kolumn in df.columns
        if kolumn.startswith("referens_")
    ]

    x_referens = df[referenskolumner].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(df)), x_referens])

    df["prediktionsvarde"] = x_design @ beta
    df["nedre_intervallgrans"] = df["prediktionsvarde"] - osakerhetsmatt
    df["ovre_intervallgrans"] = df["prediktionsvarde"] + osakerhetsmatt

    return df


# ---------------------------------------------------- Väljer kandidater till multipel modell -----------------------------------------------------

kandidat_referensror_statistiskt = (
    resultat_referensror_df["referens_id"]
    .astype(str)
    .head(urval_fran_topp)
    .tolist()
)

kandidat_granskning_df = resultat_referensror_df[
    resultat_referensror_df["referens_id"].astype(str).isin(kandidat_referensror_statistiskt)
][
    [
        "referens_id",
        "referens_namn",
        "osakerhetsmatt",
        "r_kvadrat",
        "n",
        "antal_unika_manader",
        "tackningsgrad_referensserie"
    ]
].copy()

kandidat_granskning_df = kandidat_granskning_df.reset_index(drop=True)
kandidat_granskning_df.index = range(1, len(kandidat_granskning_df) + 1)

print("\nStatistiskt föreslagna kandidater till multipel regressionsmodell:")
print(kandidat_granskning_df.to_string())

print(
    "\nGranska referensrören hydrogeologiskt innan de får användas i den multipla modellen.\n"
    "Välj endast referensrör som bedöms ha rimligt liknande hydrogeologiska egenskaper.\n"
)

godkanda_val = input(
    "Skriv numren på de referensrör som ska godkännas för multipel modell "
    "(exempel: 1,2,4). Tryck Enter för att godkänna alla: "
).strip()

if godkanda_val == "":
    kandidat_referensror = kandidat_referensror_statistiskt.copy()
else:
    valda_nummer = [
        int(x.strip())
        for x in godkanda_val.split(",")
        if x.strip() != ""
    ]

    giltiga_nummer = [
        nummer for nummer in valda_nummer
        if nummer in kandidat_granskning_df.index
    ]

    if len(giltiga_nummer) == 0:
        raise ValueError("Inga giltiga referensrör valdes för den multipla modellen.")

    kandidat_referensror = (
        kandidat_granskning_df.loc[giltiga_nummer, "referens_id"]
        .astype(str)
        .tolist()
    )

print("\nHydrogeologiskt godkända kandidater till multipel regressionsmodell:")
print(kandidat_referensror)

if len(kandidat_referensror) < min_antal_referensror_att_testa:
    raise ValueError(
        "För få referensrör godkändes för att kunna anpassa en multipel modell. "
        f"Minst {min_antal_referensror_att_testa} referensrör krävs."
    )


# ---------------------------------------------------- Bestämmer antal referensrör som ska testas -----------------------------------------------------

if valj_antal_manuellt:
    antal_att_testa = [manuellt_antal_referensror]
else:
    antal_att_testa = list(
        range(
            min_antal_referensror_att_testa,
            max_antal_referensror_att_testa + 1
        )
    )

antal_att_testa = [
    antal for antal in antal_att_testa
    if antal <= len(kandidat_referensror)
]

if len(antal_att_testa) == 0:
    raise ValueError("Inga möjliga antal referensrör finns att testa.")

print("\nAntal referensrör som testas:")
print(antal_att_testa)


# ---------------------------------------------------- Testar kombinationer av referensrör -----------------------------------------------------

multipla_modellresultat = []

for antal_referensror in antal_att_testa:

    for kombination in combinations(kandidat_referensror, antal_referensror):
        kombination = list(kombination)

        datapar_df = funktion_skapa_multipla_datapar(
            valda_referensror_id=kombination,
            observations_tidsserie_aggr=observations_tidsserie_aggr,
            referens_tidsserier_aggr=referens_tidsserier_aggr,
            tidsserie_min=tidsserie_min,
            tidsserie_max=tidsserie_max
        )

        modell = funktion_anpassa_multipel_regression(
            datapar_df=datapar_df,
            konfidensniva=konfidensniva
        )

        if modell is None:
            continue

        multipla_modellresultat.append({
            "referensror_id": kombination,
            "antal_referensror": modell["antal_referensror"],
            "antal_parametrar": modell["antal_parametrar"],
            "n": modell["n"],
            "MSE_train": modell["MSE_train"],
            "RMSE_train": modell["RMSE_train"],
            "RMSE_LOOCV": modell["RMSE_LOOCV"],
            "r_kvadrat": modell["r_kvadrat"],
            "residualstandardfel": modell["residualstandardfel"],
            "osakerhetsmatt": modell["osakerhetsmatt"],
            "datapar_varaktighet_dagar": modell["datapar_varaktighet_dagar"],
            "antal_unika_manader": modell["antal_unika_manader"],
            "andel_observationstidsblock_med_datapar": modell["andel_observationstidsblock_med_datapar"],
            "beta": modell["beta"],
            "resultat_df": modell["resultat_df"]
        })


# ---------------------------------------------------- Väljer bästa multipla modell -----------------------------------------------------

if len(multipla_modellresultat) == 0:
    raise ValueError("Ingen multipel regressionsmodell kunde beräknas med de valda kraven.")

multipla_modellresultat_df = pd.DataFrame(multipla_modellresultat)

# Modellen väljs efter lägst LOOCV-RMSE.
# Vid lika värden väljs modellen med färre referensrör.
multipla_modellresultat_df = multipla_modellresultat_df.sort_values(
    by=["RMSE_LOOCV", "antal_referensror"],
    ascending=[True, True]
).reset_index(drop=True)

# ---------------------------------------------------- Tabell: Topp 5 multipla modeller -----------------------------------------------------

def funktion_hamta_referensnamn(referens_id):
    """Hämtar referensnamn från resultat_referensror_df."""

    matchande_rad = (
        resultat_referensror_df["referens_id"].astype(str) == str(referens_id)
    )

    if matchande_rad.any():
        referens_namn = resultat_referensror_df.loc[
            matchande_rad,
            "referens_namn"
        ].iloc[0]

        if pd.notna(referens_namn):
            return str(referens_namn)

    return str(referens_id)


def funktion_formattera_referenslista(referenslista):
    """Gör lista med referensrör till en textsträng."""

    referensnamn = [
        funktion_hamta_referensnamn(referens_id)
        for referens_id in referenslista
    ]

    return ", ".join(referensnamn)


topp_5_multipla_modeller_df = multipla_modellresultat_df.head(5).copy()

topp_5_multipla_modeller_df["Rang"] = range(
    1,
    len(topp_5_multipla_modeller_df) + 1
)

topp_5_multipla_modeller_df["Referensrör"] = topp_5_multipla_modeller_df[
    "referensror_id"
].apply(funktion_formattera_referenslista)

topp_5_multipla_modeller_df["Vald"] = "Nej"
topp_5_multipla_modeller_df.loc[
    topp_5_multipla_modeller_df.index[0],
    "Vald"
] = "Ja"

topp_5_tabell_df = topp_5_multipla_modeller_df[
    [
        "Rang",
        "Referensrör",
        "antal_referensror",
        "n",
        "RMSE_LOOCV",
        "RMSE_train",
        "r_kvadrat",
        "osakerhetsmatt",
        "Vald"
    ]
].copy()

topp_5_tabell_df = topp_5_tabell_df.rename(
    columns={
        "antal_referensror": "Antal referensrör",
        "n": "n",
        "RMSE_LOOCV": "RMSE_LOOCV",
        "RMSE_train": "RMSE",
        "r_kvadrat": "R^2",
        "osakerhetsmatt": "c_S"
    }
)

# Avrundar numeriska kolumner så tabellen blir lättare att läsa.
kolumner_att_avrunda = [
    "RMSE_LOOCV",
    "RMSE",
    "R^2",
    "c_S"
]

for kolumn in kolumner_att_avrunda:
    topp_5_tabell_df[kolumn] = topp_5_tabell_df[kolumn].round(3)

print("\nTopp 5 multipla regressionsmodeller:")
print(topp_5_tabell_df.to_string(index=False))

# Sparar tabellen som csv om du vill använda den i rapporten.
topp_5_tabell_df.to_csv(
    f"topp_5_multipla_modeller_{filnamn_id}.csv",
    sep=";",
    decimal=",",
    index=False
)

multipel_oversikt_df = multipla_modellresultat_df[
    [
        "referensror_id",
        "antal_referensror",
        "n",
        "MSE_train",
        "RMSE_train",
        "RMSE_LOOCV",
        "r_kvadrat",
        "residualstandardfel",
        "osakerhetsmatt",
        "antal_unika_manader"
    ]
].head(20).copy()

multipel_oversikt_df.index = range(1, len(multipel_oversikt_df) + 1)

print("\nToppmodeller för multipel regressionsmodell:")
print(multipel_oversikt_df.to_string())


# ---------------------------------------------------- Sparar vald multipel modell -----------------------------------------------------

vald_multipel_modell = multipla_modellresultat_df.iloc[0]

valda_referensror_id = vald_multipel_modell["referensror_id"]
vald_beta = vald_multipel_modell["beta"]
periodens_resultat_df = vald_multipel_modell["resultat_df"].copy()

valda_referensnamn = []

for referens_id in valda_referensror_id:
    matchande_rad = (
        resultat_referensror_df["referens_id"].astype(str) == str(referens_id)
    )

    if matchande_rad.any():
        namn = resultat_referensror_df.loc[
            matchande_rad,
            "referens_namn"
        ].iloc[0]
    else:
        namn = str(referens_id)

    valda_referensnamn.append(str(namn))

valt_referensror_namn = "Multipel modell: " + ", ".join(valda_referensnamn)

predikterad_tidsserie_df = funktion_skapa_predikterad_tidsserie_multipel(
    valda_referensror_id=valda_referensror_id,
    referens_tidsserier_aggr=referens_tidsserier_aggr,
    beta=vald_beta,
    osakerhetsmatt=vald_multipel_modell["osakerhetsmatt"],
    tidsserie_min=tidsserie_min,
    tidsserie_max=tidsserie_max
)

if predikterad_tidsserie_df is None:
    raise ValueError("Den predikterade multipla tidsserien kunde inte skapas.")

print("\nVald multipel regressionsmodell:")
print("Valda referensrör:", valda_referensror_id)
print("Referensnamn:", valda_referensnamn)
print("Antal referensrör:", vald_multipel_modell["antal_referensror"])
print("Antal datapar:", vald_multipel_modell["n"])
print("MSE train:", vald_multipel_modell["MSE_train"])
print("RMSE train:", vald_multipel_modell["RMSE_train"])
print("RMSE LOOCV:", vald_multipel_modell["RMSE_LOOCV"])
print("r²:", vald_multipel_modell["r_kvadrat"])
print("Residualstandardfel:", vald_multipel_modell["residualstandardfel"])
print("Osäkerhetsmått:", vald_multipel_modell["osakerhetsmatt"])
print("Beta-koefficienter:", vald_beta)

print("\nTolkning av beta-koefficienter:")
print("Intercept beta_0:", vald_beta[0])

for i, referens_id in enumerate(valda_referensror_id, start=1):
    print(f"beta_{i} för referensrör {referens_id}:", vald_beta[i])

print("\nPredikterad multipel tidsserie:")
print(
    "Antal värden i observationsröret:",
    len(observations_tidsserie.loc[tidsserie_min:tidsserie_max].dropna())
)
print(
    "Antal värden i predikterad multipel serie:",
    len(predikterad_tidsserie_df)
)


# ---------------------------------------------------- Utvärderingsmått -----------------------------------------------------

def funktion_utvarderingsmatt(resultat_df: pd.DataFrame):
    """Beräknar MSE, RMSE och Durbin-Watson för residualerna."""

    df = resultat_df.dropna(
        subset=["observationsvarde", "prediktionsvarde"]
    ).copy()

    if len(df) == 0:
        return None

    residualer = df["observationsvarde"] - df["prediktionsvarde"]

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

def funktion_plotta_predikterad_tidsserie(
    predikterad_tidsserie_df: pd.DataFrame,
    observations_tidsserie: pd.Series,
    referens_namn: str,
    observationsror_namn: str,
    visa_osakerhetsintervall: bool = True
):
    """Plottar predikterad serie, observationsvärden och osäkerhetsintervall."""

    if visa_osakerhetsintervall:
        df = predikterad_tidsserie_df.dropna(
            subset=[
                "prediktionsvarde",
                "nedre_intervallgrans",
                "ovre_intervallgrans"
            ]
        ).copy()
    else:
        df = predikterad_tidsserie_df.dropna(
            subset=["prediktionsvarde"]
        ).copy()

    if len(df) == 0:
        print("Det finns inga värden att plotta för den predikterade tidsserien.")
        return None

    df = df.sort_index()

    obs = observations_tidsserie.loc[df.index.min():df.index.max()].dropna().copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    fig.suptitle(
        f"{metodnamn_multipel}\n"
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
        df["prediktionsvarde"],
        linestyle="-",
        linewidth=1.4,
        color="crimson",
        label=f"Predikterad serie ({referens_namn})"
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
        pd.Series(df["prediktionsvarde"].values),
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
        f"Histogram av residualer — {metodnamn_multipel} ({observationsror_namn})",
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
        f"Residualer över tid — {metodnamn_multipel}\n"
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
        f"SAC för residualer — {metodnamn_multipel} ({observationsror_namn})",
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

fig_huvud = funktion_plotta_predikterad_tidsserie(
    predikterad_tidsserie_df=predikterad_tidsserie_df,
    observations_tidsserie=observations_tidsserie,
    referens_namn=valt_referensror_namn,
    observationsror_namn=observationsror_namn,
    visa_osakerhetsintervall=visa_osakerhetsintervall
)

if fig_huvud is not None:
    fig_huvud.savefig(
        f"multipel_regressions_huvudfigur_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: multipel_regressions_huvudfigur_{filnamn_id}.png")

fig_hist = funktion_plotta_residualhistogram(
    resultat_df=periodens_resultat_df,
    observationsror_namn=observationsror_namn
)

if fig_hist is not None:
    fig_hist.savefig(
        f"multipel_regressions_histogram_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: multipel_regressions_histogram_{filnamn_id}.png")

fig_residualer = funktion_plotta_residualer_over_tid(
    resultat_df=periodens_resultat_df,
    observationsror_namn=observationsror_namn
)

if fig_residualer is not None:
    fig_residualer.savefig(
        f"multipel_regressions_residualer_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: multipel_regressions_residualer_{filnamn_id}.png")

fig_sac = funktion_plotta_sac_residualer(
    resultat_df=periodens_resultat_df,
    observationsror_namn=observationsror_namn,
    max_lag=10
)

if fig_sac is not None:
    fig_sac.savefig(
        f"multipel_regressions_sac_{filnamn_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Figur sparad: multipel_regressions_sac_{filnamn_id}.png")

plt.show()
