"""
stats.py - precomputes the small files the app reads.

Run it with:   python3 -m src.stats

  regional()  default rate per state, per industry, and per state + industry -> outputs/regional_stats.json
  survival()  how long loans last, overall and per industry                  -> outputs/survival_stats.json

The app reads these small files instead of the 171 MB data file, so it opens instantly.
"""
import json

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter

from src import prep

OUTPUTS_FOLDER = prep.PROJECT_FOLDER / "outputs"

MIN_LOANS_FOR_A_RATE = 100        # a default rate from fewer loans is just noise
MIN_LOANS_FOR_A_CURVE = 500       # a survival curve from fewer loans is just noise
CHECKPOINT_MONTHS = [12, 24, 36, 60, 84, 120]
TODAY = pd.Timestamp("2026-08-28")


# ---------------------------------------------------------------- regional default rates
def default_rates(loans, group_by):
    """For each group: the share of loans not paid back, and how many loans it has."""
    grouped = loans.groupby(group_by)["default"]
    return pd.DataFrame({"rate": grouped.mean(), "count": grouped.size()})


def regional():
    loans = prep.load_clean()

    states = default_rates(loans, "State").sort_values("rate", ascending=False)
    states["rank"] = range(1, len(states) + 1)          # rank 1 = worst state
    industries = default_rates(loans, "NAICS_sector").sort_values("rate", ascending=False)
    both = default_rates(loans, ["State", "NAICS_sector"])
    both = both[both["count"] >= MIN_LOANS_FOR_A_RATE]

    result = {"national_rate": round(float(loans["default"].mean()), 5),
              "min_cell": MIN_LOANS_FOR_A_RATE,
              "n_states": len(states),
              "states": {}, "sectors": {}, "state_sector": {}}

    for state, row in states.iterrows():
        result["states"][state] = {"rate": round(row["rate"], 5), "count": int(row["count"]),
                                   "rank": int(row["rank"])}
    for industry, row in industries.iterrows():
        result["sectors"][industry] = {"rate": round(row["rate"], 5), "count": int(row["count"])}
    for (state, industry), row in both.iterrows():
        result["state_sector"][state + "|" + industry] = {"rate": round(row["rate"], 5),
                                                          "count": int(row["count"])}

    (OUTPUTS_FOLDER / "regional_stats.json").write_text(json.dumps(result, indent=1))
    return result


# ---------------------------------------------------------------- survival
def read_date(text_column):
    """Read dates like '28-Feb-97'. Two-digit years such as '66' are read as 2066, so move them back 100 years."""
    dates = pd.to_datetime(text_column, format="%d-%b-%y", errors="coerce")
    is_in_future = dates > TODAY
    return dates.mask(is_in_future, dates - pd.DateOffset(years=100))


def durations(extra_columns=()):
    """
    One row per loan with:
      - duration: how many months we watched the loan
      - failed:   1 if it was written off, 0 if it was paid in full

    A loan that was written off: duration = charge-off date minus pay-out date.
    A loan that was paid in full: we only know it lasted its full term ("censored").
    """
    columns = ["DisbursementDate", "ChgOffDate", "Term", "MIS_Status", "NAICS"] + list(extra_columns)
    raw = pd.read_csv(prep.RAW_FILE, dtype=str, usecols=columns)

    status = raw["MIS_Status"].str.strip()
    loans = raw[list(extra_columns)].copy()
    loans["failed"] = (status == "CHGOFF").astype(int)

    months_until_write_off = (read_date(raw["ChgOffDate"]) - read_date(raw["DisbursementDate"])).dt.days / 30.44
    term_in_months = pd.to_numeric(raw["Term"], errors="coerce")
    loans["duration"] = np.where(loans["failed"] == 1, months_until_write_off, term_in_months)

    first_two_digits = raw["NAICS"].fillna("").str.strip().str[:2]
    loans["sector"] = first_two_digits.map(prep.INDUSTRY_NAMES).fillna("Unknown")

    # keep loans with a known result and a sensible length (more than 0, at most 40 years)
    has_result = status.isin(["CHGOFF", "P I F"])
    sensible_length = (loans["duration"] > 0) & (loans["duration"] <= 480)
    return loans[has_result & sensible_length]


def survival_curve(loans):
    """
    For one group of loans:
      - the share not yet written off after 1, 2, 3, 5, 7 and 10 years
      - the month when write-offs happened most often (the "peak")
    """
    kaplan_meier = KaplanMeierFitter().fit(loans["duration"], loans["failed"])
    event_table = kaplan_meier.event_table
    number_of_loans = len(loans)

    # A checkpoint is only reported while enough loans are still being watched.
    # Later on, most loans have finished, and the number would be based on very few loans.
    min_still_watched = max(100, int(0.05 * number_of_loans))
    survival = {}
    last_reliable_month = 0
    for month in CHECKPOINT_MONTHS:
        rows_up_to_month = event_table[event_table.index <= month]
        if len(rows_up_to_month) > 0:
            still_watched = rows_up_to_month["at_risk"].iloc[-1]
        else:
            still_watched = number_of_loans
        if still_watched >= min_still_watched:
            survival[str(month)] = round(float(kaplan_meier.predict(month)), 4)
            last_reliable_month = month

    # The peak month is only reported if it is a real peak inside the reliable window,
    # not a value that is still rising when the data runs out.
    peak_month = None
    enough_failures = loans["failed"].sum() >= 50
    if enough_failures and last_reliable_month >= 36:
        nelson_aalen = NelsonAalenFitter().fit(loans["duration"], loans["failed"])
        hazard = nelson_aalen.smoothed_hazard_(bandwidth=6).iloc[:, 0]
        hazard = hazard[(hazard.index >= 6) & (hazard.index <= last_reliable_month)]
        if len(hazard) > 4:
            highest_month = hazard.idxmax()
            if 6 < highest_month < last_reliable_month - 6:
                peak_month = int(round(float(highest_month)))

    return {"n": number_of_loans, "failed": int(loans["failed"].sum()), "survival": survival,
            "peak_month": peak_month, "horizon": last_reliable_month, "min_at_risk": min_still_watched}


def survival():
    loans = durations()
    result = {"checkpoints": CHECKPOINT_MONTHS, "min_group": MIN_LOANS_FOR_A_CURVE,
              "overall": survival_curve(loans), "by_sector": {}}
    for industry, industry_loans in loans.groupby("sector"):
        if len(industry_loans) >= MIN_LOANS_FOR_A_CURVE:
            result["by_sector"][industry] = survival_curve(industry_loans)
    (OUTPUTS_FOLDER / "survival_stats.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    regional()
    print("overall riskiest month:", survival()["overall"]["peak_month"])
