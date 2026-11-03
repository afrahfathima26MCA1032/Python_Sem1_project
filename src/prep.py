"""
prep.py - prepares the loan data.

This file is used by training, by the notebooks AND by the app,
so a loan is always prepared the same way everywhere.
"""
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- file locations
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_FOLDER / "data" / "raw" / "SBAnational.csv"
CLEAN_FILE = PROJECT_FOLDER / "data" / "processed" / "sba_clean.csv"


# ---------------------------------------------------------------- column groups
# Columns filled in only AFTER a loan failed. Using them would be cheating.
LEAKY_COLUMNS = ["ChgOffDate", "ChgOffPrinGr", "BalanceGross"]

# Columns that only identify a loan (name, ID number...). They cannot be learned from.
ID_COLUMNS = ["LoanNr_ChkDgt", "Name", "City", "Zip", "Bank"]

# Text columns turned into a number: the past default rate of that state or industry.
RATE_COLUMNS = ["State", "BankState", "NAICS_sector"]

# Text columns turned into 0/1 columns (for example UrbanRural -> UrbanRural_Urban = 1).
YES_NO_COLUMNS = ["RevLineCr", "LowDoc", "NewExist", "UrbanRural"]

# Columns that must never be given to the model.
# "default" is the answer itself. "Term" is leaky (see notebook 3).
NOT_FOR_MODEL = ["default", "Term"]

# How strongly a small group is pulled towards the average default rate.
SMOOTHING = 100

# The first two digits of an industry code -> the industry name.
INDUSTRY_NAMES = {
    "11": "Agriculture/Forestry", "21": "Mining & Extraction", "22": "Utilities",
    "23": "Construction", "31": "Manufacturing", "32": "Manufacturing",
    "33": "Manufacturing", "42": "Wholesale Trade", "44": "Retail Trade",
    "45": "Retail Trade", "48": "Transportation", "49": "Transportation",
    "51": "Information", "52": "Finance & Insurance", "53": "Real Estate",
    "54": "Professional Services", "55": "Management of Companies",
    "56": "Admin & Waste Services", "61": "Educational Services", "62": "Health Care",
    "71": "Arts & Recreation", "72": "Accommodation & Food", "81": "Other Services",
    "92": "Public Administration",
}


# ---------------------------------------------------------------- cleaning
def clean_yes_no(column):
    """Tidy a Y/N column. '0' means N, '1' means Y, anything else becomes 'Unknown'."""
    column = column.str.strip().str.upper()
    column = column.replace({"0": "N", "1": "Y"})
    is_valid = column.isin(["Y", "N"])
    return column.where(is_valid, "Unknown")


def to_number(column):
    """Turn text like '$60,000.00 ' into the number 60000.0 (unreadable values become blank)."""
    column = column.str.replace("$", "", regex=False)
    column = column.str.replace(",", "", regex=False)
    column = column.str.strip()
    return pd.to_numeric(column, errors="coerce")


def add_features(loans):
    """Add the five new columns we created ourselves."""
    # share of the loan the government guarantees
    loans["guarantee_ratio"] = loans["SBA_Appv"] / loans["GrAppv"]

    # share of the approved amount that was actually paid out
    loans["disbursement_ratio"] = loans["DisbursementGross"] / loans["GrAppv"]

    # loan size per employee (0 employees -> blank, so we never divide by zero)
    employees = loans["NoEmp"].replace(0, np.nan)
    loans["loan_per_employee"] = loans["GrAppv"] / employees

    # 1 if the bank is in the same state as the business, otherwise 0
    loans["same_state"] = (loans["State"] == loans["BankState"]).astype(int)

    # jobs created plus jobs kept
    loans["jobs_total"] = loans["CreateJob"].fillna(0) + loans["RetainedJob"].fillna(0)

    # a division by zero gives "infinity" - turn it into blank instead
    for column in ["guarantee_ratio", "disbursement_ratio", "loan_per_employee"]:
        loans[column] = loans[column].replace([np.inf, -np.inf], np.nan)
    return loans


def clean(raw):
    """Raw SBA file (read as text) -> clean table with one row per loan."""
    # 1. remove the leaky columns and the ID columns
    loans = raw.drop(columns=LEAKY_COLUMNS + ID_COLUMNS)

    # 2. remove loans with no result recorded
    loans = loans[loans["MIS_Status"].notna()].copy()

    # 3. the answer we want to predict: 1 = charged off (not paid back), 0 = paid in full
    loans["default"] = (loans["MIS_Status"].str.strip() == "CHGOFF").astype(int)
    loans = loans.drop(columns=["MIS_Status"])

    # 4. money columns: text -> numbers
    for column in ["DisbursementGross", "GrAppv", "SBA_Appv"]:
        loans[column] = to_number(loans[column])

    # 5. plain number columns
    for column in ["Term", "NoEmp", "CreateJob", "RetainedJob"]:
        loans[column] = pd.to_numeric(loans[column], errors="coerce")

    # 6. approval year: keep only the 4 digits ('1976A' -> 1976)
    year_digits = loans["ApprovalFY"].str.extract(r"(\d{4})")[0]
    loans["ApprovalFY"] = pd.to_numeric(year_digits, errors="coerce")

    # 7. fix the category codes
    loans["RevLineCr"] = clean_yes_no(loans["RevLineCr"])
    loans["LowDoc"] = clean_yes_no(loans["LowDoc"])
    loans["NewExist"] = loans["NewExist"].str.strip().map({"1": "Existing", "2": "New"})
    loans["NewExist"] = loans["NewExist"].fillna("Unknown")
    loans["UrbanRural"] = loans["UrbanRural"].str.strip().map({"1": "Urban", "2": "Rural", "0": "Undefined"})
    loans["UrbanRural"] = loans["UrbanRural"].fillna("Undefined")

    # 8. franchise: the codes 0 and 1 both mean "not a franchise"
    franchise_code = pd.to_numeric(loans["FranchiseCode"], errors="coerce").fillna(0)
    loans["is_franchise"] = (~franchise_code.isin([0, 1])).astype(int)

    # 9. industry code -> industry name
    first_two_digits = loans["NAICS"].fillna("").str.strip().str[:2]
    loans["NAICS_sector"] = first_two_digits.map(INDUSTRY_NAMES).fillna("Unknown")

    # 10. drop columns we no longer need, then add our own features
    loans = loans.drop(columns=["FranchiseCode", "NAICS", "ApprovalDate", "DisbursementDate"])
    return add_features(loans)


def load_clean():
    """Return the clean data. The first time, build it from the raw file and save it."""
    if not CLEAN_FILE.exists():
        raw = pd.read_csv(RAW_FILE, dtype=str, low_memory=False)
        CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
        clean(raw).to_csv(CLEAN_FILE, index=False)
    return pd.read_csv(CLEAN_FILE)


# ---------------------------------------------------------------- encoding
def fit_encoder(X_train, y_train):
    """
    Learn everything the encoding needs - from TRAINING loans only.

    For each state, bank state and industry we store its past default rate.
    Small groups are pulled towards the overall rate, so a state with
    only 40 loans cannot get an extreme number.
    """
    overall_rate = float(y_train.mean())
    rates = {}
    for column in RATE_COLUMNS:
        defaults = y_train.groupby(X_train[column]).sum()
        loan_counts = y_train.groupby(X_train[column]).count()
        smoothed = (defaults + overall_rate * SMOOTHING) / (loan_counts + SMOOTHING)
        rates[column] = {}
        for group_name, rate in smoothed.items():
            rates[column][str(group_name)] = float(rate)

    encoder = {"rates": rates, "overall_rate": overall_rate, "columns": None, "medians": {}}

    # encode the training loans once, to learn the final column list and the medians
    X_encoded = encode(X_train, encoder)
    encoder["columns"] = list(X_encoded.columns)
    for column in X_encoded.columns:
        encoder["medians"][column] = float(X_encoded[column].median())
    return encoder


def encode(loans, encoder):
    """
    Turn loans into the model's 24 number columns:
      - remove the answer and Term
      - state / bank state / industry -> their past default rate
      - Y/N style columns -> 0/1 columns
      - put the columns in the training order and fill gaps with the training median
    """
    X = loans.drop(columns=NOT_FOR_MODEL, errors="ignore").copy()

    for column in RATE_COLUMNS:
        rates = encoder["rates"][column]
        X[column + "_risk"] = X[column].astype(str).map(rates).fillna(encoder["overall_rate"])
    X = X.drop(columns=RATE_COLUMNS)

    learning = encoder["columns"] is None
    if learning:
        # training: drop_first removes one "baseline" 0/1 column per category
        return pd.get_dummies(X, columns=YES_NO_COLUMNS, dtype=float, drop_first=True)

    # predicting: make every 0/1 column, then keep only the training columns.
    # (drop_first would delete the answer itself when there is only one loan)
    X = pd.get_dummies(X, columns=YES_NO_COLUMNS, dtype=float)
    X = X.reindex(columns=encoder["columns"], fill_value=0.0)
    X = X.fillna(encoder["medians"])
    return X.astype(float)
