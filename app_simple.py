"""
app_simple.py - the Loan Risk Assistant web app.

Run it with:   streamlit run app_simple.py

Pages: New Assessment, My Assessments, Data Analysis, Market Lookup, About the Model.
The model is trained by src/train.py and the small data files are made by src/stats.py.
"""
import json
import uuid
from datetime import date

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

from src import prep


# ================================================================== settings
st.set_page_config(page_title="Loan Risk Assistant", page_icon="🏦", layout="wide")

# hide Streamlit's own menu buttons and keep the page a comfortable width
st.markdown("""<style>
  [data-testid="stToolbarActions"], [data-testid="stMainMenu"], .stAppDeployButton,
  [data-testid="stStatusWidget"], footer { display: none !important; }
  .block-container { padding-top: 4.2rem; max-width: 1150px; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>""", unsafe_allow_html=True)

OUTPUTS_FOLDER = prep.PROJECT_FOLDER / "outputs"
CHARTS_FOLDER = OUTPUTS_FOLDER / "figures"
SAVED_FILE = prep.PROJECT_FOLDER / "data" / "assessments.csv"

# colours
GREEN = "#2a9d8f"
ORANGE = "#e9a23b"
RED = "#e76f51"
TEXT_COLOUR = "#e8eaed"

# risk levels
LOW_RISK_BELOW = 0.10          # under 10 in 100  -> lower risk
HIGH_RISK_FROM = 0.35          # 35 in 100 or more -> higher risk

# the app never shows exactly 0% or 100% - no model is that certain about one loan
SMALLEST_RISK = 0.005
LARGEST_RISK = 0.995

# the app scores every loan as if it was approved in 2004 (a typical year: 18% default)
TYPICAL_YEAR = 2004

# suggest a check-in this many months before loans usually run into trouble
MONTHS_BEFORE_PEAK = 6

# each risk level -> its colour and the suggested next step (our own rule, not the model's)
RISK_LEVELS = {
    "Lower risk": (GREEN, "Standard checks"),
    "Medium risk": (ORANGE, "Ask for financial statements before deciding"),
    "Higher risk": (RED, "Senior manager review required"),
}

# each form question -> the model columns it creates.
# Their SHAP values are added up, so "why" shows one reason per question.
QUESTION_COLUMNS = {
    "loan": ["GrAppv", "DisbursementGross"],
    "guarantee": ["guarantee_ratio", "SBA_Appv"],
    "industry": ["NAICS_sector_risk"],
    "employees": ["NoEmp", "loan_per_employee"],
    "age": ["NewExist_New"],
    "state": ["State_risk"],
    "bank": ["BankState_risk", "same_state"],
    "area": ["UrbanRural_Urban", "UrbanRural_Undefined"],
    "credit": ["RevLineCr_Y"],
}


# ================================================================== loading (done once)
@st.cache_resource
def load_model():
    """Load the trained model file and build the SHAP explainer once."""
    model_file = joblib.load(prep.PROJECT_FOLDER / "models" / "model.joblib")
    explainer = shap.TreeExplainer(model_file["model"])
    return model_file, explainer


@st.cache_data
def load_json(file_name):
    """Read one of the small result files from the outputs folder."""
    return json.loads((OUTPUTS_FOLDER / file_name).read_text())


MODEL_FILE, EXPLAINER = load_model()
REGIONAL = load_json("regional_stats.json")
SURVIVAL = load_json("survival_stats.json")
STATES = sorted(REGIONAL["states"])
INDUSTRIES = sorted(REGIONAL["sectors"])
NATIONAL_RATE = REGIONAL["national_rate"]


# ================================================================== prediction
def risk_level(risk):
    """Turn a probability into Lower / Medium / Higher risk."""
    if risk < LOW_RISK_BELOW:
        return "Lower risk"
    if risk < HIGH_RISK_FROM:
        return "Medium risk"
    return "Higher risk"


def review_month(industry):
    """The month to check on a loan: 6 months before this industry's riskiest month."""
    peak_month = SURVIVAL["by_sector"].get(industry, {}).get("peak_month")
    if peak_month is None:
        peak_month = SURVIVAL["overall"]["peak_month"]
    month = round((peak_month - MONTHS_BEFORE_PEAK) / 6) * 6      # round to a 6-month step
    return max(12, month)


def assess(answers):
    """Form answers -> risk, risk level, review date and the reasons."""
    # 1. write the answers as one loan, using the same column names as the data
    if answers["age"] == "New":
        business_age = "New"
    else:
        business_age = "Existing"
    area_codes = {"City": "Urban", "Countryside": "Rural", "Not sure": "Undefined"}
    if answers["credit"] == "Yes":
        credit_line = "Y"
    else:
        credit_line = "N"

    loan = pd.DataFrame([{
        "State": answers["state"],
        "BankState": answers["bank_state"],
        "ApprovalFY": TYPICAL_YEAR,
        "NoEmp": answers["employees"],
        "NewExist": business_age,
        "CreateJob": 0,
        "RetainedJob": 0,
        "LowDoc": "N",
        "is_franchise": 0,
        "UrbanRural": area_codes[answers["area"]],
        "RevLineCr": credit_line,
        "NAICS_sector": answers["industry"],
        "GrAppv": answers["amount"],
        "DisbursementGross": answers["amount"],
        "SBA_Appv": answers["amount"] * answers["guarantee"] / 100,
    }])

    # 2. prepare it exactly like the training data (same code as training)
    loan = prep.add_features(loan)
    X = prep.encode(loan, MODEL_FILE["encoder"])

    # 3. the calibrated chance of default
    risk = MODEL_FILE["calibrated"].predict_proba(X)[0, 1]
    risk = float(np.clip(risk, SMALLEST_RISK, LARGEST_RISK))

    # 4. the reasons: SHAP value of every column, then added up per form question
    shap_values = EXPLAINER.shap_values(X, check_additivity=False)
    shap_for_default = pd.Series(np.array(shap_values)[0, :, 1], index=X.columns)
    reasons = {}
    for question, columns in QUESTION_COLUMNS.items():
        reasons[question] = shap_for_default[columns].sum()
    reasons = pd.Series(reasons)
    biggest_first = reasons.abs().sort_values(ascending=False).index
    reasons = reasons.reindex(biggest_first)

    # 5. when to review the loan
    months = review_month(answers["industry"])
    review_date = (pd.Timestamp(date.today()) + pd.DateOffset(months=months)).date()

    return {"answers": answers, "risk": risk, "level": risk_level(risk),
            "review_month": months, "review_date": review_date, "reasons": reasons}


def reason_text(question, answers):
    """A short label for one reason, using the user's own answer."""
    if question == "loan":
        return f"Loan of ${answers['amount']:,.0f}"
    if question == "guarantee":
        return f"{answers['guarantee']}% guarantee"
    if question == "industry":
        return answers["industry"]
    if question == "employees":
        return f"{answers['employees']} employees"
    if question == "age":
        return f"{answers['age']} business"
    if question == "state":
        return f"Business in {answers['state']}"
    if question == "bank":
        if answers["state"] == answers["bank_state"]:
            return "Bank in same state"
        return "Bank in another state"
    if question == "area":
        return answers["area"]
    if answers["credit"] == "Yes":
        return "Has a credit line"
    return "No credit line"


def load_saved():
    """All saved assessments (an empty table if there are none yet)."""
    if SAVED_FILE.exists():
        return pd.read_csv(SAVED_FILE)
    return pd.DataFrame()


# ================================================================== small display helpers
def show_plotly(figure, height=300):
    """Show a Plotly chart with a transparent background and no toolbar."""
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color=TEXT_COLOUR, height=height,
                         margin=dict(l=40, r=40, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(figure, width="stretch",
                    config={"displayModeBar": False, "scrollZoom": False})


def tile(title, value, note=None):
    """A small bordered box: a title, a bold value and an optional note."""
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"**{value}**")
        if note:
            st.caption(note)


# ================================================================== page 1: New Assessment
def new_assessment():
    st.title("New Assessment")

    # ---- the form
    with st.form("assess"):
        name = st.text_input("Business name", placeholder="e.g. Sunrise Bakery")
        column1, column2, column3 = st.columns(3)
        answers = {}
        answers["amount"] = column1.number_input("Loan amount ($)", 5_000, 5_000_000, 150_000, step=5_000)
        answers["guarantee"] = column1.slider("Government guarantee (%)", 50, 90, 75, step=5,
                                              help="Share the government repays the bank if the business can't.")
        answers["industry"] = column2.selectbox("Industry", INDUSTRIES, index=INDUSTRIES.index("Retail Trade"))
        answers["employees"] = column2.number_input("Employees", 0, 5_000, 8)
        answers["age"] = column2.selectbox("Business age", ["Established", "New"])
        answers["state"] = column3.selectbox("Business state", STATES, index=STATES.index("CA"))
        answers["bank_state"] = column3.selectbox("Bank's state", STATES, index=STATES.index("CA"))
        answers["area"] = column3.selectbox("City or countryside", ["City", "Countryside", "Not sure"])
        answers["credit"] = column3.selectbox("Credit line", ["No", "Yes"])
        pressed = st.form_submit_button("Assess", type="primary", width="stretch")

    # ---- when Assess is pressed, work out the result and remember it
    if pressed:
        result = assess(answers)
        result["name"] = name.strip()
        if result["name"] == "":
            result["name"] = "Unnamed business"
        result["id"] = uuid.uuid4().hex          # a unique id, so it can only be saved once
        st.session_state["result"] = result

    if "result" not in st.session_state:
        return
    result = st.session_state["result"]
    colour, next_step = RISK_LEVELS[result["level"]]

    # ---- name and risk badge
    st.divider()
    st.markdown(f"### {result['name']} &nbsp;<span style='background:{colour};color:#fff;"
                f"padding:3px 12px;border-radius:999px;font-size:14px'>{result['level']}</span>",
                unsafe_allow_html=True)

    # ---- risk gauge (left) and next step + review date (right)
    left, right = st.columns(2)
    with left:
        gauge = go.Indicator(
            mode="gauge+number",
            value=round(result["risk"] * 100),
            number={"suffix": " in 100", "font": {"color": colour}},
            gauge={"axis": {"range": [0, 100], "tickvals": [0, 10, 35, 100]},
                   "bar": {"color": colour},
                   "steps": [{"range": [0, 10], "color": "rgba(42,157,143,.25)"},
                             {"range": [10, 35], "color": "rgba(233,162,59,.25)"},
                             {"range": [35, 100], "color": "rgba(231,111,81,.25)"}]})
        show_plotly(go.Figure(gauge), height=210)
        st.caption(f"of loans like this were not paid back · average {NATIONAL_RATE * 100:.0f} in 100")
    with right:
        tile("Suggested next step", next_step)
        industry = result["answers"]["industry"].lower()
        tile("Review this loan on", f"{result['review_date']:%d %b %Y}",
             f"Month {result['review_month']}, just before {industry} loans usually run into trouble.")

    # ---- the three biggest reasons
    st.markdown("**Main reasons**")
    top_three = result["reasons"].head(3)
    reason_columns = st.columns(3)
    position = 0
    for question, value in top_three.items():
        label = reason_text(question, result["answers"])
        if value > 0:
            effect = ":red[▲ raises risk]"
        else:
            effect = ":green[▼ lowers risk]"
        reason_columns[position].container(border=True).markdown(label + "  \n" + effect)
        position = position + 1

    # ---- all reasons as a bar chart
    with st.expander("All reasons"):
        smallest_first = result["reasons"][::-1]
        labels = []
        colours = []
        for question, value in smallest_first.items():
            labels.append(reason_text(question, result["answers"]))
            if value > 0:
                colours.append(RED)
            else:
                colours.append(GREEN)
        figure, axis = plt.subplots(figsize=(7.5, 3))
        figure.patch.set_alpha(0)
        axis.set_facecolor("none")
        axis.barh(labels, smallest_first.values, color=colours)
        axis.axvline(0, color=TEXT_COLOUR, lw=1)
        axis.set_xticks([])
        axis.tick_params(colors=TEXT_COLOUR, labelsize=9)
        axis.set_xlabel("← lowers risk          raises risk →", color=TEXT_COLOUR)
        for side in axis.spines.values():
            side.set_visible(False)
        st.pyplot(figure, transparent=True)
        plt.close(figure)

    # ---- save / clear buttons
    if "saved_ids" not in st.session_state:
        st.session_state["saved_ids"] = set()
    already_saved = result["id"] in st.session_state["saved_ids"]
    if already_saved:
        save_label = "Saved ✓"
    else:
        save_label = "Save to My Assessments"

    button1, button2, _ = st.columns([1, 1, 2])
    if button1.button(save_label, type="primary", disabled=already_saved, width="stretch"):
        answers = result["answers"]
        main_reasons = []
        for question in result["reasons"].index[:3]:
            main_reasons.append(reason_text(question, answers))
        new_row = {
            "checked_on": date.today(),
            "business_name": result["name"],
            "loan_amount": answers["amount"],
            "guarantee_pct": answers["guarantee"],
            "industry": answers["industry"],
            "employees": answers["employees"],
            "business_age": answers["age"],
            "business_state": answers["state"],
            "bank_state": answers["bank_state"],
            "area": answers["area"],
            "credit_line": answers["credit"],
            "risk_out_of_100": round(result["risk"] * 100, 1),
            "risk_level": result["level"],
            "next_step": next_step,
            "review_date": result["review_date"],
            "main_reasons": "; ".join(main_reasons),
        }
        all_saved = pd.concat([load_saved(), pd.DataFrame([new_row])])
        SAVED_FILE.parent.mkdir(exist_ok=True)
        all_saved.to_csv(SAVED_FILE, index=False)
        st.session_state["saved_ids"].add(result["id"])
        st.rerun()

    if button2.button("Clear", width="stretch"):
        del st.session_state["result"]
        st.rerun()

    st.caption("A guide to support the decision — it does not approve or refuse loans.")


# ================================================================== page 2: My Assessments
def my_assessments():
    st.title("My Assessments")
    saved = load_saved()
    if saved.empty:
        st.info("No saved assessments yet. Assess a loan and press **Save to My Assessments**.")
        return

    # ---- summary cards
    card1, card2, card3, card4 = st.columns(4)
    card1.metric("Assessed", len(saved))
    card2.metric("Higher risk", int((saved["risk_level"] == "Higher risk").sum()))
    card3.metric("Average risk", f"{saved['risk_out_of_100'].mean():.0f} in 100")
    card4.metric("Total amount", f"${saved['loan_amount'].sum():,.0f}")

    # ---- search and filter
    search_column, filter_column = st.columns([2, 1])
    search_text = search_column.text_input("Search business name")
    chosen_levels = filter_column.multiselect("Risk level", list(RISK_LEVELS), default=list(RISK_LEVELS))
    matches_level = saved["risk_level"].isin(chosen_levels)
    matches_name = saved["business_name"].str.contains(search_text, case=False, na=False)
    shown = saved[matches_level & matches_name]

    # ---- the table, with friendly column names
    friendly_names = {"checked_on": "Checked", "business_name": "Business", "loan_amount": "Amount",
                      "industry": "Industry", "business_state": "State",
                      "risk_out_of_100": "Risk (out of 100)", "risk_level": "Level",
                      "next_step": "Suggested next step", "review_date": "Review on"}
    table = shown[list(friendly_names)].rename(columns=friendly_names)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Amount": st.column_config.NumberColumn(format="$%d"),
        "Risk (out of 100)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})

    # ---- export and delete
    export_column, delete_column, _ = st.columns([1, 1, 2])
    export_column.download_button("Export to CSV", shown.to_csv(index=False), "my_assessments.csv",
                                  width="stretch")
    with delete_column.popover("Delete all", width="stretch"):
        if st.button("Yes, delete every saved assessment", type="primary"):
            SAVED_FILE.unlink(missing_ok=True)
            st.rerun()


# ================================================================== page 3: Data Analysis
def data_analysis():
    st.title("Data Analysis")
    charts = [
        ("01_class_balance", "1. Most loans are paid back",
         "About 82 in 100 paid back in full; 18 in 100 not."),
        ("02_default_by_sector", "2. Industry makes a difference",
         "Property, finance and transport did worst; farming, mining and health care best. Dashed line = average."),
        ("18_regional_heatmap", "3. Industry × state",
         "Loans not paid back out of 100. Red = more, green = fewer, blank = too few."),
        ("15_correlation_matrix", "4. Which numbers move together",
         "Red = rise together, blue = opposite, white = no link. Every link with 'default' is weak."),
        ("16_distributions", "5. Paid-back vs unpaid loans",
         "Green = paid back, orange = not. Unpaid loans were usually smaller and bunched in the mid-2000s."),
        ("17_outliers", "6. Unusually large values",
         "Dots outside the boxes (about 1 in 10) are real loans, so they were kept."),
    ]
    # the first two charts side by side, the rest full width
    first_row = st.columns(2)
    for number, (file_name, title, note) in enumerate(charts):
        if number < 2:
            place = first_row[number]
        else:
            place = st.container()
        place.markdown(f"#### {title}")
        place.image(str(CHARTS_FOLDER / f"{file_name}.png"), width="stretch")
        place.caption(note)


# ================================================================== page 4: Market Lookup
def market_lookup():
    st.title("Market Lookup")
    state_column, industry_column = st.columns(2)
    state = state_column.selectbox("State", STATES, index=STATES.index("CA"))
    industry = industry_column.selectbox("Industry", INDUSTRIES, index=INDUSTRIES.index("Retail Trade"))

    # ---- four cards: the state, the industry, both together, and all of the US
    state_record = REGIONAL["states"][state]
    industry_record = REGIONAL["sectors"][industry]
    both_record = REGIONAL["state_sector"].get(state + "|" + industry)

    cards = st.columns(4)
    cards[0].metric(f"All loans in {state}", f"{state_record['rate'] * 100:.0f} in 100",
                    help=f"{state_record['count']:,} past loans")
    cards[1].metric(f"All {industry}", f"{industry_record['rate'] * 100:.0f} in 100",
                    help=f"{industry_record['count']:,} past loans")
    if both_record is None:
        cards[2].metric(f"{industry} in {state}", "too few", help="Fewer than 100 past loans")
    else:
        cards[2].metric(f"{industry} in {state}", f"{both_record['rate'] * 100:.0f} in 100",
                        help=f"{both_record['count']:,} past loans")
    cards[3].metric("All US", f"{NATIONAL_RATE * 100:.0f} in 100")
    st.caption("Share of past loans that were not paid back.")

    left, right = st.columns(2)

    # ---- map of every state
    with left:
        st.markdown("**By state**")
        rows = []
        for state_code, record in REGIONAL["states"].items():
            rows.append({"State": state_code,
                         "Not paid back (out of 100)": round(record["rate"] * 100),
                         "Loans": record["count"]})
        map_figure = px.choropleth(pd.DataFrame(rows), locations="State", locationmode="USA-states",
                                   scope="usa", color="Not paid back (out of 100)",
                                   color_continuous_scale="RdYlGn_r", hover_data={"Loans": ":,"})
        map_figure.update_layout(geo=dict(bgcolor="rgba(0,0,0,0)"))
        show_plotly(map_figure)

    # ---- the chosen industry in every state
    with right:
        st.markdown(f"**{industry}, by state**")
        rows = []
        for key, record in REGIONAL["state_sector"].items():
            state_code, industry_name = key.split("|")
            if industry_name == industry:
                rows.append({"State": state_code,
                             "Not paid back (out of 100)": round(record["rate"] * 100)})
        if len(rows) < 2:
            st.caption("Too few loans to compare states.")
            return
        bars = pd.DataFrame(rows).sort_values("Not paid back (out of 100)", ascending=False)
        bar_figure = px.bar(bars, x="State", y="Not paid back (out of 100)",
                            color="Not paid back (out of 100)", color_continuous_scale="RdYlGn_r")
        show_plotly(bar_figure)


# ================================================================== page 5: About the Model
def about_model():
    st.title("About the Model")
    run_summary = load_json("run_summary.json")
    calibration = load_json("calibration_summary.json")

    cards = st.columns(4)
    cards[0].metric("Trained on", f"{run_summary['clean_rows']:,} loans")
    cards[1].metric("ROC-AUC", f"{run_summary['results']['Random Forest']['roc_auc']:.2f}",
                    help="How often a failed loan is rated riskier than a repaid one.")
    cards[2].metric("Bad loans caught", f"{MODEL_FILE['metrics']['recall'] * 100:.0f}%")
    cards[3].metric("Risk number error", f"{calibration['ece_after'] * 100:.1f} points",
                    help="Average gap between the predicted and the real share of unpaid loans.")

    left, right = st.columns([1.1, 1])
    left.markdown("**Predicted vs actual, by risk level**")
    left.image(str(CHARTS_FOLDER / "13_risk_tiers.png"), width="stretch")
    right.markdown("**Good to know**")
    right.markdown("- Random Forest model, explained with SHAP, calibrated on unseen loans.\n"
                   "- Knows nothing about the business's finances (sales, profit, debts).\n"
                   "- Learned from approved US loans, 1966–2014.\n"
                   "- Suggested next steps are simple rules set for this tool.")


# ================================================================== navigation bar
st.markdown("#### 🏦 Loan Risk Assistant")
pages = [
    st.Page(new_assessment, title="New Assessment", url_path="assess", default=True),
    st.Page(my_assessments, title=f"My Assessments ({len(load_saved())})", url_path="saved"),
    st.Page(data_analysis, title="Data Analysis", url_path="analysis"),
    st.Page(market_lookup, title="Market Lookup", url_path="market"),
    st.Page(about_model, title="About the Model", url_path="model"),
]
st.navigation(pages, position="top").run()
