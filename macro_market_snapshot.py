"""
Market & macro snapshot - prints the latest values only (no score, no model).

Pulls the latest S&P 500, oil, gold, silver, platinum, rates, credit, volatility,
inflation and labor numbers and prints them in one formatted summary.
Each item tries several data sources in order; the source actually used is shown.

Install:  pip install pandas yfinance pandas-datareader
Educational use only - not investment advice.
Created using Anthropic Claude (internal draft until reviewed and verified by a human).
"""
import contextlib
import datetime as dt
import getpass
import html
import io
import os
import re
import smtplib
import ssl
from email.message import EmailMessage

import pandas as pd
import requests
import yfinance as yf
from pandas_datareader import data as pdr

# ------------------------------------------------------------- email settings
SEND_EMAIL = True                                   # set False to only print the report
EMAIL_FROM = os.environ.get("GMAIL_USER") or "sanjeev.tyagi170@gmail.com"  # sender / SMTP login
EMAIL_TO = os.environ.get("EMAIL_TO") or "sanjeev.tyagi170@gmail.com"
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465        # Gmail SMTP over SSL
# Credentials are NEVER stored in this file. Provide them through environment variables
#   GMAIL_APP_PASSWORD  an App password for the SENDER account (EMAIL_FROM below). It needs
#                       2-Step Verification; your normal Gmail password will not work for SMTP.
#   GMAIL_USER          optional; overrides the sender address (default: EMAIL_FROM)
# If the app password is not set, the script asks for it when it runs.

TODAY = dt.date.today()
MARKET_START = TODAY - dt.timedelta(days=45)      # daily series
MONTHLY_START = TODAY - dt.timedelta(days=800)    # monthly series (CPI needs 13+ months)


# ------------------------------------------------------------------ fetchers
def _clean(series) -> pd.Series:
    """Normalize to a tz-naive, sorted, NaN-free Series."""
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError(f"Expected one series, received {series.shape[1]}.")
        series = series.iloc[:, 0]
    series = series.dropna().copy()
    if series.empty:
        raise RuntimeError("no data returned")
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.sort_index()


def _yahoo(ticker: str) -> pd.Series:
    data = yf.download(ticker, start=MARKET_START, auto_adjust=True,
                       progress=False, threads=False)
    if data.empty:  # second attempt through a different code path
        data = yf.Ticker(ticker).history(start=MARKET_START, auto_adjust=True)
    if data.empty:
        raise RuntimeError("no data returned")
    return _clean(data["Close"])


def _stooq(symbol: str) -> pd.Series:
    return _clean(pdr.DataReader(symbol, "stooq", MARKET_START)["Close"])


def _fred(code: str, start=MARKET_START, yoy: bool = False) -> pd.Series:
    s = pdr.DataReader(code, "fred", start)[code]
    if yoy:  # year-over-year % change of a monthly index
        s = s.pct_change(12, fill_method=None) * 100
    return _clean(s)


HTTP_TIMEOUT = 10
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _single_point(price: float, ts) -> pd.Series:
    """Wrap one live quote as a Series so it fits the same pipeline."""
    ts = pd.Timestamp.now(tz="UTC") if pd.isna(ts) else ts
    return _clean(pd.Series([float(price)], index=[ts]))


def _gold_api(symbol: str) -> pd.Series:
    """Spot metal price (XAU, XAG, XPT) in USD from gold-api.com."""
    r = requests.get(f"https://api.gold-api.com/price/{symbol}",
                     timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
    r.raise_for_status()
    j = r.json()
    ts = pd.to_datetime(j.get("updatedAt"), utc=True, errors="coerce")
    return _single_point(j["price"], ts)


def _swissquote(symbol: str) -> pd.Series:
    """Spot metal price (XAU, XAG, XPT) in USD from Swissquote's public quote feed."""
    url = ("https://forex-data-feed.swissquote.com/public-quotes/bboquotes/"
           f"instrument/{symbol}/USD")
    r = requests.get(url, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
    r.raise_for_status()
    entry = r.json()[0]
    quotes = entry["spreadProfilePrices"]
    q = next((x for x in quotes if x.get("spreadProfile") == "premium"), quotes[0])
    mid = (float(q["bid"]) + float(q["ask"])) / 2
    ts = pd.to_datetime(entry.get("ts"), unit="ms", utc=True, errors="coerce")
    return _single_point(mid, ts)


# ---- RBI policy repo rate (no official free API, so web pages are read) ----
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Last-resort manual value, used ONLY if every web source fails. It is labelled
# "Manual entry" in the output. Update it after each RBI MPC decision, or set to
# None to disable. (rate %, date of the MPC decision)
MANUAL_REPO_RATE = (5.50, "2026-10-07")


def _page_text(url: str) -> str:
    r = requests.get(url, timeout=HTTP_TIMEOUT, headers=BROWSER_HEADERS)
    r.raise_for_status()
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))


def _repo_point(rate: float) -> pd.Series:
    if not 0 < rate < 20:
        raise ValueError(f"implausible repo rate parsed: {rate}")
    # Web pages carry no timestamp, so 'As of' is the time of this run.
    return _single_point(rate, pd.Timestamp.now(tz="UTC"))


def _rbi_site() -> pd.Series:
    """'Current Rates' block on the RBI homepage."""
    text = _page_text("https://www.rbi.org.in/")
    m = re.search(r"Policy\s+Repo\s+Rate\s*:?\s*([\d.]+)\s*%", text, re.I)
    if not m:
        raise RuntimeError("repo rate not found on page (layout may have changed)")
    return _repo_point(float(m.group(1)))


def _tradingeconomics_india() -> pd.Series:
    text = _page_text("https://tradingeconomics.com/india/interest-rate")
    m = re.search(r"last recorded at ([\d.]+) percent", text, re.I)
    if not m:
        raise RuntimeError("repo rate not found on page (layout may have changed)")
    return _repo_point(float(m.group(1)))


def _manual_repo() -> pd.Series:
    if MANUAL_REPO_RATE is None:
        raise RuntimeError("manual fallback disabled")
    rate, date = MANUAL_REPO_RATE
    return _single_point(rate, pd.Timestamp(date, tz="UTC"))


# ---- India CPI inflation (YoY %, all-India combined) ----
# Last-resort manual value, used ONLY if every other source fails. Labelled
# "Manual entry" in the output. Update after each MoSPI release (around the 12th of
# each month), or set to None to disable. (YoY %, month the figure refers to)
MANUAL_INDIA_CPI = (5.54, "2026-09-01")  # MoSPI release of 12-Oct-2026, provisional, base 2024=100


def _india_cpi_tradingeconomics() -> pd.Series:
    text = _page_text("https://tradingeconomics.com/india/inflation-cpi")
    m = re.search(r"inflation rate[^.]{0,80}?(?:to|at)\s+(-?[\d.]+)\s*(?:percent|%)\s+in\s+"
                  r"([A-Za-z]+)(?:\s+of)?\s+(\d{4})", text, re.I)
    if not m:
        raise RuntimeError("inflation figure not found on page (layout may have changed)")
    rate = float(m.group(1))
    month = pd.to_datetime(f"{m.group(2)} {m.group(3)}", format="%B %Y", errors="coerce")
    if pd.isna(month) or not -10 < rate < 40:
        raise ValueError(f"implausible parse: {m.group(0)!r}")
    return _single_point(rate, month)


def _india_cpi_fred() -> pd.Series:
    """OECD India CPI index on FRED -> YoY %. Often lagged, so stale data is rejected."""
    s = _fred("INDCPIALLMINMEI", start=MONTHLY_START, yoy=True)
    age_days = (pd.Timestamp.now() - s.index[-1]).days
    if age_days > 120:
        raise RuntimeError(f"stale series (last observation {s.index[-1]:%Y-%m-%d})")
    return s


def _india_cpi_manual() -> pd.Series:
    if MANUAL_INDIA_CPI is None:
        raise RuntimeError("manual fallback disabled")
    rate, date = MANUAL_INDIA_CPI
    return _single_point(rate, pd.Timestamp(date, tz="UTC"))


# ---- India VIX (NSE volatility index) ----
def _india_vix_nse() -> pd.Series:
    """Latest India VIX from NSE's public index feed (needs a cookie from the homepage)."""
    sess = requests.Session()
    sess.headers.update({**BROWSER_HEADERS,
                         "Accept": "application/json,text/plain,*/*",
                         "Referer": "https://www.nseindia.com/"})
    sess.get("https://www.nseindia.com/", timeout=HTTP_TIMEOUT)  # sets session cookies
    r = sess.get("https://www.nseindia.com/api/allIndices", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    row = next((d for d in payload.get("data", [])
                if "INDIA VIX" in (d.get("indexSymbol"), d.get("index"))), None)
    if row is None:
        raise RuntimeError("INDIA VIX not found in NSE response (format may have changed)")
    value = float(row["last"])
    if not 3 < value < 100:
        raise ValueError(f"implausible India VIX value: {value}")
    # NSE stamps its data in IST; the date is what is displayed.
    ts = pd.to_datetime(payload.get("timestamp"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
    return _single_point(value, ts)


# Source = (label, fetch_function, unit_override_or_None)
def yahoo(t, unit=None):
    return (f"Yahoo {t}", lambda: _yahoo(t), unit)


def stooq(sym, unit=None):
    return (f"Stooq {sym}", lambda: _stooq(sym), unit)


def goldapi(sym, unit=None):
    return (f"gold-api.com {sym}", lambda: _gold_api(sym), unit)


def swissquote(sym, unit=None):
    return (f"Swissquote {sym}/USD", lambda: _swissquote(sym), unit)


def fred(code, unit=None, **kw):
    return (f"FRED {code}", lambda: _fred(code, **kw), unit)


TROY_OZ_PER_KG = 32.1507466  # 1 kilogram = 32.1507466 troy ounces


GRAMS_PER_TROY_OZ = 31.1034768  # 1 troy ounce = 31.1034768 grams


def kg(source):
    """Convert a USD-per-troy-ounce source to USD per kilogram."""
    label, fetch, _ = source
    return (label, lambda: fetch() * TROY_OZ_PER_KG, "$/kg")


def per_10g(source):
    """Convert a USD-per-troy-ounce source to USD per 10 grams."""
    label, fetch, _ = source
    return (label, lambda: fetch() * 10 / GRAMS_PER_TROY_OZ, "$/10 g")


# ---- USD/INR exchange rate and metals in rupees ----
_FX_CACHE = {}


def _fx_point(rate: float, ts=None) -> pd.Series:
    if not 40 < rate < 200:
        raise ValueError(f"implausible USD/INR rate: {rate}")
    return _single_point(rate, ts)


def _fx_open_er_api() -> pd.Series:
    r = requests.get("https://open.er-api.com/v6/latest/USD",
                     timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
    r.raise_for_status()
    j = r.json()
    if j.get("result") != "success":
        raise RuntimeError(f"API returned result={j.get('result')!r}")
    ts = pd.to_datetime(j.get("time_last_update_unix"), unit="s", utc=True, errors="coerce")
    return _fx_point(float(j["rates"]["INR"]), ts)


def _fx_rbi() -> pd.Series:
    """RBI reference rate shown in the homepage 'Current Rates' block."""
    text = _page_text("https://www.rbi.org.in/")
    m = re.search(r"INR\s*/\s*1\s*USD\s*:?\s*([\d.]+)", text, re.I)
    if not m:
        raise RuntimeError("INR/USD not found on page (layout may have changed)")
    return _fx_point(float(m.group(1)))


def _fx_frankfurter() -> pd.Series:
    r = requests.get("https://api.frankfurter.dev/v1/latest",
                     params={"base": "USD", "symbols": "INR"},
                     timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
    r.raise_for_status()
    j = r.json()
    return _fx_point(float(j["rates"]["INR"]), pd.to_datetime(j.get("date"), errors="coerce"))


USDINR_SOURCES = [
    ("open.er-api.com", _fx_open_er_api, None),
    ("RBI reference rate (rbi.org.in)", _fx_rbi, None),
    ("Frankfurter (ECB)", _fx_frankfurter, None),
]

GOLD_USD_OZ = [goldapi("XAU"), swissquote("XAU"), stooq("XAUUSD"), yahoo("GC=F")]
SILVER_USD_OZ = [goldapi("XAG"), swissquote("XAG"), stooq("XAGUSD")]


def _first_working(sources):
    """Return (series, label) from the first (label, fetch, unit) source that works."""
    errors = []
    for label, fetch, _ in sources:
        try:
            return fetch(), label
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def _usd_inr():
    if "fx" not in _FX_CACHE:
        _FX_CACHE["fx"] = _first_working(USDINR_SOURCES)
    return _FX_CACHE["fx"]


def _usd_inr_series() -> pd.Series:
    fx, label = _usd_inr()
    out = fx.copy()
    out.attrs["source"] = label
    return out


def inr_metal(usd_oz_sources, factor):
    """USD/troy-oz metal sources -> INR per chosen unit, using the live USD/INR rate."""
    def fetch():
        usd_oz, metal_label = _first_working(usd_oz_sources)
        fx, fx_label = _usd_inr()
        out = usd_oz * float(fx.iloc[-1]) * factor
        out.attrs["source"] = f"{metal_label} x {fx_label}"
        return out
    return ("USD spot x USD/INR", fetch, None)


# (group, name, default unit, [sources in priority order])
ITEMS = [
    ("EQUITIES", "S&P 500", "index pts", [
        fred("SP500"), stooq("^SPX"), yahoo("SPY", "$ (SPY ETF)")]),

    ("WORLD INDICATORS", "WTI crude oil", "$/bbl", [
        fred("DCOILWTICO")]),
    ("WORLD INDICATORS", "Platinum", "$/oz", [
        goldapi("XPT"), swissquote("XPT"), stooq("XPTUSD"), yahoo("PL=F"),
        yahoo("PPLT", "$ (PPLT ETF)")]),

    ("WORLD INDICATORS", "VIX", "index pts", [fred("VIXCLS")]),

    ("WORLD INDICATORS", "CPI inflation (YoY)", "%", [
        fred("CPIAUCSL", start=MONTHLY_START, yoy=True)]),
    ("WORLD INDICATORS", "Unemployment rate", "%", [
        fred("UNRATE", start=MONTHLY_START)]),

    ("INDIA", "RBI policy repo rate", "%", [
        ("RBI website (rbi.org.in)", _rbi_site, None),
        ("TradingEconomics page", _tradingeconomics_india, None),
        ("Manual entry (verify)", _manual_repo, None)]),
    ("INDIA", "India CPI inflation (YoY)", "%", [
        ("TradingEconomics page", _india_cpi_tradingeconomics, None),
        ("FRED INDCPIALLMINMEI", _india_cpi_fred, None),
        ("Manual entry (verify)", _india_cpi_manual, None)]),
    ("INDIA", "India VIX", "index pts", [
        ("NSE India (nseindia.com)", _india_vix_nse, None),
        yahoo("^INDIAVIX")]),
    ("INDIA", "USD/INR exchange rate", "₹ per $", [
        ("USD/INR", _usd_inr_series, None)]),
    ("INDIA", "Gold (INR)", "₹/10 g", [
        inr_metal(GOLD_USD_OZ, 10 / GRAMS_PER_TROY_OZ)]),
    ("INDIA", "Silver (INR)", "₹/kg", [
        inr_metal(SILVER_USD_OZ, TROY_OZ_PER_KG)]),
]


# ------------------------------------------------------------------- engine
def fetch_item(sources):
    """Try sources in order. Returns (value, as_of, label, unit, failures)."""
    failures = []
    for label, fetch, unit in sources:
        try:
            s = fetch()
            used = s.attrs.get("source", label)  # composite sources name what they used
            return float(s.iloc[-1]), s.index[-1], used, unit, failures
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    return None, None, None, None, failures


def main():
    rows, notes = [], []
    for group, name, default_unit, sources in ITEMS:
        value, as_of, label, unit, failures = fetch_item(sources)
        if failures:
            status = "unavailable" if value is None else f"fell back to {label}"
            notes.append(f"{name} ({status}):\n      " + "\n      ".join(failures))
        rows.append((group, name, value, unit or default_unit, as_of, label))

    width = 98
    print("=" * width)
    print(f"MARKET & MACRO SNAPSHOT   run: {dt.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * width)

    current_group = None
    for group, name, value, unit, as_of, label in rows:
        if group != current_group:
            current_group = group
            print(f"\n{group}")
            print(f"  {'Indicator':<26}{'Latest':>14}  {'Unit':<14}{'As of':<12}Source")
            print("  " + "-" * (width - 2))
        if value is None:
            print(f"  {name:<26}{'n/a':>14}  {'':<14}{'':<12}unavailable")
        else:
            print(f"  {name:<26}{value:>14,.2f}  {unit:<14}{as_of:%Y-%m-%d}  {label}")

    print("\n" + "-" * width)
    print("Notes: FRED daily series typically lag 1-3 days; monthly series (CPI, unemployment)")
    print("are published weeks after the month ends. Metals from gold-api.com / Swissquote are")
    print("live spot quotes at run time. The RBI repo rate is read from a web page with no")
    print("timestamp, so its 'As of' is the run date (a 'Manual entry' source shows the MPC date).")
    print("India CPI is released around the 12th for the prior month; its 'As of' is that month.")
    print("INR gold (per 10 g) and silver (per kg) = USD spot x live USD/INR. They exclude import")
    print("duty and GST, so they will differ from Indian market (MCX / jeweller) prices.")
    print("'As of' is otherwise the date of each latest value.")
    if notes:
        print("\nData-source problems:")
        for n in notes:
            print(f"  - {n}")
    print("\nCreated using Anthropic Claude. Keep on internal versions until a human has")
    print("reviewed and verified the numbers. Share only with people authorized to see the data.")


def send_email(report: str) -> None:
    """Email the report (plain text + monospace HTML) through Gmail SMTP."""
    user = EMAIL_FROM
    password = os.environ.get("GMAIL_APP_PASSWORD")
    try:
        if not password:
            password = getpass.getpass(f"App password for {user} (input hidden): ")
    except (EOFError, OSError):
        raise RuntimeError("Gmail app password not found. Set the GMAIL_APP_PASSWORD "
                           "environment variable.")
    password = password.replace(" ", "")  # Google shows app passwords in 4-character groups

    msg = EmailMessage()
    msg["Subject"] = f"Market & Macro Snapshot - {dt.datetime.now():%d %b %Y %H:%M}"
    msg["From"] = user
    msg["To"] = EMAIL_TO
    msg.set_content(report)
    msg.add_alternative(
        '<pre style="font-family:Menlo,Consolas,monospace;font-size:13px;">'
        f"{html.escape(report)}</pre>", subtype="html")

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(),
                          timeout=30) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)


def run() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    report = buf.getvalue()
    print(report)  # always show the report, even if emailing fails

    if not SEND_EMAIL:
        return
    failed = True
    try:
        send_email(report)
        failed = False
        print(f"Email sent to {EMAIL_TO}")
    except smtplib.SMTPAuthenticationError:
        print("EMAIL FAILED: Gmail rejected the login. Use a Google App password (turn on "
              "2-Step Verification, then create one), not your normal password.")
    except Exception as exc:
        print(f"EMAIL FAILED: {type(exc).__name__}: {exc}\n"
              "If this mentions a certificate, a company network may be intercepting SSL; "
              "if it times out, port 465 may be blocked.")
    if failed and os.environ.get("CI"):  # GitHub Actions sets CI=true
        raise SystemExit(1)


if __name__ == "__main__":
    run()
