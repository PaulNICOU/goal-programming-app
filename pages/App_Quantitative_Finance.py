import os  # We'll set the environment variable for Mosek if you have a license.
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import time  

os.environ["MOSEKLM_LICENSE_FILE"] = "../mosek.lic"

################################################################################
# 1) MODEL PARAMETERS & GLOBAL CONSTANTS
################################################################################
SCENARIO_PROB = [0.3, 0.5, 0.2]
SCENARIO_MULTIPLIERS = [1.2, 1.0, 0.8]  # Boom, Normal, Recession
ALPHA = 0.1
EPSILON = 1e-6

MAX_RISK_ORIGINAL = 0.02
MAX_ALLOCATION_ORIGINAL = 0.3

# We'll also define relaxed parameters for the Original model's fallback solutions:
RELAXED_RISK = 0.05
RELAXED_ALLOCATION = 0.4

VERY_RELAXED_RISK = 0.20
VERY_RELAXED_ALLOCATION = 1.0

################################################################################
# 2) PARTIAL S&P 500 TICKERS & MODEL DESCRIPTIONS
################################################################################
SP500_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMZN", "AXP", "BA", "BAC", "BKNG",
    "BRK.B", "C", "CAT", "CRM", "CVX", "DIS", "GE", "GOOGL", "GOOG", "GS",
    "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "LIN", "LLY", "LOW", "MA",
    "META", "MCD", "MMM", "MRK", "MS", "MSFT", "NFLX", "NKE", "NVDA", "ORCL",
    "PEP", "PFE", "PG", "PYPL", "T", "TGT", "TSLA", "UNH", "V", "VZ",
    "WFC", "WMT", "XOM"
]

MODEL_DESCRIPTIONS = {
    "Original": """
        **Original Model**  
        - Uses a scenario-based objective with probabilities and multipliers:
        - We sum up expected `returns * scenario multipliers * scenario probabilities`.
        - Key constraints:
            - If not feasible, we try *relaxed* / *more relaxed* / *penalty* approach.
    """,
    "Minimize Risk": """
        **Minimize Risk Model**  
        - Strictly minimize portfolio variance (i.e. $x^T Σ x$).
        - Constraints:
            - Sum of weights = **1**
            - All weights >= **0**
            - Use this if the main objective is to reduce overall volatility.
    """,
    "Goal Programming": """
        **Goal Programming Model**
        - Try to keep risk near **0.02** and annual return near **0.15**.
        - Minimizes the sum of deviations from these two goals (like a multi-criteria approach).
        - Constraints:
            - Sum of weights = **1**, all weights >= **0**
    """
}

################################################################################
# 3) STREAMLIT APP CONFIG
################################################################################
st.set_page_config(
    page_title="Quantitative Finance • App",
    page_icon="📊",
)

st.write("# 📊 Quantitative Finance")
st.markdown(
    """
    **Authors:** [Paul NICOU](https://www.linkedin.com/in/paul-nicou/) & [Tristan Nercellas](https://www.linkedin.com/in/tristannercellas/)
    """
)

################################################################################
# 4) SIDEBAR: MODEL SELECTION & PARAMETERS
################################################################################
# --- Assets Section ---
st.sidebar.header("Assets")
selected_sp500 = st.sidebar.multiselect(
    "Pick assets (S&P 500 partial list, type to filter)",
    SP500_TICKERS,
    default=["AAPL", "MSFT"],
    key="selected_sp500",
)

with st.sidebar.expander("Advanced assets search", icon=":material/search:"):
    custom_input = st.text_input(
        "Advanced ticker(s) research (comma-delimited)",
        value="",
        key="custom_input",
    )
    # Split input and transform into ticker list
    custom_list = [x.strip().upper() for x in custom_input.split(",") if x.strip()]
    
    # Open the local CSV file in binary mode
    with open("assets_template.csv", "rb") as file:
        csv_data = file.read()

    # Create a download button that serves the file content
    st.download_button(
        label="Download CSV File template",
        data=csv_data,
        file_name="assets_template.csv",
        mime="text/csv",
        icon=":material/download:"
    )
    uploaded_file = st.file_uploader("Upload your CSV file", key="uploaded_file", type="csv", accept_multiple_files=False)
    
    csv_tickers = []
    if uploaded_file is not None:
        try:
            uploaded_file_data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
        else:
            # Check if the CSV contains the "Ticker" column and at least one row
            if "Ticker" not in uploaded_file_data.columns:
                st.error("CSV file does not contain the required column: 'Ticker'.")
            elif uploaded_file_data.empty:
                st.error("CSV file is empty. Please provide at least one row of data.")
            else:
                st.caption(f"Imported tickers from **{uploaded_file.name}** file")
                st.dataframe(uploaded_file_data, hide_index=True)
                csv_tickers = uploaded_file_data["Ticker"].dropna().astype(str).str.upper().tolist()

# Combine assets from selections and make sure to return a unique list
assets = list(set(selected_sp500 + custom_list + csv_tickers))

st.sidebar.write("## Time range")

start_date_input = st.sidebar.date_input(
    "From",
    date(2015, 1, 1),
    min_value=date(2000, 1, 1),
    max_value=date.today(),

    format="YYYY-MM-DD"
)
end_date_input = st.sidebar.date_input(
    "To",
    date(2019, 7, 5),
    min_value=date(2000, 1, 2),
    max_value=date.today(),
    format="YYYY-MM-DD"
)

st.sidebar.write("## Model")

model_choice = st.sidebar.radio(
    "Select Optimization Model",
    ["Original", "Minimize Risk", "Goal Programming"],
    index=0
)

max_risk_original_input = MAX_RISK_ORIGINAL
max_single_asset_input = MAX_ALLOCATION_ORIGINAL
if model_choice == "Original":
    st.sidebar.markdown(MODEL_DESCRIPTIONS[model_choice])
    max_risk_original_input = st.sidebar.number_input("Risk <=", value=MAX_RISK_ORIGINAL, placeholder="Type the maximum Risk", min_value=0.01, max_value=0.05, step=0.01)
    max_single_asset_input = st.sidebar.number_input("Max single-asset <=", value=MAX_ALLOCATION_ORIGINAL, placeholder="Max single-asset", min_value=0.1, max_value=0.5, step=0.05)
    st.sidebar.markdown("")
else:
    st.sidebar.markdown(MODEL_DESCRIPTIONS[model_choice])
    max_risk_original_input = MAX_RISK_ORIGINAL
    max_single_asset_input = MAX_ALLOCATION_ORIGINAL

run_button = st.sidebar.button("Run Optimization")

################################################################################
# 5) HELPER FUNCTIONS FOR DATA & METRICS
################################################################################
def load_data(assets: list[str], start_date: date, end_date: date, progress_bar):
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Step #1
    progress_bar.progress(0)
    progress_text.text(f"Fetching data... {0}%")
    time.sleep(0.5)
    raw_data = yf.download(assets, start=start_date_str, end=end_date_str, auto_adjust=True)
    progress_bar.progress(25)
    progress_text.text(f"Fetching data... {25}%")
    time.sleep(0.75)
    
    # Step #2
    if raw_data.empty:
        st.warning("Yahoo Finance returned no data. Using constant 100 as placeholder.")
        data = pd.DataFrame(
            100,
            index=pd.date_range(start_date_str, end_date_str, freq="B"),
            columns=assets
        )
    else:
        if "Close" not in raw_data.columns:
            st.error("No 'Close' column in data; cannot proceed.")
            return None, None, None, None
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data["Close"].copy()
        else:
            data = raw_data[["Close"]].copy()
        data.columns = assets

    data = data.ffill().bfill()
    progress_bar.progress(50)
    progress_text.text(f"Generating Returns, Expected Returns, and Covariance Matrix... {50}%")
    time.sleep(1)
    
    if data.isnull().any().any():
        for col in data.columns:
            data[col] = data[col].fillna(data[col].mean())

    returns = data.pct_change().dropna()
    if returns.empty:
        st.warning("No valid returns; setting returns to zero.")
        returns = pd.DataFrame(0, index=data.index, columns=data.columns)

    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    if cov_matrix.isnull().values.any() or not np.isfinite(cov_matrix.values).all():
        st.warning("Covariance invalid; filling with zeros.")
        cov_matrix = cov_matrix.fillna(0)
    if np.allclose(cov_matrix.values, 0.0, atol=1e-12):
        st.warning("Covariance near zero. Adding small epsilon.")
        cov_matrix = pd.DataFrame(
            np.eye(len(assets)) * 1e-6,
            index=assets,
            columns=assets
        )
        
    progress_bar.progress(75)
    progress_text.text(f"Optimizing... {75}%")
    time.sleep(1.5)
        
    progress_bar.progress(100)
    progress_text.text(f"Optimizing... {100}%")
    time.sleep(0.5)
    
    return data, returns, expected_returns, cov_matrix

def max_drawdown(returns_series):
    """Compute the maximum drawdown for a returns series."""
    if returns_series is None or returns_series.empty:
        return np.nan
    cum_ret = (1 + returns_series).cumprod()
    running_max = cum_ret.cummax()
    drawdown = cum_ret / running_max - 1
    return drawdown.min()

def compute_portfolio_metrics(weights, returns_df):
    """
    Basic metrics from daily returns of the final portfolio:
      - Annual Return
      - Annual Volatility
      - Sharpe Ratio
      - Max Drawdown
    """
    pf_returns = (returns_df * weights).sum(axis=1)
    if pf_returns.empty:
        return {
            "Annual Return": np.nan,
            "Annual Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan
        }
    daily_mean = pf_returns.mean()
    daily_std = pf_returns.std()
    ann_ret = daily_mean * 252
    ann_vol = daily_std * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    mdd = max_drawdown(pf_returns)
    return {
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd
    }

def compute_drawdown_and_cumret(weights, returns_df):
    pf_returns = (returns_df * weights).sum(axis=1)
    cum_ret = (1 + pf_returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = cum_ret / running_max - 1
    return pf_returns, cum_ret, drawdown

################################################################################
# 6) ORIGINAL MODEL: MULTIPLE SOLUTIONS (ORIGINAL, RELAXED, VERY RELAXED, PENALTY)
################################################################################
def solve_original_portfolio(
    max_risk, max_allocation, alpha,
    expected_returns, cov_matrix, assets
):
    x = cp.Variable(len(assets))
    risk_expr = cp.quad_form(x, cov_matrix.values)

    constraints_ = [
        cp.sum(x) == 1,
        x >= 0,
        x <= max_allocation
    ]

    # Scenario-based return
    stoch_expr = 0
    for p, mult in zip(SCENARIO_PROB, SCENARIO_MULTIPLIERS):
        stoch_expr += p * cp.sum(cp.multiply(expected_returns.values * mult, x))

    # Slack for risk 
    d_risk = cp.Variable(nonneg=True)

    constraints_ += [
        risk_expr - d_risk <= max_risk,
    ]

    obj = cp.Maximize(stoch_expr - alpha * d_risk + alpha)
    prob = cp.Problem(obj, constraints_)
    result_ = prob.solve()  # or solver=cp.MOSEK if you prefer

    return {
        "status": prob.status,
        "objective": result_,
        "stochastic_return": stoch_expr.value,
        "risk_obj": risk_expr.value,
        "x_value": x.value
    }

def solve_penalty_approach(
    expected_returns, cov_matrix, assets,
    max_risk=max_risk_original_input,
    big_penalty=1000.0
):
    x_var = cp.Variable(len(assets), nonneg=True)
    risk_expr = cp.quad_form(x_var, cov_matrix.values)

    slack_sum = cp.Variable(nonneg=True)
    slack_risk = cp.Variable(nonneg=True)
    slack_liq = cp.Variable(nonneg=True)

    constraints_penalty = [
        cp.sum(x_var) + slack_sum == 1,
        x_var <= 1,
        risk_expr <= max_risk + slack_risk
    ]

    stoch_expr = 0
    for p, mult in zip(SCENARIO_PROB, SCENARIO_MULTIPLIERS):
        stoch_expr += p * cp.sum(cp.multiply(expected_returns.values * mult, x_var))

    final_obj = cp.Maximize(stoch_expr - big_penalty * (slack_sum + slack_risk + slack_liq))
    penalty_prob = cp.Problem(final_obj, constraints_penalty)
    penalty_result = penalty_prob.solve()

    return {
        "status": penalty_prob.status,
        "objective": penalty_result,
        "stochastic_return": stoch_expr.value,
        "risk_obj": risk_expr.value,
        "x_value": x_var.value
    }

def compare_solutions(expected_returns, cov_matrix, assets):
    solutions = {}

    sol_original = solve_original_portfolio(
        max_risk=max_risk_original_input,
        max_allocation=max_single_asset_input,
        alpha=ALPHA,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        assets=assets
    )
    sol_relaxed = solve_original_portfolio(
        max_risk=RELAXED_RISK,
        max_allocation=RELAXED_ALLOCATION,
        alpha=ALPHA,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        assets=assets
    )
    sol_very_relaxed = solve_original_portfolio(
        max_risk=VERY_RELAXED_RISK,
        max_allocation=VERY_RELAXED_ALLOCATION,
        alpha=ALPHA,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        assets=assets
    )
    sol_penalty = solve_penalty_approach(
        expected_returns, cov_matrix, assets,
        max_risk=max_risk_original_input
    )

    solutions["Original"] = sol_original
    solutions["Relaxed"] = sol_relaxed
    solutions["Very Relaxed"] = sol_very_relaxed
    solutions["Penalty"] = sol_penalty

    feasible_order = ["Original", "Relaxed", "Very Relaxed", "Penalty"]
    final_label = None
    final_sol = None
    for label in feasible_order:
        sol = solutions[label]
        if sol["x_value"] is not None and sol["status"] in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            final_label = label
            final_sol = sol
            break

    return solutions, final_label, final_sol

################################################################################
# 7) MINIMIZE RISK MODEL
################################################################################
def solve_minimize_risk(expected_returns, cov_matrix, assets):
    x = cp.Variable(len(assets))
    risk_expr = cp.quad_form(x, cov_matrix.values)
    constraints_ = [cp.sum(x) == 1, x >= EPSILON]
    obj = cp.Minimize(risk_expr)
    prob = cp.Problem(obj, constraints_)
    result_ = prob.solve()
    return {
        "status": prob.status,
        "objective": result_,
        "risk_value": risk_expr.value,
        "x_value": x.value
    }

################################################################################
# 8) GOAL PROGRAMMING MODEL
################################################################################
def solve_goal_programming(expected_returns, cov_matrix, assets):
    x = cp.Variable(len(assets))
    risk_expr = cp.quad_form(x, cov_matrix.values)
    ret_expr = cp.sum(cp.multiply(expected_returns.values, x))

    target_risk = 0.02
    target_return = 0.15

    dev_risk = cp.pos(risk_expr - target_risk)
    dev_return = cp.pos(target_return - ret_expr)

    constraints_ = [cp.sum(x) == 1, x >= EPSILON]
    obj = cp.Minimize(0.5 * dev_risk + 0.5 * dev_return)
    prob = cp.Problem(obj, constraints_)
    result_ = prob.solve()

    return {
        "status": prob.status,
        "objective": result_,
        "risk_value": risk_expr.value,
        "return_value": ret_expr.value,
        "x_value": x.value
    }

################################################################################
# 9) MONTE CARLO & HELPER VISUALIZATIONS
################################################################################
def monte_carlo_simulation(x_val, expected_returns, cov_matrix, num_simulations=1000, horizon=252):
    daily_mean = expected_returns / 252
    daily_cov = cov_matrix / 252
    portfolio_returns = []
    for _ in range(num_simulations):
        sim_daily = np.random.multivariate_normal(daily_mean, daily_cov, horizon)
        cumulative_return = np.prod(1 + sim_daily @ x_val) - 1
        portfolio_returns.append(cumulative_return)
        
    return portfolio_returns


def stress_test(returns_series, shock_multiplier=1.0):
    if returns_series is None or returns_series.empty:
        return None
    std_dev = returns_series.std()
    
    return returns_series - shock_multiplier * std_dev


################################################################################
# 10) MAIN STREAMLIT LOGIC
################################################################################

# We'll store final results in st.session_state for retrieval after the progress bar
if "opt_results" not in st.session_state:
    st.session_state["opt_results"] = None

if run_button:
    if not assets:
        st.warning("No assets selected. Please pick at least one asset.")
    else:
        # 1) Start the progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # 2) Remove / hide the progress bar by re-initializing with 0
        progress_bar.empty()
        progress_text.empty()

        # 3) Perform the actual computations (but do not show them yet)
        data, returns_df, exp_returns, cov_mat = load_data(
            assets=assets,
            start_date=start_date_input,
            end_date=end_date_input,
            progress_bar=progress_bar
        )
        
        # 4) Remove / hide the progress bar by re-initializing with 0
        progress_bar.empty()
        progress_text.empty()
        
        # 5) Indicate success and prompt
        st.success("Optimization complete!")  

        # In case data is empty or invalid
        if data is None or returns_df is None or exp_returns is None or cov_mat is None:
            st.warning("Data invalid or empty. Please check your tickers.")
            st.session_state["opt_results"] = None
        else:
            # We'll store everything we need to display in session state
            st.session_state["opt_results"] = {
                "model_choice": model_choice,
                "data": data,
                "returns_df": returns_df,
                "exp_returns": exp_returns,
                "cov_mat": cov_mat
            }
else:
    st.info("*No optimization yet. Please run the optimization...*")


# 5) Provide a button to reveal the analysis
if st.session_state["opt_results"] is not None:
    # Now we retrieve the data
    model_choice = st.session_state["opt_results"]["model_choice"]
    data = st.session_state["opt_results"]["data"]
    returns_df = st.session_state["opt_results"]["returns_df"]
    exp_returns = st.session_state["opt_results"]["exp_returns"]
    cov_mat = st.session_state["opt_results"]["cov_mat"]

    ####################################################################
    # RUN THE SELECTED MODEL & DISPLAY THE RESULTS
    ####################################################################
    st.write(f"## Running the `{model_choice}` Model")

    if model_choice == "Original":
        st.markdown(f"""
            ### Analysis: Original Model
            In this model, we attempt to:
            1. **Maximize** scenario-based returns:

            $$\\sum_{{\\text{{scenario}}}} \\text{{prob}} * \\Big( x^T \\mu * \\text{{multiplier}} \\Big)$$
            
            2. **Constrain** risk <= **{max_risk_original_input}** and single-asset <= **{max_single_asset_input}**.
            3. If that fails, we automatically try "*Relaxed*" constraints, then "*Very Relaxed*", 
            and finally a "*Penalty*" approach ensuring we get **some** feasible solution.

            **Next**: We'll show which solution was chosen and how it performs.
        """)

        solutions_dict, final_label, final_sol = compare_solutions(exp_returns, cov_mat, assets)
        
        if final_sol is None:
            st.error("No feasible solution found, even in penalty approach.")
        else:
            st.write(f"**Chosen solution**: {final_label}")
            st.write(f"**Solver status**: {final_sol['status']}")
            x_val = final_sol["x_value"]

            # Summarize each approach in a table
            rows = []
            for lbl, sol in solutions_dict.items():
                if sol["x_value"] is not None:
                    rows.append({
                        "Method": lbl,
                        "Status": sol["status"],
                        "Objective": f"{sol['objective']:.4f}" if sol["objective"] is not None else None,
                        "Risk": f"{sol['risk_obj']:.6f}" if "risk_obj" in sol else None
                    })
                else:
                    rows.append({
                        "Method": lbl,
                        "Status": sol["status"],
                        "Objective": None,
                        "Risk": None
                    })
            df_comp = pd.DataFrame(rows)
            st.divider()
            st.write("### Comparison of all Solutions attempted:")
            st.dataframe(df_comp, hide_index=True)

            # Show final solution's risk if present
            if "risk_obj" in final_sol:
                st.write(f"**Final risk:** {final_sol['risk_obj']:.6f}")
            if "stochastic_return" in final_sol:
                st.write(f"**Stochastic return (Scenario-based)**: {final_sol['stochastic_return']:.6f}")

            st.subheader("Final portfolio weights")
            for a, wgt in zip(assets, x_val):
                st.write(f"{a}: {wgt:.4f}")

            df_alloc = pd.DataFrame({"Asset": assets, "Weight": x_val})
            fig_alloc = px.bar(df_alloc, x="Asset", y="Weight", title=f"Portfolio Allocation [{final_label}]")
            fig_alloc.update_layout(xaxis_title="Assets", yaxis_title="Allocation (%)", yaxis=dict(tickformat=".2%"))
            st.plotly_chart(fig_alloc, use_container_width=True)

            # Compute daily portfolio returns
            pf_returns = (returns_df * x_val).sum(axis=1)

            st.divider()
            st.write("### Performance Metrics (Daily Returns → Annualized)")
            metrics = compute_portfolio_metrics(x_val, returns_df)
            df_metrics = pd.DataFrame([metrics])
            
            # Ensure the columns are floats and convert to percentage
            df_metrics["Annual Return"] = df_metrics["Annual Return"] * 100
            df_metrics["Annual Volatility"] = df_metrics["Annual Volatility"] * 100
            df_metrics["Max Drawdown"] = df_metrics["Max Drawdown"] * 100

            st.dataframe(df_metrics, hide_index=True, column_config={
                "Annual Return": st.column_config.NumberColumn(format="%.2f %%"),
                "Annual Volatility": st.column_config.NumberColumn(format="%.2f %%"),
                "Max Drawdown": st.column_config.NumberColumn(format="%.2f %%"),
            })

            # Cumulative returns & drawdowns
            pf_cum = (1 + pf_returns).cumprod()
            pf_draw = (pf_cum / pf_cum.cummax()) - 1
            df_line = pd.DataFrame({"Date": pf_cum.index, "Cumulative": pf_cum.values, "Drawdown": pf_draw.values})

            fig_cum = px.line(df_line, x="Date", y="Cumulative", title="Cumulative returns (final solution)")
            # fig_cum.update_yaxes(tickformat=".2%") # Not sure about this one
            st.plotly_chart(fig_cum, use_container_width=True)

            fig_dd = px.line(df_line, x="Date", y="Drawdown", title="Drawdown over time")
            fig_dd.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig_dd, use_container_width=True)

            st.divider()
            st.write("### Monte Carlo distribution of cumulative returns")
            mc_returns = monte_carlo_simulation(x_val, exp_returns, cov_mat)
            mean_ret = np.mean(mc_returns)
            std_ret = np.std(mc_returns)
            percentiles = np.percentile(mc_returns, [5, 50, 95])
            st.write(f"**Mean:** {mean_ret:.2%}, **Std Dev:** {std_ret:.2%}, **5th:** {percentiles[0]:.2%}, **95th:** {percentiles[2]:.2%}")

            fig_mc = px.histogram(mc_returns, nbins=30,
                title="Monte Carlo simulation of final portfolio returns"
            )
            fig_mc.add_vline(x=mean_ret, line_color="red", line_dash="dash",
                                annotation_text=f"Mean={mean_ret:.2%}", annotation_position="top left")
            fig_mc.add_vline(x=percentiles[0], line_color="black", line_dash="dot",
                                annotation_text=f"5%={percentiles[0]:.2%}", annotation_position="top left")
            fig_mc.add_vline(x=percentiles[1], line_color="purple", line_dash="dot",
                                annotation_text=f"Median={percentiles[1]:.2%}", annotation_position="bottom right")
            fig_mc.add_vline(x=percentiles[2], line_color="green", line_dash="dot",
                                annotation_text=f"95%={percentiles[2]:.2%}", annotation_position="top right")
            st.plotly_chart(fig_mc, use_container_width=True)

    elif model_choice == "Minimize Risk":
        st.markdown("""
        ### Analysis: Minimize Risk Model
        This model aims to strictly minimize the portfolio variance 
        $$(x^T \\Sigma x$$), subject to:
        - Weights >= 0 (long-only)
        - Sum of weights = 1 (fully invested)
        
        By ignoring return constraints, it focuses purely on obtaining 
        the **lowest possible volatility**. 
        Often, such a portfolio is heavily skewed toward lower-volatility 
        assets. Let's see the results:
        """)

        sol_minrisk = solve_minimize_risk(exp_returns, cov_mat, assets)
        if sol_minrisk["x_value"] is None:
            st.error("No feasible solution found for Minimize Risk model.")
        else:
            x_val = sol_minrisk["x_value"]
            st.write(f"**Solver status**: {sol_minrisk['status']}")
            st.write(f"**Objective** (Portfolio Variance): {sol_minrisk['risk_value']:.6f}")
            
            st.subheader("Portfolio Weights")
            for a, wgt in zip(assets, x_val):
                st.write(f"{a}: {wgt:.4f}")

            df_alloc = pd.DataFrame({"Asset": assets, "Weight": x_val})
            fig_alloc = px.bar(df_alloc, x="Asset", y="Weight", title="Min-Risk Portfolio Allocation")
            fig_alloc.update_layout(xaxis_title="Assets", yaxis_title="Allocation (%)", yaxis=dict(tickformat=".2%"))
            st.plotly_chart(fig_alloc, use_container_width=True)

            pf_returns = (returns_df * x_val).sum(axis=1)
            pf_cum = (1 + pf_returns).cumprod()
            pf_draw = (pf_cum / pf_cum.cummax()) - 1

            st.divider()
            st.write("### Performance Metrics")
            metrics = compute_portfolio_metrics(x_val, returns_df)
            df_metrics = pd.DataFrame([metrics])
            st.dataframe(df_metrics, hide_index=True)

            df_line = pd.DataFrame({"Date": pf_cum.index, "Cumulative": pf_cum.values, "Drawdown": pf_draw.values})
            fig_cum = px.line(df_line, x="Date", y="Cumulative", title="Cumulative feturns (min-risk)")
            st.plotly_chart(fig_cum, use_container_width=True)

            fig_dd = px.line(df_line, x="Date", y="Drawdown", title="Drawdown (min-risk)")
            st.plotly_chart(fig_dd, use_container_width=True)

    else:  # "Goal Programming"
        st.markdown("""
        ### Analysis: Goal Programming Model
        We define two major targets:
        - Risk near 0.02
        - Return near 0.15
        
        If the portfolio's risk is above 0.02 or return is below 0.15, 
        we incur slack (deviations). We minimize the sum of these deviations.
        
        This approach attempts to *balance multiple objectives*, 
        rather than purely focusing on either returns or risk alone.
        """)

        sol_goal = solve_goal_programming(exp_returns, cov_mat, assets)
        if sol_goal["x_value"] is None:
            st.error("No feasible solution found for Goal Programming model.")
        else:
            x_val = sol_goal["x_value"]
            st.write(f"**Solver status**: {sol_goal['status']}")
            st.write(f"**Objective** (sum of deviations): {sol_goal['objective']:.6f}")
            st.write(f"**Final Risk**: {sol_goal['risk_value']:.6f}")
            st.write(f"**Final Return**: {sol_goal['return_value']:.6f}")

            st.subheader("Portfolio Weights")
            for a, wgt in zip(assets, x_val):
                st.write(f"{a}: {wgt:.4f}")

            df_alloc = pd.DataFrame({"Asset": assets, "Weight": x_val})
            fig_alloc = px.bar(df_alloc, x="Asset", y="Weight", title="Goal Programming Allocation")
            fig_alloc.update_layout(xaxis_title="Assets", yaxis_title="Allocation (%)", yaxis=dict(tickformat=".2%"))
            st.plotly_chart(fig_alloc, use_container_width=True)

            pf_returns = (returns_df * x_val).sum(axis=1)
            pf_cum = (1 + pf_returns).cumprod()
            pf_draw = (pf_cum / pf_cum.cummax()) - 1

            st.divider()
            st.write("### Performance Metrics")
            metrics = compute_portfolio_metrics(x_val, returns_df)
            df_metrics = pd.DataFrame([metrics])
            st.dataframe(df_metrics, hide_index=True)

            df_line = pd.DataFrame({"Date": pf_cum.index, "Cumulative": pf_cum.values, "Drawdown": pf_draw.values})
            fig_cum = px.line(df_line, x="Date", y="Cumulative", title="Cumulative Returns (Goal Programming)")
            st.plotly_chart(fig_cum, use_container_width=True)

            fig_dd = px.line(df_line, x="Date", y="Drawdown", title="Drawdown (Goal Programming)")
            st.plotly_chart(fig_dd, use_container_width=True)