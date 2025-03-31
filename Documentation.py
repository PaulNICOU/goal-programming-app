import streamlit as st

# 1) Set page config
st.set_page_config(
    page_title="Quantitative Finance • Documentation",
    page_icon="📖",
)

# 2) Page Header & Intro
st.write("# 📖 Documentation")
st.markdown(
    """
    **Author:** [Paul NICOU](https://www.linkedin.com/in/paul-nicou/) & 
                [Tristan Nercellas](https://www.linkedin.com/in/tristannercellas/)
                
    Welcome to our **Interactive Usage Guide** for the Quantitative Finance app. 
    Expand each step below to see detailed instructions, tips, and screenshots.
    """
)

# 3) Interactive, Expander-Based Tutorial
with st.expander("Step 1: Selecting Assets"):
    st.markdown(
        """
        1. **Open the Sidebar** on the left.
        2. Use the **multiselect** to search and pick your desired tickers 
           (e.g., AAPL, MSFT). Start typing a company name or ticker to filter.
        3. (Optional) Add **custom tickers** by typing them into the 
           “Add custom tickers” box (comma-separated).
        4. Confirm you see all your chosen assets in the “Selected Tickers” field.

        **Tip:** More tickers can mean longer optimization times, 
        but it also gives the solver more diversification choices.
        """
    )
    st.image(
        "images/select_assets.png",
        caption="A sample screenshot for selecting assets",
        width=400
    )

with st.expander("Step 2: Choosing the Date Range"):
    st.markdown(
        """
        1. Still in the sidebar, find the **start date** and **end date** pickers.
        2. Click each date field to open a calendar and pick your range 
           (e.g., from 2015-01-01 to 2019-07-05).
        3. We recommend a multi-year window to capture meaningful market conditions.

        **Tip:** Very narrow date ranges can produce sparse data 
        or lead to infeasible solutions.
        """
    )
    st.image(
        "images/date_range.png",
        caption="Choosing a date range",
        width=400
    )

with st.expander("Step 3: Picking the Optimization Model"):
    st.markdown(
        """
        1. Select **Original**, **Minimize Risk**, or **Goal Programming** in the sidebar radio buttons.
        2. Notice the **snapshot** that describes the chosen model's objective and constraints.
        3. If unsure, try **Original** first, then experiment with the others 
           to see how your portfolio allocations change.

        **Tip:** Each model suits different goals. 
        “Minimize Risk” heavily reduces volatility, 
        while “Goal Programming” tries to match specific risk/return targets.
        """
    )
    st.image(
        "images/select_model.png",
        caption="Selecting the optimization model",
        width=400
    )

with st.expander("Step 4: Running the Solver"):
    st.markdown(
        """
        1. Click **Run Optimization** (typically near the bottom of the sidebar).
        2. The app fetches historical prices from Yahoo Finance 
           and runs the solver (this may take a few seconds).
        3. If constraints are too strict or data is missing, you may see “No feasible solution found.”
           Otherwise, the main page shows your final portfolio.

        **Tip:** If data is unavailable (e.g., a delisted ticker), 
        the app might fill placeholders or show warnings. 
        Double-check your ticker symbols and date range if this happens.
        """
    )
    st.image(
        "images/run_button.png",
        caption="Clicking the run button to start optimization",
        width=400
    )

with st.expander("Step 5: Interpreting Results & Charts"):
    st.markdown(
        """
        After the solver finishes, the main area displays:

        - **Solver Status & Key Metrics**: e.g., 
          “**OPTIMAL**,” “Portfolio Risk = 0.018,” “Portfolio Return = 0.15,” etc.
        - **Portfolio Weights**: A list or table showing each asset’s weight 
          (e.g., AAPL: 0.30, MSFT: 0.25, etc.).
        - **Allocation Bar Chart**: Hover over bars to see exact percentages.
        - **Monte Carlo Distribution**: A histogram simulating 
          random daily returns over the chosen horizon. 
          Vertical lines often highlight mean, 5th percentile, 95th percentile, etc.
        - **Drawdown & Cumulative Return Plots**: Interactive line charts 
          showing how the portfolio’s value might evolve or drop from peak 
          (with a “Stressed” version if a shock is applied).
        """
    )
    
    st.image(
        "images/allocation_bar.png",
        caption="Allocation bar chart example",
        width=400
    )
    
    st.markdown(
        """
        **Tip:** All charts are interactive. 
        Hover to see tooltips with exact values, zoom/pan around, 
        or export them by clicking the Plotly toolbar icons.
        """
    )

with st.expander("Step 6: Tweaking & Experimenting"):
    st.markdown(
        """
        1. **Change Tickers**: Replace or add new ones to see how the solver adjusts allocations.
        2. **Change Date Range**: Focus on a recession period (e.g., 2008) to see the solver 
           handle more volatile data, or pick a bullish period for different results.
        3. **Switch Models**: Try “Minimize Risk” after “Original” 
           to see how volatility drastically changes. 
           Or “Goal Programming” if you want to target a specific risk/return combo.
        
        By experimenting, you’ll see how each model responds to varying input parameters 
        and historical price data.
        """
    )


