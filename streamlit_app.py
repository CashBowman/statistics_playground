import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

# Page Configuration
st.set_page_config(layout="wide")

# Main Menu
st.sidebar.title("Statistical Demonstrations")
demo = st.sidebar.radio("Select Demonstration", [
    "Confidence Interval Visualizer",
    "Monte Carlo π Simulator",
    "Central Limit Theorem Demo"
])

# Common Components
def reset_button():
    if st.sidebar.button("Reset Current Demo"):
        st.session_state.clear()
        st.rerun()

# Demonstration 1: Binary Proportion Confidence Intervals (Normal Approximation only)
if demo == "Confidence Interval Visualizer":
    st.title("Proportion Confidence Interval Simulation (Normal Approximation)")
    st.markdown("""
    **Visualize how 95% confidence intervals behave for binary proportions:**
    - Each line represents one sample's confidence interval
    - Green intervals contain the true proportion (p)
    - Red intervals miss the true proportion
    - Expected coverage: ~95% of intervals contain p
    """)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Number of Samples", 10, 200, 50)
        sample_size = st.slider("Sample Size", 10, 1000, 100)
    with col2:
        true_p = st.slider("True Proportion (p)", 0.01, 0.99, 0.5)
    
    # Simulation
    if st.button("Run Simulation"):
        plt.figure(figsize=(10, 6))
        plt.axvline(true_p, color='black', linestyle='--', linewidth=1)
        
        contained = 0
        for i in range(n_samples):
            # Generate binary sample
            sample = np.random.binomial(1, true_p, sample_size)
            successes = sample.sum()
            
            # Calculate normal approximation CI
            ci_low, ci_upp = proportion_confint(successes, sample_size, alpha=0.05, method='normal')
            
            # Clip to [0,1] for plotting
            ci_low = max(0, ci_low)
            ci_upp = min(1, ci_upp)
            
            # Check coverage and plot
            if ci_low <= true_p <= ci_upp:
                plt.plot([ci_low, ci_upp], [i, i], color='green', alpha=0.7, solid_capstyle='round')
            else:
                plt.plot([ci_low, ci_upp], [i, i], color='red', alpha=0.7, solid_capstyle='round')
        
        plt.title(f"95% Confidence Intervals for Proportion (p = {true_p})")
        plt.xlabel("Proportion Value")
        plt.ylabel("Sample Index")
        plt.xlim(max(0, true_p - 0.2), min(1, true_p + 0.2))
        st.pyplot(plt)
        plt.clf()
    
    reset_button()

# Demonstration 2: Monte Carlo π
elif demo == "Monte Carlo π Simulator":
    st.title("Monte Carlo π Approximation")
    st.markdown("""
    **How it works:**
    - We generate random points in a 1×1 square
    - Calculate distance from origin: √(x² + y²)
    - Points within radius 1 (area = π/4) are counted
    - π ≈ 4 × (points_inside / total_points)
    """)
    
    # Initialize session state
    if 'mc_points' not in st.session_state:
        st.session_state.mc_points = np.zeros((0, 3))
    if 'mc_running' not in st.session_state:
        st.session_state.mc_running = False
    if 'mc_iter' not in st.session_state:
        st.session_state.mc_iter = 0

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        pts_per_step = st.slider("Points/Iteration", 1, 100, 10)
    with col2:
        if st.button("Start Simulation"):
            st.session_state.mc_running = True
        if st.button("Stop Simulation"):
            st.session_state.mc_running = False

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Circle((0, 0), 1, color='blue', alpha=0.1))
    scatter = ax.scatter([], [], s=10, c=[], cmap='viridis', alpha=0.6)
    
    plot_ph = st.empty()
    text_ph = st.empty()

    # Simulation loop
    while st.session_state.mc_running:
        # Generate points
        new_pts = np.random.rand(pts_per_step, 2)
        dists = np.sqrt(new_pts[:,0]**2 + new_pts[:,1]**2)
        in_circle = (dists <= 1).astype(int)
        
        # Store with iteration
        iterations = np.full((pts_per_step, 1), st.session_state.mc_iter)
        new_pts = np.hstack([new_pts, iterations])
        st.session_state.mc_points = np.vstack([st.session_state.mc_points, new_pts])
        st.session_state.mc_iter += 1
        
        # Update plot
        if len(st.session_state.mc_points) > 0:
            scatter.set_offsets(st.session_state.mc_points[:, :2])
            scatter.set_array(st.session_state.mc_points[:, 2])
            scatter.set_clim(0, st.session_state.mc_iter)
        
        # Calculate stats
        total_pts = len(st.session_state.mc_points)
        inside = np.sum(np.sqrt(st.session_state.mc_points[:,0]**2 + 
                               st.session_state.mc_points[:,1]**2) <= 1)
        pi_est = 4 * inside / total_pts
        
        # Display
        plot_ph.pyplot(fig)
        text_ph.markdown(f"""
        **π Estimate:** {pi_est:.5f}  
        **Points:** {total_pts:,}  
        **Iterations:** {st.session_state.mc_iter}
        """)
        
        time.sleep(0.05)
    
    reset_button()

# Demonstration 3: Central Limit Theorem
else:
    st.title("Central Limit Theorem Demo")
    st.markdown("""
    **Key insight:** Sample means become normally distributed as sample size increases, 
    regardless of population distribution shape.
    """)

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        pop_dist = st.selectbox("Population Distribution", 
                              ["Exponential", "Uniform", "Binomial", "Poisson"])
        if pop_dist == "Exponential":
            scale = st.slider("Scale (β)", 0.1, 10.0, 1.0)
        elif pop_dist == "Uniform":
            low = st.slider("Lower Bound", -10.0, 0.0, 0.0)
            high = st.slider("Upper Bound", 1.0, 20.0, 10.0)
        elif pop_dist == "Binomial":
            n_trials = st.slider("Number of Trials", 1, 100, 10)
            p_success = st.slider("Success Probability", 0.01, 0.99, 0.3)
        else:  # Poisson
            lambda_ = st.slider("Lambda (λ)", 0.1, 20.0, 5.0)

    with col2:
        sample_size = st.slider("Sample Size (n)", 1, 50, 5)
        n_samples = st.slider("Number of Samples", 100, 10000, 1000)

    if st.button("Run CLT Simulation"):
        sample_means = []

        for _ in range(n_samples):
            if pop_dist == "Exponential":
                sample = np.random.exponential(scale, sample_size)
            elif pop_dist == "Uniform":
                sample = np.random.uniform(low, high, sample_size)
            elif pop_dist == "Binomial":
                sample = np.random.binomial(n_trials, p_success, sample_size)
            else:  # Poisson
                sample = np.random.poisson(lambda_, sample_size)
            sample_means.append(np.mean(sample))

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Population distribution
        if pop_dist == "Exponential":
            pop_data = np.random.exponential(scale, 100000)
            ax1.hist(pop_data, bins=50, density=True, alpha=0.7)
        elif pop_dist == "Uniform":
            pop_data = np.random.uniform(low, high, 100000)
            ax1.hist(pop_data, bins=50, density=True, alpha=0.7)
        elif pop_dist == "Binomial":
            from scipy.stats import binom
            x = np.arange(0, n_trials + 1)
            pmf = binom.pmf(x, n_trials, p_success)
            ax1.bar(x, pmf, alpha=0.7)
        else:  # Poisson
            from scipy.stats import poisson
            x = np.arange(0, int(lambda_ + 4 * np.sqrt(lambda_)))
            pmf = poisson.pmf(x, lambda_)
            ax1.bar(x, pmf, alpha=0.7)

        ax1.set_title(f"Population Distribution ({pop_dist})")

        # Sample means distribution
        ax2.hist(sample_means, bins=50, density=True, alpha=0.7, color='orange')
        mu = np.mean(sample_means)
        std = np.std(sample_means)
        x = np.linspace(mu - 3*std, mu + 3*std, 100)
        ax2.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2)
        ax2.set_title(f"Distribution of Sample Means (n={sample_size})")
        ax2.annotate(f"μ = {mu:.3f}\nσ = {std:.3f}", 
                     xy=(0.95, 0.95), xycoords='axes fraction',
                     ha='right', va='top', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        st.pyplot(fig)
        plt.clf()

    reset_button()