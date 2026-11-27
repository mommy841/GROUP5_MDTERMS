import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SIM_DURATION = 900.0
MEAN_SERVICE_TIME = 4.0
PEAK_RATE_PER_MIN = 45.0 / 60.0
OFFPEAK_RATE_PER_MIN = 15.0 / 60.0

SLA_MAX_WAIT = 5.0
SLA_MAX_UTIL = 85.0
N_REPLICATIONS = 30
OUTPUT_DIR = "techserve_outputs"


@dataclass
class ScenarioConfig:
    name: str
    short_label: str
    schedule: List[Tuple[float, float, int]]
    deflection_rate: float = 0.0
    arrival_multiplier: float = 1.0
    mean_service_time: float = MEAN_SERVICE_TIME


def is_peak_time(t: float) -> bool:
    return (120.0 <= t < 360.0) or (420.0 <= t < 600.0)


def get_arrival_rate(t: float, multiplier: float = 1.0) -> float:
    base_rate = PEAK_RATE_PER_MIN if is_peak_time(t) else OFFPEAK_RATE_PER_MIN
    return base_rate * multiplier


class HelpdeskSimulation:
    def __init__(self, env: simpy.Environment, config: ScenarioConfig, rng: np.random.Generator):
        self.env = env
        self.config = config
        self.rng = rng

        self.tech_pool = simpy.Store(env)
        self.active_tokens = 0
        self.retire_debt = 0

        self.wait_times: List[float] = []
        self.system_times: List[float] = []
        self.busy_time_total: float = 0.0
        self.tickets_served: int = 0
        self.tickets_deflected: int = 0
        self.max_queue_length: int = 0

    def _update_queue_tracker(self) -> None:
        q_len = len(self.tech_pool.get_queue)
        if q_len > self.max_queue_length:
            self.max_queue_length = q_len

    def manage_shifts(self):
        for start_t, end_t, target_techs in self.config.schedule:
            if self.env.now < start_t:
                yield self.env.timeout(start_t - self.env.now)

            effective_staff = self.active_tokens - self.retire_debt
            delta = target_techs - effective_staff

            if delta > 0:
                reclaimed = min(delta, self.retire_debt)
                self.retire_debt -= reclaimed
                to_add = delta - reclaimed
                for _ in range(to_add):
                    self.active_tokens += 1
                    yield self.tech_pool.put(f"tech_{self.active_tokens}")
            elif delta < 0:
                to_remove = abs(delta)
                idle_available = min(to_remove, len(self.tech_pool.items))
                for _ in range(idle_available):
                    yield self.tech_pool.get()
                    self.active_tokens -= 1
                self.retire_debt += (to_remove - idle_available)

    def handle_ticket(self, ticket_id: int, arrival_time: float):
        req = self.tech_pool.get()
        self._update_queue_tracker()
        token = yield req

        service_start = self.env.now
        wait_time = service_start - arrival_time
        self.wait_times.append(wait_time)

        service_duration = self.rng.exponential(self.config.mean_service_time)
        yield self.env.timeout(service_duration)

        effective_end = min(self.env.now, SIM_DURATION)
        if service_start < SIM_DURATION:
            self.busy_time_total += (effective_end - service_start)

        self.tickets_served += 1
        self.system_times.append(self.env.now - arrival_time)

        if self.retire_debt > 0:
            self.retire_debt -= 1
            self.active_tokens -= 1
        else:
            yield self.tech_pool.put(token)

    def arrival_generator(self):
        lambda_max = PEAK_RATE_PER_MIN * self.config.arrival_multiplier
        t = 0.0
        ticket_id = 0

        while True:
            dt = self.rng.exponential(1.0 / lambda_max)
            t += dt
            if t >= SIM_DURATION:
                break

            yield self.env.timeout(dt)

            current_lambda = get_arrival_rate(self.env.now, self.config.arrival_multiplier)
            if self.rng.random() <= (current_lambda / lambda_max):
                ticket_id += 1
                if self.config.deflection_rate > 0.0 and self.rng.random() < self.config.deflection_rate:
                    self.tickets_deflected += 1
                else:
                    self.env.process(self.handle_ticket(ticket_id, self.env.now))


def run_single_replication(config: ScenarioConfig, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    sim = HelpdeskSimulation(env, config, rng)

    env.process(sim.manage_shifts())
    env.process(sim.arrival_generator())

    env.run(until=SIM_DURATION)
    if len(sim.tech_pool.get_queue) > 0 or sim.active_tokens > len(sim.tech_pool.items):
        env.run()

    scheduled_tech_minutes = sum(
        (end_t - start_t) * count for start_t, end_t, count in config.schedule
    )
    utilization_pct = (sim.busy_time_total / scheduled_tech_minutes) * 100.0

    return {
        "avg_wait_min": float(np.mean(sim.wait_times)) if sim.wait_times else 0.0,
        "avg_system_min": float(np.mean(sim.system_times)) if sim.system_times else 0.0,
        "utilization_pct": utilization_pct,
        "max_queue_len": sim.max_queue_length,
        "tickets_served": sim.tickets_served,
        "tickets_deflected": sim.tickets_deflected,
        "total_arrivals": sim.tickets_served + sim.tickets_deflected,
    }


def run_experiment(config: ScenarioConfig, n_reps: int = N_REPLICATIONS, base_seed: int = 1000) -> pd.DataFrame:
    records = []
    for rep in range(n_reps):
        res = run_single_replication(config, seed=base_seed + rep)
        res["replication"] = rep + 1
        res["scenario"] = config.name
        res["short_label"] = config.short_label
        records.append(res)
    return pd.DataFrame(records)


def build_scenarios() -> List[ScenarioConfig]:
    staggered_schedule = [
        (0.0, 60.0, 1),
        (60.0, 360.0, 3),
        (360.0, 420.0, 1),
        (420.0, 600.0, 3),
        (600.0, 900.0, 1),
    ]

    return [
        ScenarioConfig(
            name="Scenario 0: Baseline (2 Techs)",
            short_label="S0: Baseline\n(2 Techs)",
            schedule=[(0.0, 900.0, 2)],
            deflection_rate=0.0,
        ),
        ScenarioConfig(
            name="Scenario 1: +1 FTE (3 Techs)",
            short_label="S1: +1 FTE\n(3 Techs)",
            schedule=[(0.0, 900.0, 3)],
            deflection_rate=0.0,
        ),
        ScenarioConfig(
            name="Scenario 2: 2 Techs + AI Chatbot (30%)",
            short_label="S2: AI Bot\n(2 Techs)",
            schedule=[(0.0, 900.0, 2)],
            deflection_rate=0.30,
        ),
        ScenarioConfig(
            name="Scenario 3: Staggered Shifts (1/3/1/3/1)",
            short_label="S3: Staggered\nShifts",
            schedule=staggered_schedule,
            deflection_rate=0.0,
        ),
        ScenarioConfig(
            name="Scenario 4: AI Chatbot + Staggered Shifts",
            short_label="S4: AI Bot +\nStaggered",
            schedule=staggered_schedule,
            deflection_rate=0.30,
        ),
    ]


def summarize_results(df_all: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("avg_wait_min", "Avg Wait (min)"),
        ("avg_system_min", "Avg System Time (min)"),
        ("utilization_pct", "Utilization (%)"),
        ("max_queue_len", "Max Queue Length"),
        ("tickets_served", "Tickets Served"),
        ("tickets_deflected", "Tickets Deflected"),
    ]
    rows = []
    for scenario_name, group in df_all.groupby("scenario", sort=False):
        row = {"Scenario": scenario_name}
        for col, label in metrics:
            row[f"{label} Mean"] = group[col].mean()
            row[f"{label} SD"] = group[col].std(ddof=1)
            row[f"{label} Min"] = group[col].min()
            row[f"{label} Max"] = group[col].max()
        rows.append(row)
    return pd.DataFrame(rows)


def run_sensitivity_analysis() -> pd.DataFrame:
    sens_records = []

    for mult in [0.80, 1.00, 1.20, 1.40]:
        cfg = ScenarioConfig(
            name=f"Arrival {int(mult * 100)}%",
            short_label=f"{int(mult * 100)}%",
            schedule=[(0.0, 900.0, 2)],
            arrival_multiplier=mult,
        )
        df_rep = run_experiment(cfg, n_reps=N_REPLICATIONS, base_seed=5000)
        sens_records.append({
            "Parameter": "Arrival Rate",
            "Setting": f"{int((mult - 1.0) * 100):+d}%" if mult != 1.0 else "Baseline (0%)",
            "Numeric_Value": mult * 100,
            "Mean_Wait": df_rep["avg_wait_min"].mean(),
            "SD_Wait": df_rep["avg_wait_min"].std(ddof=1),
            "Mean_Util": df_rep["utilization_pct"].mean(),
            "SLA_Pass": (df_rep["avg_wait_min"].mean() <= SLA_MAX_WAIT) and (df_rep["utilization_pct"].mean() <= SLA_MAX_UTIL),
        })

    for svc in [3.0, 4.0, 5.0, 6.0]:
        cfg = ScenarioConfig(
            name=f"Service {svc}m",
            short_label=f"{svc} min",
            schedule=[(0.0, 900.0, 2)],
            mean_service_time=svc,
        )
        df_rep = run_experiment(cfg, n_reps=N_REPLICATIONS, base_seed=5000)
        sens_records.append({
            "Parameter": "Mean Service Time",
            "Setting": f"{svc:.1f} min",
            "Numeric_Value": svc,
            "Mean_Wait": df_rep["avg_wait_min"].mean(),
            "SD_Wait": df_rep["avg_wait_min"].std(ddof=1),
            "Mean_Util": df_rep["utilization_pct"].mean(),
            "SLA_Pass": (df_rep["avg_wait_min"].mean() <= SLA_MAX_WAIT) and (df_rep["utilization_pct"].mean() <= SLA_MAX_UTIL),
        })

    for techs in [1, 2, 3, 4]:
        cfg = ScenarioConfig(
            name=f"Servers {techs}",
            short_label=f"{techs} Techs",
            schedule=[(0.0, 900.0, techs)],
        )
        df_rep = run_experiment(cfg, n_reps=N_REPLICATIONS, base_seed=5000)
        sens_records.append({
            "Parameter": "Server Count",
            "Setting": f"{techs} Tech{'s' if techs > 1 else ''}",
            "Numeric_Value": techs,
            "Mean_Wait": df_rep["avg_wait_min"].mean(),
            "SD_Wait": df_rep["avg_wait_min"].std(ddof=1),
            "Mean_Util": df_rep["utilization_pct"].mean(),
            "SLA_Pass": (df_rep["avg_wait_min"].mean() <= SLA_MAX_WAIT) and (df_rep["utilization_pct"].mean() <= SLA_MAX_UTIL),
        })

    return pd.DataFrame(sens_records)


def generate_charts(df_all: pd.DataFrame, df_sens: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    scenarios = df_all["short_label"].unique()
    grouped = df_all.groupby("short_label", sort=False)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    means = grouped["avg_wait_min"].mean().values
    colors = ["#d9534f" if m > SLA_MAX_WAIT else "#2b7bba" for m in means]

    bars = ax.bar(scenarios, means, color=colors, width=0.55, edgecolor="black", linewidth=0.7)
    ax.axhline(SLA_MAX_WAIT, color="#b22222", linestyle="--", linewidth=1.8, label="SLA Target Threshold (<= 5.0 min)")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.45, f"{val:.2f} m",
                ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    ax.set_title("Chart 1: Average Ticket Wait Time by Scenario vs. SLA Target", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Mean Wait Time (Minutes)")
    ax.set_ylim(0, max(means) * 1.22)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart1_scenario_comparison.png")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    wait_mean = grouped["avg_wait_min"].mean().values
    wait_sd = grouped["avg_wait_min"].std(ddof=1).values
    sys_mean = grouped["avg_system_min"].mean().values
    sys_sd = grouped["avg_system_min"].std(ddof=1).values

    ax1.bar(scenarios, wait_mean, yerr=wait_sd, capsize=5, color="#4a90e2", edgecolor="black", alpha=0.9)
    ax1.axhline(SLA_MAX_WAIT, color="#b22222", linestyle="--", linewidth=1.5, label="SLA Limit (5.0 min)")
    ax1.set_title("Average Queue Wait Time (±1 SD)", fontweight="bold")
    ax1.set_ylabel("Minutes")
    ax1.legend()

    ax2.bar(scenarios, sys_mean, yerr=sys_sd, capsize=5, color="#50b89e", edgecolor="black", alpha=0.9)
    ax2.set_title("Average Total Time in System (±1 SD)", fontweight="bold")
    ax2.set_ylabel("Minutes")

    fig.suptitle("Chart 2: Scenario Performance Across 30 Replications (Mean ± Standard Deviation)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart2_performance_errorbars.png")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=300)
    params = ["Arrival Rate", "Mean Service Time", "Server Count"]
    panel_colors = ["#e67e22", "#8e44ad", "#16a085"]

    for ax, param, col in zip(axes, params, panel_colors):
        sub = df_sens[df_sens["Parameter"] == param]
        ax.errorbar(sub["Setting"], sub["Mean_Wait"], yerr=sub["SD_Wait"], marker="o",
                    linewidth=2.2, capsize=4, color=col, label="Mean Wait (±1 SD)")
        ax.axhline(SLA_MAX_WAIT, color="#b22222", linestyle="--", linewidth=1.3, label="5.0 min SLA")
        for x_idx, (y_val) in enumerate(sub["Mean_Wait"]):
            ax.annotate(f"{y_val:.1f}m", (x_idx, y_val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
        ax.set_title(f"Sensitivity: {param}", fontweight="bold")
        ax.set_ylabel("Mean Wait Time (Minutes)")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(loc="best", fontsize=8.5)

    fig.suptitle("Chart 3: One-at-a-Time Sensitivity Analysis on Ticket Wait Time (Baseline = 2 Techs)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart3_sensitivity_analysis.png")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    box_data = [df_all[df_all["short_label"] == s]["avg_wait_min"].values for s in scenarios]
    bp = ax1.boxplot(box_data, tick_labels=scenarios, patch_artist=True, notch=False)
    box_palette = ["#f4a582", "#92c5de", "#a6dba0", "#dfc27d", "#80cdc1"]
    for patch, color in zip(bp["boxes"], box_palette):
        patch.set_facecolor(color)
    ax1.axhline(SLA_MAX_WAIT, color="#b22222", linestyle="--", linewidth=1.4, label="SLA Target (5.0 min)")
    ax1.set_title("Wait Time Distribution Across 30 Runs", fontweight="bold")
    ax1.set_ylabel("Average Wait Time (Minutes)")
    ax1.legend()

    for s in scenarios:
        vals = df_all[df_all["short_label"] == s]["avg_wait_min"].values
        cum_mean = np.cumsum(vals) / np.arange(1, len(vals) + 1)
        ax2.plot(np.arange(1, len(vals) + 1), cum_mean, marker=".", linewidth=1.8, label=s.replace("\n", " "))
    ax2.axhline(SLA_MAX_WAIT, color="#b22222", linestyle="--", linewidth=1.4)
    ax2.set_title("Cumulative Mean Convergence (N = 30 Replications)", fontweight="bold")
    ax2.set_xlabel("Replication Number")
    ax2.set_ylabel("Cumulative Mean Wait Time (Minutes)")
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle("Chart 4: Replication Variability & Statistical Convergence", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart4_replication_variability.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    util_means = grouped["utilization_pct"].mean().values
    util_sds = grouped["utilization_pct"].std(ddof=1).values
    util_colors = ["#d9534f" if u > SLA_MAX_UTIL else "#2e8b57" for u in util_means]

    bars = ax.bar(scenarios, util_means, yerr=util_sds, capsize=5, color=util_colors, width=0.55, edgecolor="black")
    ax.axhline(SLA_MAX_UTIL, color="#b22222", linestyle="--", linewidth=1.8, label="Max Utilization SLA Cap (85.0%)")

    for bar, val in zip(bars, util_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2.2, f"{val:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    ax.set_title("Chart 5: Technician Resource Utilization vs. 85% SLA Ceiling", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Technician Utilization (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart5_resource_utilization.png")
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("Running TechServe University IT Helpdesk Simulation (30 Replications/Scenario)...")
    scenarios = build_scenarios()
    all_runs = pd.concat(
        [run_experiment(cfg, n_reps=N_REPLICATIONS, base_seed=2026) for cfg in scenarios],
        ignore_index=True,
    )

    summary_df = summarize_results(all_runs).round(2)

    compact_df = pd.DataFrame({
        "Scenario": summary_df["Scenario"],
        "Wait Mean (SD)": summary_df.apply(lambda r: f"{r['Avg Wait (min) Mean']:.2f} ({r['Avg Wait (min) SD']:.2f})", axis=1),
        "Wait [Min, Max]": summary_df.apply(lambda r: f"[{r['Avg Wait (min) Min']:.2f}, {r['Avg Wait (min) Max']:.2f}]", axis=1),
        "Sys Time Mean (SD)": summary_df.apply(lambda r: f"{r['Avg System Time (min) Mean']:.2f} ({r['Avg System Time (min) SD']:.2f})", axis=1),
        "Util % Mean (SD)": summary_df.apply(lambda r: f"{r['Utilization (%) Mean']:.2f}% ({r['Utilization (%) SD']:.2f})", axis=1),
        "Max Q Mean": summary_df["Max Queue Length Mean"],
        "Served / Deflected": summary_df.apply(lambda r: f"{r['Tickets Served Mean']:.1f} / {r['Tickets Deflected Mean']:.1f}", axis=1),
    })

    print("\n=== SCENARIO SUMMARY TABLE (30 REPLICATIONS) ===\n")
    print(compact_df.to_string(index=False))

    print("\nRunning Sensitivity Analysis...")
    sens_df = run_sensitivity_analysis().round(2)
    print("\n=== SENSITIVITY ANALYSIS TABLE (30 REPLICATIONS PER SETTING) ===")
    print(sens_df.to_string(index=False))

    summary_df.to_csv(f"{OUTPUT_DIR}/scenario_summary_full.csv", index=False)
    sens_df.to_csv(f"{OUTPUT_DIR}/sensitivity_analysis.csv", index=False)
    all_runs.to_csv(f"{OUTPUT_DIR}/all_replications_raw.csv", index=False)

    try:
        excel_path = f"{OUTPUT_DIR}/techserve_simulation_results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Scenario_Summary", index=False)
            sens_df.to_excel(writer, sheet_name="Sensitivity_Analysis", index=False)
            all_runs.to_excel(writer, sheet_name="Raw_30_Replications", index=False)
        print(f"\nExcel workbook saved to: ./{excel_path}")
    except ImportError:
        print("\n(Tip: Run 'pip install openpyxl' if you also want the combined .xlsx Excel file.)")

    generate_charts(all_runs, sens_df)
    print(f"All CSV files and 5 PNG charts saved to: ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()