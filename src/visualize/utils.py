import wandb
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
from typing import Callable, List


def compute_percent_improvement(row, target_model="BSARec"):
    """Compute relative change for target model wrt best other model"""
    values = row.dropna().astype(float) # drop NaNs, convert to float
    target_model_value = values[target_model]
    values = values.drop(target_model)
    best_baseline = values.max()

    return ((target_model_value - best_baseline) / best_baseline) * 100


def get_unique_seeds_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts number of unique seeds for each model and dataset combination.
    Must contain columns 'model_name', 'dataset', and 'seed'.
    """
    return df.loc[:, ["model_name", "dataset", "seed"]].groupby(["model_name", "dataset"]).nunique()


def t_test_target_baseline(
        df,
        target_model = "BSARec",
        signficant_col = "significant",
        alpha = 0.05,
        paired = False,
        cut_off = False,
        debug = False
    ):
    """
    Performs t-test between target model and best other model for each dataset and metric.

    Arguments:
        df: DataFrame containing columns 'model_name', 'dataset', 'metric', 'value', and 'seed'.
            'value' should contain the metric values to compare.
        target_model: Name of the target model to compare against the best other model.
        signficant_col: Name of the column to store significance results.
        alpha: Significance level for the t-test.
        paired: Whether to perform a paired t-test (True) or an independent t-test (False).
        cut_off: If True, will cut off the longer list to match the length of the shorter one
                 when comparing two models with different number of seeds.
        debug: If True, will include additional debug information in the results.

    Returns:
        DataFrame containing the results of the t-tests, indexed by dataset and metric.
        If all pairs all invalid, the DataFrame will be empty.
    """
    # set t- test
    ttest_fn = ttest_rel if paired else ttest_ind

    df_ttest = df.copy(deep=True)
    df_seed = df.copy(deep=True)
    df_ttest = df_ttest.drop(columns=["seed"])
    mean_df = df_ttest.groupby(['model_name', 'dataset', 'metric'])['value'].mean().reset_index()
    pivot_df = mean_df.pivot_table(index=['dataset', 'metric'], columns='model_name', values='value')

    assert len(pivot_df.columns) >= 2

    results = []

    for (dataset, metric), row in pivot_df.iterrows():
        row_wo_target = row.drop(labels=target_model, errors='ignore').sort_values(ascending=False)

        if row_wo_target.count() < 1:
            continue

        best_other_model = row_wo_target.index[0]

        # Extract values for both models
        target_vals = df_seed[(df_seed['model_name'] == target_model) &
                            (df_seed['dataset'] == dataset) &
                            (df_seed['metric'] == metric)]
        target_vals = target_vals.sort_values(["seed"])
        target_seeds = target_vals["seed"].values



        best_vals = df_seed[(df_seed['model_name'] == best_other_model) &
                            (df_seed['dataset'] == dataset) &
                            (df_seed['metric'] == metric)]
        best_vals = best_vals.sort_values(["seed"])
        best_seeds = best_vals["seed"].values

        target_vals = target_vals["value"].values
        best_vals = best_vals["value"].values

        # Ensure seeds match for paired t-test
        if paired and not np.all(target_seeds == best_seeds):
            print(f"Skipping {dataset} - {metric}: Seed mismatch between {target_model} and {best_other_model}")
            print(f"Seeds: [{target_model}] {target_seeds} vs [{best_other_model}] {best_seeds}")
            continue
        elif not paired and len(target_seeds) != len(best_seeds):
            if not cut_off:
                print(f"Skipping {dataset} - {metric}: Length mismatch between {target_model} and {best_other_model}")
                print(f"Length: [{target_model}] {len(target_seeds)} vs [{best_other_model}] {len(best_seeds)}")
                continue
            max_samples = min(len(target_seeds), len(best_seeds))
            print(f"Taking first {max_samples} samples {dataset} - {metric}: Length mismatch between {target_model} and {best_other_model}")
            print(f"Length: [{target_model}] {len(target_seeds)} vs [{best_other_model}] {len(best_seeds)}")
            target_vals = target_vals[:max_samples]
            best_vals = best_vals[:max_samples]


        _, p_value = ttest_fn(target_vals, best_vals)

        if debug:
            results.append({
                'dataset': dataset,
                'metric': metric,
                'target_model': target_model,
                'best_other_model': best_other_model,
                'target_mean': target_vals.mean(),
                'best_other_mean': best_vals.mean(),
                signficant_col: p_value < alpha,
            })
        else:
            results.append({
                'dataset': dataset,
                'metric': metric,
                signficant_col: p_value < alpha,
            })

    stats_df = pd.DataFrame(results)

    if len(stats_df) > 0:
        stats_df = stats_df.set_index(["dataset", "metric"])

    return stats_df


def format_row(row: pd.Series, rel_improvement_col, significant_col, show_second_best=True) -> list:
    """Helper function to format a row for LaTeX table output.
    adds bold and underline formatting for the best and second-best values,
    """
    formatted_row = {}

    stat_cols = [rel_improvement_col, significant_col]

    row = pd.to_numeric(row, errors='coerce')
    top2 = row.drop(stat_cols).nlargest(2).index.tolist()
    first, second = top2

    for name, val in row.items():
        if name in stat_cols:
            continue

        fval = val

        if np.isnan(fval):
            fval = ""
        elif isinstance(val, float):
            fval = f"{val:.4f}"
        elif isinstance(val, str):
            pass
        else:
            fval = str(val)

        if name == first:
            fval = "\\textbf{" + fval + "}"
        elif name == second and show_second_best:
            fval = "\\underline{" + fval + "}"

        formatted_row[name] = fval

    modifier = {
        True: "\\textsuperscript{*}",
        False: ""
    }

    formatted_row[rel_improvement_col] = f"{row[rel_improvement_col]:.2f}" + modifier.get(row[significant_col], " X")

    return pd.Series(formatted_row)


def create_main_latex_str(df, models, col_order, signficant_col, paired, table_label, table_caption):
    ncols = len(models) + 3
    latex = df.to_latex(escape=False, na_rep="", multicolumn=True, multirow=True)
    latex = latex.replace(
        "\\begin{tabular}{" + "l" * ncols + "}",
        "\\begin{tabular}{lll" + "c" * len(models) + "}\n\\toprule"
    )
    latex = latex.replace("\\\\\n", " \\\\\n\\midrule\n", 1)  # Add \midrule after header
    latex = latex.replace("\\cline{1-" + str(ncols) + "}\n\\bottomrule", "\\bottomrule")
    latex = latex.replace("\\cline{1-" + str(ncols) + "}", "\\midrule")

    # fix the header line
    lines = latex.splitlines()
    toprule_idx = None
    for i, line in enumerate(lines):
        if r"\toprule" in line:
            toprule_idx = i
            break

    # Replace next two lines after \toprule with single header line
    # Assuming they exist
    header_line = r"\textbf{Dataset} & \textbf{Metric} &" + " & ".join([f"\\textbf{{{col}}}" for col in col_order if col != signficant_col]) + r" \\"
    lines[toprule_idx + 1] = header_line

    del lines[toprule_idx+2]

    # insert \midrule after new header line if not already present
    if r"\midrule" not in lines[toprule_idx+2]:
        lines.insert(toprule_idx+2, r"\midrule")

    latex_fixed = "\n".join(lines)
    latex_fixed = latex_fixed.replace(f"\\midrule\ndataset & metric &{'  &' * (len(models) - 1)}  &  \\\\\n", "")
    latex_fixed = latex_fixed.replace("\\multirow[t]{6}{*}{", "\\multirow{6}{*}{\\centering ")
    # latex_fixed = latex_fixed.replace("\\toprule", "", 1)


    latex_fixed = r"""
    \begin{table*}[h]
        \centering
        \caption{""" + table_caption + r"""}
        \begin{adjustbox}{max width=\textwidth}
        \label{""" + table_label + "}" + latex_fixed

    latex_fixed += "\n\\end{adjustbox}\n\\end{table*}\n"

    return latex_fixed


def create_main_table(
        df,
        models,
        metrics,
        target_model,
        alpha = 0.05,
        paired = False,
        cut_off = False,
        debug = False,
        rel_improvement_col= "Diff.",
        dropna=True,
        show_second_best=True,
        table_label="tab:label",
        table_caption="TODO"
    ):
    """Creates a main table comparing models across datasets and metrics.

    Arguments:
        df: DataFrame containing columns 'model_name', 'dataset', 'metric', 'value', and 'seed'.
            'value' should contain the metric values to compare.
        models: List of model names to include in the table.
        metrics: List of metrics to include in the table.
        target_model: Name of the target model to compare against the best other model.
        alpha: Significance level for the t-test.
        paired: Whether to perform a paired t-test (True) or an independent t-test (False).
        cut_off: If True, will cut off the longer list to match the length of the shorter one
                 when comparing two models with different number of seeds.
        debug: If True, will include additional debug information in the results.
        rel_improvement_col: Name of the column to store relative improvement values.
        dropna: If True, will drop rows with NaN values.
        show_second_best: If True, will highlight the second-best model in the table.
        table_label: Label for the LaTeX table.
        table_caption: Caption for the LaTeX table.

    Returns:
        latex: LaTeX string for the table.
        df: DataFrame containing the results of the t-tests and relative improvements.
        stats_df: DataFrame containing the results of the t-tests.
    """
    # define constants
    df_immutable = df.copy(deep=True)
    signficant_col = "significant",

    df = df_immutable.copy(deep=True).drop(columns=["seed"])
    df = (
        df.groupby(["model_name", "dataset", "metric"])
        .agg(["mean"])
        .stack(future_stack=True)
        .reset_index()
        .pivot_table(
            index=['dataset', 'metric'],
            columns='model_name',
            values='value'
        )
    )

    # compute improvement only on 'mean' rows
    df[rel_improvement_col] = np.nan
    df.loc[:, rel_improvement_col] = df.apply(
        lambda x: compute_percent_improvement(x, target_model),
        axis=1
    )

    # add statistic significance column
    stats_df = t_test_target_baseline(
        df_immutable.copy(deep=True),
        target_model=target_model,
        signficant_col=signficant_col,
        paired=paired,
        cut_off=cut_off,
        debug=debug,
        alpha=alpha
    )
    df = df.join(stats_df, how="left")

    # sort metrics
    df = df.copy().reset_index()
    df["metric"] = pd.Categorical(df["metric"], categories=metrics, ordered=True)

    df = df.sort_values(["dataset", "metric"])
    df = df.set_index(["dataset", "metric"])

    # re-order columns to match the models list
    col_order = [
        mname.replace("_", " ")
        for mname in models
        if mname.replace("_", " ") in df.columns
    ] + [rel_improvement_col, signficant_col]
    df = df[col_order]

    if dropna:
        df = df.dropna()

    df = df.apply(
        lambda row: format_row(
            row,
            rel_improvement_col,
            signficant_col,
            show_second_best=show_second_best
        ),
        axis=1
    )

    latex = create_main_latex_str(
        df,
        models,
        col_order,
        signficant_col,
        paired,
        table_label,
        table_caption
    )

    return latex, df, stats_df


class WandbParser():
    def __init__(self, entity: str, api=None, verbose: bool = False, users=None) -> None:
        self.entity = entity
        self.api = api if api is not None else wandb.Api()
        self.cfg_constraints = {}
        self.verbose = verbose
        self._admissible_states = ["finished"]
        self.users = users if users is not None else []

    def vprint(self, value):
        if self.verbose:
            print(value)

    @property
    def admissible_states(self):
        return self._admissible_states

    @admissible_states.setter
    def admissible_states(self, obj):
        if isinstance(obj, str):
            self._admissible_states = [obj]
        elif isinstance(obj, list):
            self._admissible_states = obj
        else:
            print(f"Warning: Ignoring states. Input must be set str or List<str>, but received {type(obj)}")

    def reset(self, full=False):
        """
        Reset the parser, clearing all registered projects and constraints.
        """
        self.vprint("Resetting parser")
        self.cfg_constraints = {}
        self._admissible_states = ["finished"]

        if full:
            self.users = []
            self.api = wandb.Api()

    def register_project(self, project, **constraints):
        """
        Register project and cfg constraints

        Arguments:
            project: wandb project name
            constraints: catches contrains. Must be in the form of <str, List[Any] | None>
                         If constraint is None, then it will always be considered as 'satisfied'

        """
        self.vprint(f"Added {project} to parser")
        self.cfg_constraints[project] = constraints

    def register_and_parse(
            self,
            project,
            constraints,
            cfg: List[str], summary: List[str], post_processing: Callable = None):
        self.register_project(project, **constraints)
        return self.parse(cfg, summary, post_processing)

    def parse(self, cfg: List[str], summary: List[str], post_processing: Callable = None):
        """
        Parses runs from wandb.

        Arguments:
            cfg: list containing cfg attributes to load
            summary: list containing name of statistics to retrieve.
                      Each summary statistic is added as separate row.
            post_processing: post processor for dataframe. If None, no post processing is applied

        Returns:
            DataFrame containing cfg and summary statistics
        """
        data = []

        for project, constraints in self.cfg_constraints.items():
            filters = {
                f"config.{attr}": {"$in": attr_constraint}
                for attr, attr_constraint
                in constraints.items()
                if attr_constraint is not None
            }
            filters["state"] = {"$in": self._admissible_states}
            self.vprint(f"Loading runs from {project} with filters {filters}")

            # parse runs
            runs = self.api.runs(f"{self.entity}/{project}", filters=filters)

            for run in runs:
                if len(self.users) > 0 and run.user.username not in self.users:
                    self.vprint(f"Skipping run {run.id} from user {run.user.username}")
                    continue

                # get configuration
                run_cfg_attrs = {attr: run.config.get(attr, None) for attr in cfg}

                # get summary statistics
                for summary_attr in summary:
                    summary_attr_value = run.summary.get(summary_attr, None)
                    data.append({**run_cfg_attrs, "metric": summary_attr, "value": summary_attr_value})

        df = pd.DataFrame(data)

        if post_processing is not None:
            self.vprint("Applying post processing")
            df = post_processing(df)

        return df
