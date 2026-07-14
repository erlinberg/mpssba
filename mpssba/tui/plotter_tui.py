import os
import json
import re
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, SelectionList, Input, Button, Select
from textual.containers import Horizontal
from textual.binding import Binding


def parse_run_info(run_path):
    run_name = os.path.basename(run_path)
    config_path = os.path.join(run_path, "config.json")

    date_str = f"{run_name[:2]}/{run_name[2:4]}" if len(run_name) >= 4 else "??"
    time_str = f"{run_name[5:7]}:{run_name[7:9]}" if len(run_name) >= 9 else "??"

    n_qubits = "?"
    bond_dim = "?"
    mask_type = "?"
    mask_val = "?"
    entropy = "?"
    init_method = "?"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                n_qubits = config.get("model", {}).get("n_qubits", "?")
                bond_dim = config.get("model", {}).get("bond_dim", "?")
                mask_type = (
                    config.get("training", {}).get("masking", {}).get("type", "?")
                )
                mask_val = (
                    config.get("training", {}).get("masking", {}).get("value", "?")
                )
                entropy = (
                    config.get("dist", {}).get("kwargs", {}).get("target_entropy", "?")
                )
                if entropy == "?":
                    match = re.search(r"_ent([0-9.]+)_", run_name)
                    if match:
                        entropy = match.group(1)
                init_method = config.get("init", {}).get("method", "?")
        except Exception:
            pass

    # Format nicely
    formatted = f"{date_str} {time_str} | NQ:{n_qubits:<2} BD:{bond_dim:<3} | Mask:{mask_type}:{mask_val} | Ent:{entropy} | Init:{init_method} | {run_name}"
    return formatted


def get_all_runs():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    runs_dir = os.path.join(root_dir, "results", "runs")

    runs = []
    if os.path.exists(runs_dir):
        for run_name in sorted(os.listdir(runs_dir), reverse=True):
            run_path = os.path.join(runs_dir, run_name)
            if os.path.isdir(run_path):
                formatted = parse_run_info(run_path)
                runs.append((formatted, run_path))
    return runs


class PlotterApp(App[dict]):
    CSS_PATH = "plotter_tui.tcss"
    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select All"),
        Binding("ctrl+d", "deselect_all", "Deselect All"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.all_runs = get_all_runs()
        self.selected_runs = []

    def compose(self) -> ComposeResult:
        yield Header()

        plot_types = [
            ("All Plots", "all"),
            ("Masses", "masses"),
            ("Learning Curves", "learning_curves"),
            ("Deltas", "deltas"),
            ("Gradients", "gradients"),
            ("Covariance", "covariance"),
            ("Learning Time", "learning_time"),
        ]

        with Horizontal(id="top_bar"):
            yield Input(placeholder="Search/Filter runs...", id="search_input")
            yield Select(plot_types, value="all", id="plot_type_select")
            yield Button("Plot Selected", id="plot_button", variant="success")

        yield SelectionList(id="runs_list")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "mpssba plotter"
        self._update_list()
        self.query_one(Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "plot_button":
            self.action_plot()

    def _update_list(self, filter_text: str = ""):
        runs_list = self.query_one("#runs_list", SelectionList)
        runs_list.clear_options()

        filter_text = filter_text.lower()
        options = []
        for formatted, run_path in self.all_runs:
            if filter_text in formatted.lower() or filter_text in run_path.lower():
                options.append((formatted, run_path, run_path in self.selected_runs))

        runs_list.add_options(options)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search_input":
            self._update_list(event.value)

    def on_selection_list_selection_toggled(
        self, event: SelectionList.SelectionToggled
    ) -> None:
        run_path = event.selection.value
        if run_path in event.selection_list.selected:
            if run_path not in self.selected_runs:
                self.selected_runs.append(run_path)
        else:
            if run_path in self.selected_runs:
                self.selected_runs.remove(run_path)

    def action_plot(self) -> None:
        if not self.selected_runs:
            self.notify("No runs selected!", severity="warning")
            return

        run_paths = list(self.selected_runs)
        plot_type = self.query_one("#plot_type_select", Select).value

        self.exit({"runs": run_paths, "plot_type": plot_type})

    def action_select_all(self) -> None:
        runs_list = self.query_one("#runs_list", SelectionList)
        runs_list.select_all()
        for option in runs_list._options:
            if option.value in runs_list.selected:
                if option.value not in self.selected_runs:
                    self.selected_runs.append(option.value)

    def action_deselect_all(self) -> None:
        runs_list = self.query_one("#runs_list", SelectionList)
        runs_list.deselect_all()
        for option in runs_list._options:
            if option.value in self.selected_runs:
                self.selected_runs.remove(option.value)


if __name__ == "__main__":
    PlotterApp().run()
