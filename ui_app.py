"""
AEDA UI Application
"""

import os
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core.html_report_generator import HTMLReportGenerator
from core.report import Report
from preprocessing.dataset import Dataset
from report_components.base_component import AnalysisContext
from report_components.simple_components.missing_values import MissingValuesReport
from report_components.simple_components.dataset_overview import DatasetOverviewComponent
from report_components.simple_components.exact_duplicates import ExactDuplicateDetectionComponent
from report_components.core_components.outlier_detection import OutlierDetectionComponent
from report_components.core_components.categoircal_outlier_detection import CategoricalOutlierDetectionComponent
from report_components.core_components.distribution_modelling import DistributionModelingComponent
from report_components.core_components.label_noise_detection import LabelNoiseDetectionComponent
from report_components.core_components.composite_quality_score import CompositeQualityScoreComponent
from report_components.core_components.relational_consistency import RelationalConsistencyComponent
from report_components.core_components.near_duplicate_detection import NearDuplicateDetectionComponent
from report_components.core_components.llm_dataset_summary import LLMDatasetSummaryComponent


class ModernStyle:
    """Modern color scheme and styling constants."""
    # Colors
    BG_PRIMARY = "#1a1b26"
    BG_SECONDARY = "#24283b"
    BG_TERTIARY = "#414868"
    ACCENT = "#7aa2f7"
    ACCENT_HOVER = "#89b4fa"
    SUCCESS = "#9ece6a"
    WARNING = "#e0af68"
    ERROR = "#f7768e"
    TEXT_PRIMARY = "#c0caf5"
    TEXT_SECONDARY = "#565f89"
    TEXT_MUTED = "#414868"

    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_LARGE = 24
    FONT_SIZE_MEDIUM = 12
    FONT_SIZE_SMALL = 10

    @classmethod
    def configure_styles(cls, root):
        """Configure ttk styles for modern look."""
        style = ttk.Style(root)
        style.theme_use('clam')

        # Configure main frame
        style.configure("Main.TFrame", background=cls.BG_PRIMARY)
        style.configure("Card.TFrame", background=cls.BG_SECONDARY)

        # Configure labels
        style.configure("Title.TLabel",
                       background=cls.BG_PRIMARY,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_LARGE, "bold"))

        style.configure("Subtitle.TLabel",
                       background=cls.BG_PRIMARY,
                       foreground=cls.TEXT_SECONDARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))

        style.configure("Card.TLabel",
                       background=cls.BG_SECONDARY,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))

        style.configure("CardTitle.TLabel",
                       background=cls.BG_SECONDARY,
                       foreground=cls.ACCENT,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM, "bold"))

        style.configure("Status.TLabel",
                       background=cls.BG_PRIMARY,
                       foreground=cls.TEXT_SECONDARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_SMALL))

        style.configure("Progress.TLabel",
                       background=cls.BG_PRIMARY,
                       foreground=cls.ACCENT,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM, "bold"))

        # Configure buttons
        style.configure("Accent.TButton",
                       background=cls.ACCENT,
                       foreground=cls.BG_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM, "bold"),
                       padding=(20, 10))

        style.map("Accent.TButton",
                 background=[("active", cls.ACCENT_HOVER), ("disabled", cls.BG_TERTIARY)])

        style.configure("Secondary.TButton",
                       background=cls.BG_TERTIARY,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM),
                       padding=(15, 8))

        style.map("Secondary.TButton",
                 background=[("active", cls.BG_SECONDARY)])

        # Configure checkbutton
        style.configure("Card.TCheckbutton",
                       background=cls.BG_SECONDARY,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))

        style.map("Card.TCheckbutton",
                 background=[("active", cls.BG_SECONDARY)])

        # Configure entry
        style.configure("Card.TEntry",
                       fieldbackground=cls.BG_TERTIARY,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))

        # Configure progressbar
        style.configure("Custom.Horizontal.TProgressbar",
                       background=cls.ACCENT,
                       troughcolor=cls.BG_TERTIARY,
                       borderwidth=0,
                       lightcolor=cls.ACCENT,
                       darkcolor=cls.ACCENT)

        # Configure combobox
        style.configure("Card.TCombobox",
                       fieldbackground=cls.BG_TERTIARY,
                       background=cls.BG_TERTIARY,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))


class AEDAApp:
    """Main AEDA Application with Modern UI."""

    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AEDA - Automated Exploratory Data Analysis")
        self.root.geometry("700x750")
        self.root.configure(bg=ModernStyle.BG_PRIMARY)
        self.root.resizable(True, True)
        self.root.minsize(600, 500)

        # Center window
        self._center_window()

        # Configure styles
        ModernStyle.configure_styles(self.root)

        # Variables
        self.file_path = tk.StringVar()
        self.is_ml_dataset = tk.BooleanVar(value=False)
        self.target_column = tk.StringVar()
        self.progress_var = tk.DoubleVar(value=0)
        self.status_text = tk.StringVar(value="Ready")
        self.current_component = tk.StringVar(value="")
        self.spinner_text = tk.StringVar(value="")

        # State
        self.dataset: Optional[Dataset] = None
        self.columns: list = []
        self.is_running = False
        self.cancel_requested = False
        self.spinner_index = 0
        self.spinner_job = None

        # Build UI
        self._build_ui()

    def _center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = 700
        height = 750
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _build_ui(self):
        """Build the main UI with scrollable content."""
        # Create canvas for scrolling
        self.canvas = tk.Canvas(self.root, bg=ModernStyle.BG_PRIMARY, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)

        # Scrollable frame inside canvas
        self.scrollable_frame = ttk.Frame(self.canvas, style="Main.TFrame")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Bind canvas resize to adjust inner frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Enable mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        # Pack scrollbar and canvas
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Main container inside scrollable frame
        main_frame = ttk.Frame(self.scrollable_frame, style="Main.TFrame", padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        self._build_header(main_frame)

        # File Selection Card
        self._build_file_card(main_frame)

        # ML/DL Options Card
        self._build_ml_card(main_frame)

        # Progress Section
        self._build_progress_section(main_frame)

        # Log Section
        self._build_log_section(main_frame)

        # Run Button
        self._build_run_button(main_frame)

    def _on_canvas_configure(self, event):
        """Adjust the inner frame width when canvas is resized."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")

    def _build_header(self, parent):
        """Build the header section."""
        header_frame = ttk.Frame(parent, style="Main.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 20))

        # Logo/Title
        title_label = ttk.Label(header_frame,
                               text="🔍 AEDA",
                               style="Title.TLabel")
        title_label.pack(anchor=tk.CENTER)

        subtitle_label = ttk.Label(header_frame,
                                  text="Automated Exploratory Data Analysis",
                                  style="Subtitle.TLabel")
        subtitle_label.pack(anchor=tk.CENTER)

    def _build_file_card(self, parent):
        """Build the file selection card."""
        card = self._create_card(parent, "📁 Dataset Selection")

        # File path display
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X, pady=(10, 0))

        self.path_entry = tk.Entry(path_frame,
                                   textvariable=self.file_path,
                                   font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_MEDIUM, "bold"),
                                   bg=ModernStyle.BG_TERTIARY,
                                   fg="#1a1a1a",
                                   insertbackground="#1a1a1a",
                                   relief=tk.FLAT,
                                   state="readonly")
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))

        browse_btn = tk.Button(path_frame,
                              text="Browse",
                              command=self._browse_file,
                              bg=ModernStyle.BG_TERTIARY,
                              fg=ModernStyle.TEXT_PRIMARY,
                              activebackground=ModernStyle.ACCENT,
                              activeforeground=ModernStyle.BG_PRIMARY,
                              font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_MEDIUM),
                              relief=tk.FLAT,
                              cursor="hand2",
                              padx=15,
                              pady=5)
        browse_btn.pack(side=tk.RIGHT)

        # File info label
        self.file_info_label = ttk.Label(card,
                                         text="Supported formats: CSV, Parquet",
                                         style="Card.TLabel")
        self.file_info_label.pack(anchor=tk.W, pady=(10, 0))

    def _build_ml_card(self, parent):
        """Build the ML/DL options card."""
        card = self._create_card(parent, "🤖 Machine Learning Options")

        # Checkbox for ML dataset
        self.ml_check = ttk.Checkbutton(card,
                                        text="This dataset is for ML/DL (includes target labels)",
                                        variable=self.is_ml_dataset,
                                        command=self._toggle_ml_options,
                                        style="Card.TCheckbutton")
        self.ml_check.pack(anchor=tk.W, pady=(10, 0))

        # Target column frame (initially hidden)
        self.target_frame = ttk.Frame(card, style="Card.TFrame")

        target_label = ttk.Label(self.target_frame,
                                text="Target Column:",
                                style="Card.TLabel")
        target_label.pack(anchor=tk.W, pady=(10, 5))

        self.target_combo = ttk.Combobox(self.target_frame,
                                         textvariable=self.target_column,
                                         state="readonly",
                                         font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_MEDIUM))
        self.target_combo.pack(fill=tk.X, ipady=5)

        self.target_hint = ttk.Label(self.target_frame,
                                    text="Select the column containing labels for noise detection",
                                    style="Card.TLabel")
        self.target_hint.configure(foreground=ModernStyle.TEXT_SECONDARY)
        self.target_hint.pack(anchor=tk.W, pady=(5, 0))

    def _build_progress_section(self, parent):
        """Build the progress section."""
        self.progress_frame = ttk.Frame(parent, style="Main.TFrame")
        self.progress_frame.pack(fill=tk.X, pady=20)

        # Progress label with spinner
        progress_header = ttk.Frame(self.progress_frame, style="Main.TFrame")
        progress_header.pack(fill=tk.X)

        # Spinner label
        self.spinner_label = ttk.Label(progress_header,
                                       textvariable=self.spinner_text,
                                       style="Progress.TLabel",
                                       width=2)
        self.spinner_label.pack(side=tk.LEFT)

        self.progress_label = ttk.Label(progress_header,
                                        textvariable=self.status_text,
                                        style="Status.TLabel")
        self.progress_label.pack(side=tk.LEFT, padx=(5, 0))

        self.percent_label = ttk.Label(progress_header,
                                       text="0%",
                                       style="Progress.TLabel")
        self.percent_label.pack(side=tk.RIGHT)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.progress_frame,
                                           variable=self.progress_var,
                                           maximum=100,
                                           mode='determinate',
                                           style="Custom.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=(10, 0), ipady=3)

        # Current component label
        self.component_label = ttk.Label(self.progress_frame,
                                        textvariable=self.current_component,
                                        style="Status.TLabel")
        self.component_label.pack(anchor=tk.W, pady=(10, 0))

    def _build_log_section(self, parent):
        """Build the log section."""
        log_frame = ttk.Frame(parent, style="Main.TFrame")
        log_frame.pack(fill=tk.X, pady=(0, 15))

        log_label = ttk.Label(log_frame,
                             text="📋 Activity Log",
                             style="Subtitle.TLabel")
        log_label.pack(anchor=tk.W, pady=(0, 5))

        # Log text widget with scrollbar
        log_container = ttk.Frame(log_frame, style="Card.TFrame")
        log_container.pack(fill=tk.X)

        self.log_text = tk.Text(log_container,
                               height=6,
                               font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_SMALL),
                               bg=ModernStyle.BG_SECONDARY,
                               fg=ModernStyle.TEXT_PRIMARY,
                               relief=tk.FLAT,
                               padx=10,
                               pady=10,
                               state=tk.DISABLED,
                               wrap=tk.WORD)

        log_scrollbar = ttk.Scrollbar(log_container, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _add_log(self, message: str):
        """Add a log entry with timestamp (thread-safe)."""
        def update():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(0, update)

    def _start_spinner(self):
        """Start the spinner animation."""
        self.spinner_index = 0
        self._animate_spinner()

    def _animate_spinner(self):
        """Animate the spinner."""
        if self.is_running:
            self.spinner_text.set(self.SPINNER_CHARS[self.spinner_index])
            self.spinner_index = (self.spinner_index + 1) % len(self.SPINNER_CHARS)
            self.spinner_job = self.root.after(100, self._animate_spinner)

    def _stop_spinner(self):
        """Stop the spinner animation."""
        if self.spinner_job:
            self.root.after_cancel(self.spinner_job)
            self.spinner_job = None
        self.spinner_text.set("")

    def _build_run_button(self, parent):
        """Build the run button."""
        button_frame = ttk.Frame(parent, style="Main.TFrame")
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self.run_btn = tk.Button(button_frame,
                                text="🚀 Generate Report",
                                command=self._run_analysis,
                                bg=ModernStyle.ACCENT,
                                fg=ModernStyle.BG_PRIMARY,
                                activebackground=ModernStyle.ACCENT_HOVER,
                                activeforeground=ModernStyle.BG_PRIMARY,
                                font=(ModernStyle.FONT_FAMILY, 14, "bold"),
                                relief=tk.FLAT,
                                cursor="hand2",
                                padx=30,
                                pady=12)
        self.run_btn.pack(fill=tk.X)

    def _create_card(self, parent, title):
        """Create a styled card widget."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=20)
        card.pack(fill=tk.X, pady=(0, 15))

        # Card title
        title_label = ttk.Label(card, text=title, style="CardTitle.TLabel")
        title_label.pack(anchor=tk.W)

        return card

    def _browse_file(self):
        """Open file browser dialog."""
        file_types = [
            ("Data files", "*.csv *.parquet"),
            ("CSV files", "*.csv"),
            ("Parquet files", "*.parquet"),
            ("All files", "*.*")
        ]

        filepath = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=file_types,
            initialdir=os.getcwd()
        )

        if filepath:
            self.file_path.set(filepath)
            self._load_dataset_preview(filepath)

    def _load_dataset_preview(self, filepath: str):
        """Load dataset and update UI with preview info."""
        try:
            ext = Path(filepath).suffix.lower()

            if ext == ".csv":
                self.dataset = Dataset.from_csv(filepath)
            elif ext == ".parquet":
                self.dataset = Dataset.from_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # Get columns for target selection
            self.columns = list(self.dataset.df.columns)
            self.target_combo['values'] = self.columns

            # Update file info
            rows, cols = self.dataset.df.shape
            self.file_info_label.config(
                text=f"✓ Loaded: {rows:,} rows × {cols} columns",
                foreground=ModernStyle.SUCCESS
            )

        except Exception as e:
            self.dataset = None
            self.columns = []
            self.file_info_label.config(
                text=f"✗ Error: {str(e)[:50]}",
                foreground=ModernStyle.ERROR
            )
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")

    def _toggle_ml_options(self):
        """Toggle ML options visibility."""
        if self.is_ml_dataset.get():
            self.target_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.target_frame.pack_forget()
            self.target_column.set("")

    def _validate_inputs(self) -> bool:
        """Validate user inputs before running."""
        if not self.file_path.get():
            messagebox.showwarning("Validation Error", "Please select a dataset file.")
            return False

        if self.dataset is None:
            messagebox.showwarning("Validation Error", "Failed to load dataset. Please select a valid file.")
            return False

        if self.is_ml_dataset.get():
            if not self.target_column.get():
                messagebox.showwarning("Validation Error",
                                      "Please select a target column for ML/DL dataset.")
                return False

            if self.target_column.get() not in self.columns:
                messagebox.showwarning("Validation Error",
                                      f"Target column '{self.target_column.get()}' not found in dataset.")
                return False

        return True

    def _update_progress(self, value: float, status: str, component: str = ""):
        """Update progress bar and status (thread-safe)."""
        def update():
            self.progress_var.set(value)
            self.percent_label.config(text=f"{int(value)}%")
            self.status_text.set(status)
            self.current_component.set(component)

        self.root.after(0, update)

    def _run_analysis(self):
        """Run the analysis in a separate thread."""
        if self.is_running:
            return

        if not self._validate_inputs():
            return

        self.is_running = True
        self.cancel_requested = False
        # swap button to Abort and enable it so user can click Abort
        self.root.after(0, lambda: self.run_btn.config(text='Abort', command=self._abort_analysis))
        # make Abort visible and clickable
        self.run_btn.config(state=tk.NORMAL, bg=ModernStyle.WARNING, fg=ModernStyle.BG_PRIMARY)

        # Start spinner
        self._start_spinner()

        # Clear and add initial log
        self.root.after(0, lambda: self.log_text.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.log_text.delete(1.0, tk.END))
        self.root.after(0, lambda: self.log_text.config(state=tk.DISABLED))
        self._add_log("Starting analysis...")

        # Run in separate thread
        thread = threading.Thread(target=self._analysis_thread, daemon=True)
        thread.start()

    def _abort_analysis(self):
        """Request cancellation of the running analysis."""
        if not self.is_running:
            return
        self.cancel_requested = True
        self._add_log("Abort requested by user. Will stop after current component.")
        self._update_progress(self.progress_var.get(), "Aborting...", "")
        # visually indicate abort requested
        self.root.after(0, lambda: self.run_btn.config(state=tk.DISABLED))

    def _analysis_thread(self):
        """Analysis thread worker."""
        try:
            self._add_log(f"Loading dataset: {Path(self.file_path.get()).name}")
            context = AnalysisContext(self.dataset)
            report = Report()

            # Define components
            components_config = [
                ("Dataset Overview", lambda: DatasetOverviewComponent(context)),
                ("Missing Values", lambda: MissingValuesReport(context)),
                ("Exact Duplicates", lambda: ExactDuplicateDetectionComponent(context)),
                ("Near Duplicates", lambda: NearDuplicateDetectionComponent(context)),
                ("Outlier Detection", lambda: OutlierDetectionComponent(context)),
                ("Categorical Outliers", lambda: CategoricalOutlierDetectionComponent(context)),
            ]

            # Add label noise only if ML dataset with target
            if self.is_ml_dataset.get() and self.target_column.get():
                components_config.append(
                    ("Label Noise Detection",
                     lambda: LabelNoiseDetectionComponent(context, self.target_column.get()))
                )

            components_config.extend([
                ("Relational Consistency", lambda: RelationalConsistencyComponent(context)),
                ("Distribution Modeling", lambda: DistributionModelingComponent(context)),
                ("Composite Quality Score", lambda: CompositeQualityScoreComponent(context)),
                ("Dataset Summary", lambda: LLMDatasetSummaryComponent(context)),
            ])

            total_components = len(components_config)

            # Add and run components
            for i, (name, component_factory) in enumerate(components_config):
                if self.cancel_requested:
                    self._add_log("Analysis aborted by user. Stopping further components.")
                    break

                progress = (i / total_components) * 100
                self._update_progress(progress, "Analyzing...", f"Running: {name}")
                self._add_log(f"Running: {name}")

                component = component_factory()
                report.add_component(component)
                component.analyze()

                self._add_log(f"✓ Completed: {name}")

                # Store results
                try:
                    context.component_results[component.__class__.__name__] = component.summarize()
                except:
                    pass

            # If cancelled, skip report generation
            if self.cancel_requested:
                self._update_progress(0, "Aborted", "")
                self._add_log("Process aborted. Partial results saved in memory.")
                return

            # Generate report
            self._update_progress(95, "Generating report...", "Creating HTML report")
            self._add_log("Generating HTML report...")

            generator = HTMLReportGenerator("AEDA Data Quality Report")
            report_path = generator.generate(report.components, "data_quality_report.html")
            full_path = os.path.abspath(report_path)

            self._update_progress(100, "Complete!", f"Report saved: {full_path}")
            self._add_log(f"✓ Report saved: {full_path}")

            # Open report
            self.root.after(500, lambda: webbrowser.open('file://' + full_path))

            # Show success
            self.root.after(100, lambda: messagebox.showinfo(
                "Success",
                f"Report generated successfully!\n\nSaved to: {full_path}"
            ))

        except Exception as e:
            self._update_progress(0, f"Error: {str(e)[:50]}", "")
            self._add_log(f"✗ Error: {str(e)}")
            self.root.after(100, lambda: messagebox.showerror("Error", f"Analysis failed:\n{str(e)}"))

        finally:
            self.is_running = False
            self.cancel_requested = False
            self._stop_spinner()
            # restore run button text and command
            self.root.after(0, lambda: self.run_btn.config(
                state=tk.NORMAL,
                bg=ModernStyle.ACCENT,
                text='🚀 Generate Report',
                command=self._run_analysis
            ))

    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    app = AEDAApp()
    app.run()


if __name__ == "__main__":
    main()
