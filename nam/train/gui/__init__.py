# File: gui.py
# Created Date: Saturday February 25th 2023
# Author: Steven Atkinson (steven@atkinson.mn)
# Modernized: Neural Amp Modeler Team

"""
GUI for training Neural Amp Modeler models with modern CustomTkinter interface,
non-blocking multi-threaded task queue (max 3 concurrent jobs), and per-job live log viewports.

Usage:
>>> from nam.train.gui import run
>>> run()
"""

import os
import re
import sys
import tkinter as tk
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import requests

try:
    import customtkinter as ctk

    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    _HAVE_CUSTOMTKINTER = True
except ImportError:
    _HAVE_CUSTOMTKINTER = False

try:
    import torch

    from nam import __version__
    from nam.data import Split
    from nam.models.metadata import GearType, ToneType, UserMetadata
    from nam.train import core as _core
    from nam.train import metadata as _metadata
    from nam.train._version import get_current_version
    _get_current_version = get_current_version
    from nam.train.gui._resources import settings
    from nam.train.gui.task_manager import TaskManager, TaskStatus, TrainingTask

    _INSTALL_IS_VALID = True
    _HAVE_ACCELERATOR = torch.cuda.is_available() or torch.backends.mps.is_available()
except ImportError:
    _INSTALL_IS_VALID = False
    _HAVE_ACCELERATOR = False

if _HAVE_ACCELERATOR:
    _DEFAULT_NUM_EPOCHS = 100
else:
    _DEFAULT_NUM_EPOCHS = 20


def _is_mac() -> bool:
    return sys.platform == "darwin"


def _get_latest_version_from_github() -> Optional[str]:
    url = "https://api.github.com/repos/sdatkinson/neural-amp-modeler/releases"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            releases = response.json()
            if releases and len(releases) > 0:
                tag = releases[0].get("tag_name", "")
                return tag.lstrip("v")
    except Exception:
        pass
    return None


class JobTabUI:
    """
    Represents a dynamic tab in the UI for a single training job.
    """

    def __init__(self, tabview: ctk.CTkTabview, task: TrainingTask, task_manager: TaskManager, on_close_tab_cb: Callable[[str], None]):
        self.tabview = tabview
        self.task = task
        self.task_manager = task_manager
        self.on_close_tab_cb = on_close_tab_cb
        self.tab_name = f"#{task.task_id}: {task.name[:16]}"

        self.tabview.add(self.tab_name)
        self.frame = self.tabview.tab(self.tab_name)

        self._build_ui()
        self.update_status()

    def _build_ui(self):
        # Header status card
        self.header_frame = ctk.CTkFrame(self.frame, corner_radius=10)
        self.header_frame.pack(fill="x", padx=10, pady=10)

        # Title & audio file details
        info_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True, padx=15, pady=10)

        self.title_label = ctk.CTkLabel(
            info_frame,
            text=f"Task #{self.task.task_id}: {self.task.name}",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        )
        self.title_label.pack(anchor="w")

        self.detail_label = ctk.CTkLabel(
            info_frame,
            text=f"Output File: {self.task.output_path.name}  |  Epochs: {self.task.epochs}",
            font=ctk.CTkFont(size=12),
            text_color="gray70",
            anchor="w",
        )
        self.detail_label.pack(anchor="w")

        self.metrics_label = ctk.CTkLabel(
            info_frame,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00E676",
            anchor="w",
        )
        self.metrics_label.pack(anchor="w", pady=(2, 0))

        # Status badge & controls
        control_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        control_frame.pack(side="right", padx=15, pady=10)

        self.status_badge = ctk.CTkLabel(
            control_frame,
            text=self.task.status.value.upper(),
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=6,
            fg_color=self._get_status_color(self.task.status),
            text_color="white",
            padx=12,
            pady=4,
        )
        self.status_badge.pack(side="top", pady=5)

        button_subframe = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_subframe.pack(side="top", pady=5)

        self.cancel_button = ctk.CTkButton(
            button_subframe,
            text="Cancel Job",
            fg_color="#D32F2F",
            hover_color="#B71C1C",
            width=90,
            height=28,
            command=self._cancel_job,
        )
        self.cancel_button.pack(side="left", padx=3)

        self.open_dir_button = ctk.CTkButton(
            button_subframe,
            text="Open Folder",
            fg_color="#2E7D32",
            hover_color="#1B5E20",
            width=90,
            height=28,
            command=self._open_export_dir,
            state="disabled",
        )
        self.open_dir_button.pack(side="left", padx=3)

        self.close_tab_button = ctk.CTkButton(
            button_subframe,
            text="Close Tab",
            fg_color="gray40",
            hover_color="gray30",
            width=80,
            height=28,
            command=self._close_tab,
            state="disabled",
        )
        self.close_tab_button.pack(side="left", padx=3)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.frame, height=8, corner_radius=4)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.progress_bar.set(0)

        # Console Log Viewport Card
        log_frame = ctk.CTkFrame(self.frame, corner_radius=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        log_title = ctk.CTkLabel(
            log_frame,
            text="Commandline Output Viewport",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        log_title.pack(anchor="w", padx=12, pady=(10, 5))

        self.log_textbox = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas" if os.name == "nt" else "Menlo", size=11),
            fg_color="#0D0D0D",
            text_color="#00FF66",
            wrap="word",
            corner_radius=6,
        )
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _get_status_color(self, status: TaskStatus) -> str:
        if status == TaskStatus.QUEUED:
            return "#F57C00"  # Amber
        elif status == TaskStatus.RUNNING:
            return "#0288D1"  # Cyan/Blue
        elif status == TaskStatus.COMPLETED:
            return "#2E7D32"  # Green
        elif status == TaskStatus.FAILED:
            return "#C62828"  # Red
        else:
            return "#616161"  # Grey

    def update_status(self):
        status = self.task.status
        self.status_badge.configure(
            text=status.value.upper(),
            fg_color=self._get_status_color(status),
        )
        if status in (TaskStatus.RUNNING, TaskStatus.QUEUED):
            self.cancel_button.configure(state="normal")
            self.open_dir_button.configure(state="disabled")
            self.close_tab_button.configure(state="disabled")
            if status == TaskStatus.RUNNING:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self.progress_bar.set(self.task.progress_ratio)
                if self.task.current_epoch > 0:
                    pct = int(self.task.progress_ratio * 100)
                    self.metrics_label.configure(
                        text=f"Progress: Epoch {self.task.current_epoch}/{self.task.total_epochs} ({pct}%)  |  Speed: {self.task.it_per_sec:.2f} it/s ({self.task.epoch_dur:.2f}s/ep)  |  ESR: {self.task.esr:.6f}"
                    )
                else:
                    self.metrics_label.configure(text="Training initializing...")
            else:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self.progress_bar.set(0.0)
                self.metrics_label.configure(text="Queued in line...")
        elif status == TaskStatus.COMPLETED:
            self.cancel_button.configure(state="disabled")
            self.open_dir_button.configure(state="normal")
            self.close_tab_button.configure(state="normal")
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(1.0)
            if self.task.esr > 0:
                self.metrics_label.configure(
                    text=f"Completed!  |  Final ESR: {self.task.esr:.6f}  |  Speed: {self.task.it_per_sec:.2f} it/s ({self.task.epoch_dur:.2f}s/ep)"
                )
            else:
                self.metrics_label.configure(text="Completed successfully!")
        elif status in (TaskStatus.FAILED, TaskStatus.CANCELED):
            self.cancel_button.configure(state="disabled")
            self.open_dir_button.configure(state="disabled")
            self.close_tab_button.configure(state="normal")
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(0.0)
            if status == TaskStatus.FAILED:
                self.metrics_label.configure(text="Training Failed", text_color="#EF5350")
            else:
                self.metrics_label.configure(text="Training Canceled", text_color="#FFB74D")

    def poll_logs(self):
        has_new = False
        while not self.task.log_queue.empty():
            try:
                line = self.task.log_queue.get_nowait()
                self.log_textbox.insert("end", line)
                has_new = True
            except Exception:
                break
        if has_new:
            self.log_textbox.see("end")

    def _cancel_job(self):
        self.task_manager.cancel_task(self.task.task_id)

    def _open_export_dir(self):
        path = str(self.task.train_destination.resolve())
        if sys.platform == "darwin":
            os.system(f'open "{path}"')
        elif os.name == "nt":
            os.startfile(path)
        else:
            os.system(f'xdg-open "{path}"')

    def _close_tab(self):
        if self.task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED):
            self.on_close_tab_cb(self.tab_name)


class ModernNAMGUI(ctk.CTk):
    """
    Main Application Window for Neural Amp Modeler.
    """

    def __init__(self):
        super().__init__()

        self.title("NAM Trainer")
        self.minsize(920, 680)
        self._center_window(1100, 800)

        self.task_manager = TaskManager(max_parallel=3)
        self.job_tabs: Dict[str, JobTabUI] = {}

        # Paths
        self.input_path: Optional[Path] = settings.get_last_path(settings.PathKey.INPUT_FILE)
        self.output_paths: List[Path] = []
        last_out = settings.get_last_path(settings.PathKey.OUTPUT_FILE)
        if last_out and last_out.exists():
            self.output_paths.append(last_out)

        self.train_destination: Optional[Path] = settings.get_last_path(settings.PathKey.TRAINING_DESTINATION)

        self._build_ui()
        self._check_for_updates()

        # Start periodic UI polling timer
        self.after(100, self._periodic_poll)

    def _center_window(self, width: int, height: int):
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = max(0, int((screen_width / 2) - (width / 2)))
        y = max(0, int((screen_height / 2) - (height / 2)))
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _build_ui(self):
        # Header Banner
        header = ctk.CTkFrame(self, corner_radius=0, fg_color="#1E1E1E", height=70)
        header.pack(fill="x", side="top")

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=20, pady=10)

        title_lbl = ctk.CTkLabel(
            title_frame,
            text="NAM TRAINER",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#4FC3F7",
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ctk.CTkLabel(
            title_frame,
            text=f"Version {__version__}  |  Deep Learning Amp & Pedal Modeling",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        )
        sub_lbl.pack(anchor="w")

        self.update_badge = ctk.CTkLabel(
            header,
            text="Checking for updates...",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        )
        self.update_badge.pack(side="right", padx=20)

        # Tabview
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=15)

        # Setup Tab
        self.tab_setup = self.tabview.add("Setup & Queue")
        self._build_setup_tab()

    def _build_setup_tab(self):
        # Scrollable container for setup tab
        setup_container = ctk.CTkScrollableFrame(self.tab_setup, fg_color="transparent")
        setup_container.pack(fill="both", expand=True)

        # 2-column layout
        left_col = ctk.CTkFrame(setup_container, fg_color="transparent")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=5)

        right_col = ctk.CTkFrame(setup_container, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=(8, 0), pady=5)

        # LEFT COLUMN - File Selection & Training Parameters
        # 1. File Selection Card
        file_card = ctk.CTkFrame(left_col, corner_radius=10)
        file_card.pack(fill="x", pady=(0, 10))

        file_card_lbl = ctk.CTkLabel(
            file_card, text="1. Audio File Configuration", font=ctk.CTkFont(size=14, weight="bold")
        )
        file_card_lbl.pack(anchor="w", padx=15, pady=(12, 8))

        # Input Path Selector
        in_btn_frame = ctk.CTkFrame(file_card, fg_color="transparent")
        in_btn_frame.pack(fill="x", padx=15, pady=4)

        self.in_btn = ctk.CTkButton(
            in_btn_frame, text="Select Input WAV...", width=160, command=self._browse_input_path
        )
        self.in_btn.pack(side="left")

        self.in_lbl = ctk.CTkLabel(
            in_btn_frame,
            text=str(self.input_path) if self.input_path else "No file selected",
            text_color="white" if self.input_path else "#E57373",
            anchor="w",
        )
        self.in_lbl.pack(side="left", padx=10, fill="x", expand=True)

        # Output Path Selector
        out_btn_frame = ctk.CTkFrame(file_card, fg_color="transparent")
        out_btn_frame.pack(fill="x", padx=15, pady=4)

        self.out_btn = ctk.CTkButton(
            out_btn_frame, text="Select Output WAV(s)...", width=160, command=self._browse_output_paths
        )
        self.out_btn.pack(side="left")

        self.out_lbl = ctk.CTkLabel(
            out_btn_frame,
            text=self._format_output_label_text(),
            text_color="white" if self.output_paths else "#E57373",
            anchor="w",
        )
        self.out_lbl.pack(side="left", padx=10, fill="x", expand=True)

        # Destination Directory Selector
        dest_btn_frame = ctk.CTkFrame(file_card, fg_color="transparent")
        dest_btn_frame.pack(fill="x", padx=15, pady=(4, 15))

        self.dest_btn = ctk.CTkButton(
            dest_btn_frame, text="Select Export Folder...", width=160, command=self._browse_destination
        )
        self.dest_btn.pack(side="left")

        self.dest_lbl = ctk.CTkLabel(
            dest_btn_frame,
            text=str(self.train_destination) if self.train_destination else "No folder selected",
            text_color="white" if self.train_destination else "#E57373",
            anchor="w",
        )
        self.dest_lbl.pack(side="left", padx=10, fill="x", expand=True)

        # 2. Options Card
        opts_card = ctk.CTkFrame(left_col, corner_radius=10)
        opts_card.pack(fill="x", pady=5)

        opts_card_lbl = ctk.CTkLabel(
            opts_card, text="2. Training Parameters", font=ctk.CTkFont(size=14, weight="bold")
        )
        opts_card_lbl.pack(anchor="w", padx=15, pady=(12, 8))

        # Epochs & Latency
        ep_frame = ctk.CTkFrame(opts_card, fg_color="transparent")
        ep_frame.pack(fill="x", padx=15, pady=4)

        ctk.CTkLabel(ep_frame, text="Epochs:", width=80, anchor="w").pack(side="left")
        self.epochs_entry = ctk.CTkEntry(ep_frame, width=80)
        self.epochs_entry.insert(0, str(_DEFAULT_NUM_EPOCHS))
        self.epochs_entry.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(ep_frame, text="Latency (samples):", width=120, anchor="w").pack(side="left")
        self.latency_entry = ctk.CTkEntry(ep_frame, width=80, placeholder_text="Auto")
        self.latency_entry.pack(side="left")

        # Checkboxes
        chk_frame = ctk.CTkFrame(opts_card, fg_color="transparent")
        chk_frame.pack(fill="x", padx=15, pady=(8, 15))

        self.var_ignore_checks = ctk.BooleanVar(value=False)
        self.chk_ignore = ctk.CTkCheckBox(
            chk_frame, text="Ignore Data Checks", variable=self.var_ignore_checks
        )
        self.chk_ignore.pack(side="left", padx=(0, 15))

        self.var_save_plot = ctk.BooleanVar(value=True)
        self.chk_plot = ctk.CTkCheckBox(
            chk_frame, text="Save Plot", variable=self.var_save_plot
        )
        self.chk_plot.pack(side="left", padx=(0, 15))

        self.var_silent = ctk.BooleanVar(value=False)
        self.chk_silent = ctk.CTkCheckBox(
            chk_frame, text="Silent Training", variable=self.var_silent
        )
        self.chk_silent.pack(side="left")

        # RIGHT COLUMN - User Metadata & Queue Overview
        # 1. Metadata Card
        meta_card = ctk.CTkFrame(right_col, corner_radius=10)
        meta_card.pack(fill="x", pady=(0, 10))

        meta_card_lbl = ctk.CTkLabel(
            meta_card, text="3. Model Metadata", font=ctk.CTkFont(size=14, weight="bold")
        )
        meta_card_lbl.pack(anchor="w", padx=15, pady=(12, 8))

        # Gear & Tone Type
        gt_frame = ctk.CTkFrame(meta_card, fg_color="transparent")
        gt_frame.pack(fill="x", padx=15, pady=4)

        ctk.CTkLabel(gt_frame, text="Gear Type:", width=80, anchor="w").pack(side="left")
        self.gear_dropdown = ctk.CTkOptionMenu(
            gt_frame, values=[g.value for g in GearType], width=140
        )
        self.gear_dropdown.set(GearType.AMP.value)
        self.gear_dropdown.pack(side="left", padx=(0, 15))

        ctk.CTkLabel(gt_frame, text="Tone Type:", width=80, anchor="w").pack(side="left")
        self.tone_dropdown = ctk.CTkOptionMenu(
            gt_frame, values=[t.value for t in ToneType], width=140
        )
        self.tone_dropdown.set(ToneType.OVERDRIVE.value)
        self.tone_dropdown.pack(side="left")

        # User Name & Description
        user_frame = ctk.CTkFrame(meta_card, fg_color="transparent")
        user_frame.pack(fill="x", padx=15, pady=(4, 15))

        ctk.CTkLabel(user_frame, text="Author Name:", width=80, anchor="w").pack(side="left")
        self.author_entry = ctk.CTkEntry(user_frame, width=140, placeholder_text="Optional")
        self.author_entry.pack(side="left", padx=(0, 15))

        ctk.CTkLabel(user_frame, text="Model Name:", width=80, anchor="w").pack(side="left")
        self.model_name_entry = ctk.CTkEntry(user_frame, width=140, placeholder_text="Auto from file")
        self.model_name_entry.pack(side="left")

        # 2. Queue Overview Card
        queue_card = ctk.CTkFrame(right_col, corner_radius=10)
        queue_card.pack(fill="both", expand=True, pady=5)

        queue_card_lbl = ctk.CTkLabel(
            queue_card, text="4. Task Queue Overview (Max 3 Parallel)", font=ctk.CTkFont(size=14, weight="bold")
        )
        queue_card_lbl.pack(anchor="w", padx=15, pady=(12, 8))

        self.queue_info_label = ctk.CTkLabel(
            queue_card, text="No tasks submitted yet.", text_color="gray70", anchor="w"
        )
        self.queue_info_label.pack(anchor="w", padx=15, pady=(0, 8))

        self.queue_list_frame = ctk.CTkScrollableFrame(queue_card, fg_color="transparent", height=140)
        self.queue_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Bottom Action Card - Start Training
        action_card = ctk.CTkFrame(left_col, corner_radius=10, fg_color="transparent")
        action_card.pack(fill="x", pady=10)

        self.start_btn = ctk.CTkButton(
            action_card,
            text="Add Training Task(s) to Queue",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#0288D1",
            hover_color="#0277BD",
            height=45,
            command=self._on_start_training,
        )
        self.start_btn.pack(fill="x")

    def _format_output_label_text(self) -> str:
        if not self.output_paths:
            return "No output file(s) selected"
        elif len(self.output_paths) == 1:
            return str(self.output_paths[0])
        else:
            return f"{len(self.output_paths)} files selected (Batch Re-amp Mode)"

    def _browse_input_path(self):
        initial_dir = self.input_path.parent if self.input_path else Path.home()
        filename = tk.filedialog.askopenfilename(
            title="Select Input Audio File",
            initialdir=initial_dir,
            filetypes=[("WAV files", "*.wav"), ("Audio files", "*.wav *.flac *.mp3")],
        )
        if filename:
            p = Path(filename)
            self.input_path = p
            settings.set_last_path(settings.PathKey.INPUT_FILE, p)
            self.in_lbl.configure(text=str(p), text_color="white")

    def _browse_output_paths(self):
        initial_dir = self.output_paths[0].parent if self.output_paths else Path.home()
        filenames = tk.filedialog.askopenfilenames(
            title="Select Output Audio File(s)",
            initialdir=initial_dir,
            filetypes=[("WAV files", "*.wav"), ("Audio files", "*.wav *.flac *.mp3")],
        )
        if filenames:
            paths = [Path(f) for f in filenames]
            self.output_paths = paths
            settings.set_last_path(settings.PathKey.OUTPUT_FILE, paths[0])
            self.out_lbl.configure(text=self._format_output_label_text(), text_color="white")

    def _browse_destination(self):
        initial_dir = self.train_destination if self.train_destination else Path.home()
        foldername = tk.filedialog.askdirectory(
            title="Select Training Destination Directory", initialdir=initial_dir
        )
        if foldername:
            p = Path(foldername)
            self.train_destination = p
            settings.set_last_path(settings.PathKey.TRAINING_DESTINATION, p)
            self.dest_lbl.configure(text=str(p), text_color="white")

    def _on_start_training(self):
        # Validation checks
        if not self.input_path or not self.input_path.exists():
            tk.messagebox.showerror("Error", "Please select a valid input WAV audio file.")
            return

        if not self.output_paths:
            tk.messagebox.showerror("Error", "Please select at least one output WAV audio file.")
            return

        if not self.train_destination or not self.train_destination.exists():
            tk.messagebox.showerror(
                "Error", "Please select a valid destination folder for exporting trained models."
            )
            return

        try:
            epochs = int(self.epochs_entry.get().strip())
        except ValueError:
            tk.messagebox.showerror("Error", "Epochs must be a valid integer.")
            return

        latency_text = self.latency_entry.get().strip()
        latency = int(latency_text) if latency_text else None

        # Build User Metadata
        user_meta = UserMetadata(
            name=self.author_entry.get().strip(),
            model_name=self.model_name_entry.get().strip(),
            gear_type=GearType(self.gear_dropdown.get()),
            tone_type=ToneType(self.tone_dropdown.get()),
        )

        # Create tasks for each output file
        first_created_tab = None
        for out_path in self.output_paths:
            custom_model_name = self.model_name_entry.get().strip()
            task_name = custom_model_name if custom_model_name else re.sub(r"\.wav$", "", out_path.name)

            task = self.task_manager.add_task(
                name=task_name,
                input_path=self.input_path,
                output_path=out_path,
                train_destination=self.train_destination,
                epochs=epochs,
                latency=latency,
                silent=self.var_silent.get(),
                save_plot=self.var_save_plot.get(),
                ignore_checks=self.var_ignore_checks.get(),
                user_metadata=user_meta,
            )

            # Create Tab UI for this task
            job_tab = JobTabUI(
                tabview=self.tabview,
                task=task,
                task_manager=self.task_manager,
                on_close_tab_cb=self._on_close_job_tab,
            )
            self.job_tabs[job_tab.tab_name] = job_tab
            if first_created_tab is None:
                first_created_tab = job_tab.tab_name

        self._refresh_queue_overview()

        # Switch view to the newly created task tab
        if first_created_tab:
            self.tabview.set(first_created_tab)

    def _on_close_job_tab(self, tab_name: str):
        if tab_name in self.job_tabs:
            del self.job_tabs[tab_name]
            try:
                self.tabview.delete(tab_name)
            except Exception:
                pass

    def _refresh_queue_overview(self):
        active = self.task_manager.get_active_count()
        queued = self.task_manager.get_queued_count()
        total = len(self.task_manager.tasks)

        self.queue_info_label.configure(
            text=f"Total Tasks: {total}  |  Active Running: {active}/3  |  Queued: {queued}"
        )

        # Clear existing items in list frame
        for child in self.queue_list_frame.winfo_children():
            child.destroy()

        for task in self.task_manager.tasks:
            row = ctk.CTkFrame(self.queue_list_frame, fg_color="#2B2B2B", corner_radius=6)
            row.pack(fill="x", pady=2, padx=2)

            name_lbl = ctk.CTkLabel(
                row,
                text=f"#{task.task_id}: {task.name}",
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w",
            )
            name_lbl.pack(side="left", padx=10, pady=5)

            status_lbl = ctk.CTkLabel(
                row,
                text=task.status.value.upper(),
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=JobTabUI._get_status_color(None, task.status),
            )
            status_lbl.pack(side="right", padx=10, pady=5)

    def _periodic_poll(self):
        # Poll logs and status updates for all active job tabs
        for job_tab in list(self.job_tabs.values()):
            job_tab.poll_logs()
            job_tab.update_status()

        self._refresh_queue_overview()
        self.after(100, self._periodic_poll)

    def _check_for_updates(self):
        def _check():
            latest = _get_latest_version_from_github()
            if latest:
                try:
                    curr = get_current_version()
                    if latest > curr:
                        self.update_badge.configure(
                            text=f"Update Available: v{latest}!", text_color="#FFB74D"
                        )
                    else:
                        self.update_badge.configure(text=f"Up to date (v{curr})", text_color="gray60")
                except Exception:
                    self.update_badge.configure(text=f"Latest release: v{latest}", text_color="gray60")
            else:
                self.update_badge.configure(text=f"Version {__version__}", text_color="gray60")

        import threading

        threading.Thread(target=_check, daemon=True).start()


def run():
    if not _HAVE_CUSTOMTKINTER:
        root = tk.Tk()
        root.title("Error")
        tk.Label(
            root, text="customtkinter is required to run the NAM GUI.\nPlease run: pip install customtkinter"
        ).pack(padx=20, pady=20)
        root.mainloop()
        return

    if _INSTALL_IS_VALID:
        app = ModernNAMGUI()
        app.mainloop()
        print("Shut down NAM trainer GUI.")
    else:
        root = tk.Tk()
        root.title("Install Error")
        tk.Label(
            root, text="The NAM training software has not been installed correctly."
        ).pack(padx=20, pady=20)
        root.mainloop()


GUI = ModernNAMGUI

if __name__ == "__main__":
    run()
