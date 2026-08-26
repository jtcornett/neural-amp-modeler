# File: task_manager.py
# Created Date: Monday August 3rd 2026
# Author: Neural Amp Modeler Team

import io
import queue
import re
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg", force=True)

from nam.models.metadata import UserMetadata
from nam.train import core as _core
from nam.train import metadata as _metadata


class TaskStatus(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELED = "Canceled"


class LogStreamWriter(io.TextIOBase):
    """
    Custom stream writer that forwards stdout/stderr writes to a queue for UI updates.
    """

    def __init__(self, log_queue: queue.Queue, original_stream=None):
        super().__init__()
        self.log_queue = log_queue
        self.original_stream = original_stream

    def write(self, s: str) -> int:
        if s:
            clean_s = s.replace("\r\n", "\n").replace("\r", "\n")
            self.log_queue.put(clean_s)
            if self.original_stream:
                try:
                    self.original_stream.write(s)
                    self.original_stream.flush()
                except Exception:
                    pass
        return len(s)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass


@dataclass
class TrainingTask:
    task_id: int
    name: str
    input_path: Path
    output_path: Path
    train_destination: Path
    epochs: int = 100
    latency: Optional[int] = None
    silent: bool = False
    save_plot: bool = False
    ignore_checks: bool = False
    threshold_esr: Optional[float] = None
    user_metadata: Optional[UserMetadata] = None
    core_kwargs: Dict[str, Any] = field(default_factory=dict)

    status: TaskStatus = TaskStatus.QUEUED
    log_queue: queue.Queue = field(default_factory=queue.Queue)
    thread: Optional[threading.Thread] = None
    cancel_requested: bool = False
    error_message: Optional[str] = None

    current_epoch: int = 0
    total_epochs: int = 100
    progress_ratio: float = 0.0
    it_per_sec: float = 0.0
    epoch_dur: float = 0.0
    esr: float = 0.0

    def cancel(self):
        self.cancel_requested = True
        if self.status == TaskStatus.QUEUED:
            self.status = TaskStatus.CANCELED
            self.log_queue.put("\nTask canceled before starting.\n")


class TaskManager:
    """
    Manages a queue of training tasks with a maximum concurrency limit of 3 parallel active tasks.
    """

    def __init__(self, max_parallel: int = 3):
        self.max_parallel = max_parallel
        self.tasks: List[TrainingTask] = []
        self.lock = threading.Lock()
        self._next_id = 1
        self.status_change_callbacks: List[Callable[[TrainingTask], None]] = []

    def register_status_callback(self, callback: Callable[[TrainingTask], None]):
        self.status_change_callbacks.append(callback)

    def _notify_status_change(self, task: TrainingTask):
        for cb in self.status_change_callbacks:
            try:
                cb(task)
            except Exception as e:
                print(f"Error in task status callback: {e}")

    def add_task(
        self,
        name: str,
        input_path: Path,
        output_path: Path,
        train_destination: Path,
        epochs: int = 100,
        latency: Optional[int] = None,
        silent: bool = False,
        save_plot: bool = False,
        ignore_checks: bool = False,
        threshold_esr: Optional[float] = None,
        user_metadata: Optional[UserMetadata] = None,
        core_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TrainingTask:
        with self.lock:
            task = TrainingTask(
                task_id=self._next_id,
                name=name,
                input_path=Path(input_path),
                output_path=Path(output_path),
                train_destination=Path(train_destination),
                epochs=epochs,
                latency=latency,
                silent=silent,
                save_plot=save_plot,
                ignore_checks=ignore_checks,
                threshold_esr=threshold_esr,
                user_metadata=user_metadata or UserMetadata(),
                core_kwargs=core_kwargs or {},
            )
            self._next_id += 1
            self.tasks.append(task)

        task.log_queue.put(f"Task added to queue (ID: {task.task_id})...\n")
        self._notify_status_change(task)
        self._process_queue()
        return task

    def cancel_task(self, task_id: int):
        with self.lock:
            task = next((t for t in self.tasks if t.task_id == task_id), None)
            if task and task.status in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                task.cancel()
                self._notify_status_change(task)
        self._process_queue()

    def get_active_count(self) -> int:
        with self.lock:
            return sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING)

    def get_queued_count(self) -> int:
        with self.lock:
            return sum(1 for t in self.tasks if t.status == TaskStatus.QUEUED)

    def _process_queue(self):
        with self.lock:
            active_count = sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING)
            if active_count >= self.max_parallel:
                return

            queued_tasks = [t for t in self.tasks if t.status == TaskStatus.QUEUED and not t.cancel_requested]
            slots_available = self.max_parallel - active_count

            for task in queued_tasks[:slots_available]:
                task.status = TaskStatus.RUNNING
                self._notify_status_change(task)
                t = threading.Thread(target=self._run_task, args=(task,), daemon=True)
                task.thread = t
                t.start()

    def _run_task(self, task: TrainingTask):
        import logging

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        writer = LogStreamWriter(task.log_queue, original_stdout)
        sys.stdout = writer
        sys.stderr = writer

        log_handler = logging.StreamHandler(writer)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        try:
            print(f"=== Starting Training Task #{task.task_id}: {task.name} ===")
            print(f"Input file: {task.input_path}")
            print(f"Output file: {task.output_path}")
            print(f"Destination: {task.train_destination}")
            print(f"Epochs: {task.epochs}")

            if task.cancel_requested:
                task.status = TaskStatus.CANCELED
                print("Task was canceled prior to start execution.")
                return

            basename = re.sub(r"\.wav$", "", str(task.output_path.name))

            def _on_progress(epoch, max_epochs, epoch_dur, it_per_sec, esr):
                task.current_epoch = epoch
                task.total_epochs = max_epochs
                task.progress_ratio = max(0.0, min(1.0, epoch / max_epochs)) if max_epochs > 0 else 0.0
                task.epoch_dur = epoch_dur
                task.it_per_sec = it_per_sec
                task.esr = esr
                self._notify_status_change(task)

            train_output = _core.train(
                str(task.input_path),
                str(task.output_path),
                str(task.train_destination),
                epochs=task.epochs,
                latency=task.latency,
                silent=task.silent,
                save_plot=task.save_plot,
                modelname=basename,
                ignore_checks=task.ignore_checks,
                local=True,
                threshold_esr=task.threshold_esr,
                user_metadata=task.user_metadata,
                is_canceled_fn=lambda: task.cancel_requested,
                progress_callback=_on_progress,
                **task.core_kwargs,
            )

            if task.cancel_requested:
                task.status = TaskStatus.CANCELED
                print("Task canceled during execution.")
                return

            if train_output is None or train_output.model is None:
                task.status = TaskStatus.FAILED
                task.error_message = "Training returned no model output."
                print("\n[ERROR] Model training failed! Export skipped.")
            else:
                print("\nModel training complete! Exporting trained model...")
                train_output.model.net.export(
                    str(task.train_destination),
                    basename=basename,
                    user_metadata=task.user_metadata,
                    other_metadata={
                        _metadata.TRAINING_KEY: train_output.metadata.model_dump()
                    },
                )
                print(f"Export successful to: {task.train_destination}")
                task.status = TaskStatus.COMPLETED

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            print(f"\n[ERROR] Exception encountered during training: {e}")
            import traceback

            traceback.print_exc(file=writer)

        finally:
            try:
                root_logger.removeHandler(log_handler)
            except Exception:
                pass
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self._notify_status_change(task)
            self._process_queue()
