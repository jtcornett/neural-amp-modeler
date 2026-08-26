# File: test_task_manager.py
# Created Date: Monday August 3rd 2026
# Author: Neural Amp Modeler Team

import time
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from nam.train.gui.task_manager import TaskManager, TaskStatus
from nam.train.core import _CancellationCallback


def test_task_manager_concurrency_limit(tmp_path):
    tm = TaskManager(max_parallel=3)

    executed_count = 0

    def mock_train(*args, **kwargs):
        nonlocal executed_count
        executed_count += 1
        time.sleep(0.1)
        mock_output = MagicMock()
        mock_output.model = MagicMock()
        return mock_output

    with patch("nam.train.core.train", side_effect=mock_train):
        # Queue 5 tasks
        tasks = []
        for i in range(5):
            t = tm.add_task(
                name=f"task_{i}",
                input_path=tmp_path / "in.wav",
                output_path=tmp_path / f"out_{i}.wav",
                train_destination=tmp_path / "dest",
                epochs=1,
            )
            tasks.append(t)

        # Immediately check: active tasks should be at most 3
        assert tm.get_active_count() <= 3
        assert tm.get_queued_count() >= 2

        # Wait for all tasks to finish
        for t in tasks:
            if t.thread:
                t.thread.join(timeout=2.0)

        # After all threads finish, active and queued should be 0
        assert tm.get_active_count() == 0
        assert tm.get_queued_count() == 0
        assert all(t.status == TaskStatus.COMPLETED for t in tasks)


def test_task_manager_cancel_task(tmp_path):
    tm = TaskManager(max_parallel=1)

    def mock_train(*args, **kwargs):
        time.sleep(0.5)

    with patch("nam.train.core.train", side_effect=mock_train):
        t1 = tm.add_task(
            name="task_1",
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "out_1.wav",
            train_destination=tmp_path / "dest",
        )
        t2 = tm.add_task(
            name="task_2",
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "out_2.wav",
            train_destination=tmp_path / "dest",
        )

        assert t2.status == TaskStatus.QUEUED
        tm.cancel_task(t2.task_id)
        assert t2.status == TaskStatus.CANCELED


def test_cancellation_callback():
    canceled = False
    cb = _CancellationCallback(lambda: canceled)
    trainer = MagicMock()
    trainer.should_stop = False

    cb.on_train_batch_end(trainer, MagicMock(), None, None, 0)
    assert trainer.should_stop is False

    canceled = True
    cb.on_train_batch_end(trainer, MagicMock(), None, None, 0)
    assert trainer.should_stop is True


if __name__ == "__main__":
    pytest.main([__file__])
