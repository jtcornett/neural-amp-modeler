# File: test_job_tab.py
# Created Date: Monday August 3rd 2026
# Author: Neural Amp Modeler Team

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from nam.train.gui.task_manager import TaskStatus, TrainingTask


def test_job_tab_close_button_state():
    """
    Test that Close Tab button is disabled when job is RUNNING or QUEUED,
    and enabled when job is COMPLETED, FAILED, or CANCELED.
    """
    # Mock CTk components to avoid GUI window rendering during tests
    mock_task = TrainingTask(
        task_id=1,
        name="test_job",
        input_path=Path("in.wav"),
        output_path=Path("out.nam"),
        train_destination=Path("dest"),
        epochs=10,
        status=TaskStatus.QUEUED,
    )
    mock_task_manager = MagicMock()
    close_cb = MagicMock()

    # Mock CTkTabview, CTkButton, CTkFrame, CTkProgressBar, CTkTextbox, CTkLabel
    mock_button = MagicMock()
    mock_button.cget.side_effect = lambda param: mock_button.state if param == "state" else None

    def mock_configure(**kwargs):
        if "state" in kwargs:
            mock_button.state = kwargs["state"]

    mock_button.configure.side_effect = mock_configure
    mock_button.state = "disabled"

    mock_tabview = MagicMock()
    mock_frame = MagicMock()
    mock_tabview.tab.return_value = mock_frame

    with (
        pytest.MonkeyPatch.context() as mp
    ):
        mp.setattr("customtkinter.CTkFrame", MagicMock())
        mp.setattr("customtkinter.CTkLabel", MagicMock())
        mp.setattr("customtkinter.CTkProgressBar", MagicMock())
        mp.setattr("customtkinter.CTkTextbox", MagicMock())
        mp.setattr("customtkinter.CTkFont", MagicMock())
        mp.setattr("customtkinter.CTkButton", lambda *args, **kwargs: mock_button)

        from nam.train.gui import JobTabUI

        # Create JobTabUI instance while task is QUEUED
        tab_ui = JobTabUI(
            tabview=mock_tabview,
            task=mock_task,
            task_manager=mock_task_manager,
            on_close_tab_cb=close_cb,
        )

        # Queued status: Close Tab should be disabled
        assert tab_ui.close_tab_button.cget("state") == "disabled"
        tab_ui._close_tab()
        close_cb.assert_not_called()

        # Running status: Close Tab should be disabled
        mock_task.status = TaskStatus.RUNNING
        tab_ui.update_status()
        assert tab_ui.close_tab_button.cget("state") == "disabled"
        tab_ui._close_tab()
        close_cb.assert_not_called()

        # Completed status: Close Tab should be enabled
        mock_task.status = TaskStatus.COMPLETED
        tab_ui.update_status()
        assert tab_ui.close_tab_button.cget("state") == "normal"
        tab_ui._close_tab()
        close_cb.assert_called_once_with(tab_ui.tab_name)
        close_cb.reset_mock()

        # Canceled status: Close Tab should be enabled
        mock_task.status = TaskStatus.CANCELED
        tab_ui.update_status()
        assert tab_ui.close_tab_button.cget("state") == "normal"
        tab_ui._close_tab()
        close_cb.assert_called_once_with(tab_ui.tab_name)
        close_cb.reset_mock()

        # Failed status: Close Tab should be enabled
        mock_task.status = TaskStatus.FAILED
        tab_ui.update_status()
        assert tab_ui.close_tab_button.cget("state") == "normal"
        tab_ui._close_tab()
        close_cb.assert_called_once_with(tab_ui.tab_name)
