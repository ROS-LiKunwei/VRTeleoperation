import logging

from beavr.scripts.control_robot import _RecordTerminalLogFilter


def _record(level: int, message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_record_terminal_log_filter_keeps_dataset_prompts_and_errors():
    log_filter = _RecordTerminalLogFilter()

    assert log_filter.filter(_record(logging.INFO, "Recording episode 0"))
    assert log_filter.filter(_record(logging.INFO, "Reset the environment"))
    assert log_filter.filter(_record(logging.ERROR, "real error"))
    assert not log_filter.filter(_record(logging.INFO, "SYSMO-32 hand action: left action=1"))
    assert not log_filter.filter(_record(logging.WARNING, "[SYSMO-32 Safety] CartesianTarget stale"))
