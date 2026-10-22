"""
Telemetry and C++ background log suppressor.
Silences Google MediaPipe's internal portable_clearcut_uploader.cc,
TensorFlow Lite feedback managers, and protobuf/absl C++ error spam
from polluting the console or terminal while safely preserving
Python's standard logging, stdout, and stderr.
"""

import os
import sys
import ctypes

_SUPPRESSED = False


def silence_clearcut_telemetry() -> None:
    """
    Suppresses MediaPipe's internal C++ telemetry (portable_clearcut_uploader)
    and underlying TensorFlow Lite / absl log warnings from spamming stderr.
    """
    global _SUPPRESSED
    if _SUPPRESSED:
        return
    _SUPPRESSED = True

    # 1. Environment variables for Google Logging (glog) and TensorFlow
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["GLOG_stderrthreshold"] = "3"
    os.environ["GLOG_logtostderr"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

    # 2. Abseil logging verbosity (if absl is installed)
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.FATAL)
        absl_logging.set_stderrthreshold(absl_logging.FATAL)
    except Exception:
        pass

    # 3. Suppress low-level C runtime stderr (file descriptor 2)
    # MediaPipe's C++ portable_clearcut_uploader writes directly to CRT fd 2.
    try:
        if sys.stderr and hasattr(sys.stderr, "fileno") and hasattr(os, "dup2"):
            try:
                sys.stderr.flush()
                real_stderr_fd = os.dup(2)
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, 2)
                os.close(devnull_fd)
                # Keep Python's sys.stderr functional by rebinding to the preserved duplicate fd
                sys.stderr = os.fdopen(real_stderr_fd, "w", encoding="utf-8", errors="replace")
            except Exception:
                pass
    except Exception:
        pass

    # 4. On Windows, redirect Win32 OS-level STD_ERROR_HANDLE to NUL device
    if sys.platform.startswith("win"):
        try:
            kernel32 = ctypes.windll.kernel32
            STD_ERROR_HANDLE = -12
            GENERIC_WRITE = 0x40000000
            FILE_SHARE_READ = 1
            FILE_SHARE_WRITE = 2
            OPEN_EXISTING = 3
            INVALID_HANDLE_VALUE = -1

            nul_handle = kernel32.CreateFileW(
                "NUL",
                GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                None,
                OPEN_EXISTING,
                0,
                None
            )
            if nul_handle != INVALID_HANDLE_VALUE:
                kernel32.SetStdHandle(STD_ERROR_HANDLE, nul_handle)
        except Exception:
            pass


# Automatically silence upon module import as well
silence_clearcut_telemetry()
