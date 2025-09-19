# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for creating explanations for vision models."""

from .version import name, version
from .wrappers.ultralytics_yolo import UltralyticsYoloWrapper

__name__ = name
__version__ = version
