# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-framework adapters - the "which agent framework am I tuning" boundary."""

from hep.adapters.base import HarnessAdapter
from hep.adapters.deepagents import DeepAgentsAdapter
from hep.adapters.deepagents_terminal_two import DeepAgentsTerminalTwoAdapter

__all__ = ["DeepAgentsAdapter", "DeepAgentsTerminalTwoAdapter", "HarnessAdapter"]
