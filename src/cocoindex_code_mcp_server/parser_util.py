#!/usr/bin/env python3
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 aanno <aanno@users.noreply.github.com>
#
# This file is part of cocoindex_code_mcp_server from
# https://github.com/aanno/cocoindex-code-mcp-server
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Iterable, Sized
from typing import Any, Dict


def is_empty_iterable(value: Any) -> bool:
    return (
        isinstance(value, Iterable)
        and not isinstance(value, (str, bytes))
        and isinstance(value, Sized)
        and len(value) == 0
    )


def update_defaults(d: Dict[str, Any], defaults: Dict[str, Any]) -> None:
    for k, v in defaults.items():
        if isinstance(v, Iterable) and not isinstance(v, (str, bytes)) and isinstance(v, Sized) and len(v) > 0:
            current_value = d.get(k)
            if current_value is None or is_empty_iterable(current_value):
                d[k] = v
        else:
            d.setdefault(k, v)
