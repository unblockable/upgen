#!/usr/bin/env python3

import typing


def str_of_optional(value: typing.Optional[typing.Any], none_str: str = "") -> str:
    if value is None:
        return none_str
    return str(value)
