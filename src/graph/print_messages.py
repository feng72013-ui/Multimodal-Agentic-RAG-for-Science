from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage, convert_to_messages


def pretty_print_messages(update, last_message: bool = False) -> None:
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        is_subgraph = True

    for _, node_update in update.items():
        if not node_update or not hasattr(node_update, "__iter__"):
            continue
        if "messages" not in node_update:
            if isinstance(node_update, Sequence) and node_update and isinstance(node_update[-1], BaseMessage):
                pretty_print_message(node_update[-1])
            continue

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]
        for message in messages:
            pretty_print_message(message, indent=is_subgraph)
        print()


def pretty_print_message(message, indent: bool = False) -> None:
    pretty_message = message.pretty_repr(html=True)
    if indent:
        pretty_message = "\n".join("\t" + line for line in pretty_message.splitlines())
    print(pretty_message)

