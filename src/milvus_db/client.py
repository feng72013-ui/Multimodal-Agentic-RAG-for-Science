from __future__ import annotations

import argparse
import json
import sys

from pymilvus import MilvusClient, MilvusException

from .config import MilvusSettings, add_connection_args, settings_from_args


def create_client(settings: MilvusSettings | None = None) -> MilvusClient:
    settings = settings or MilvusSettings()
    kwargs = {
        "uri": settings.uri,
        "user": settings.user,
        "password": settings.password,
        "timeout": settings.timeout,
    }
    if settings.token:
        kwargs["token"] = settings.token
    if settings.db_name:
        kwargs["db_name"] = settings.db_name
    return MilvusClient(**kwargs)


def connection_summary(settings: MilvusSettings) -> dict:
    return {
        "uri": settings.uri,
        "user": settings.user,
        "db_name": settings.db_name or "default",
        "collection_name": settings.collection_name,
        "timeout": settings.timeout,
    }


def ping(settings: MilvusSettings) -> dict:
    client = create_client(settings)
    databases = client.list_databases(timeout=settings.timeout)
    collections = client.list_collections(timeout=settings.timeout)
    return {
        "connection": connection_summary(settings),
        "databases": databases,
        "collections": collections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Milvus connection.")
    add_connection_args(parser)
    parser.add_argument("--ping", action="store_true", help="Connect and list databases/collections.")
    args = parser.parse_args()
    settings = settings_from_args(args)

    print(json.dumps(connection_summary(settings), ensure_ascii=False, indent=2))
    if args.ping:
        try:
            print(json.dumps(ping(settings), ensure_ascii=False, indent=2))
        except MilvusException as exc:
            print(f"[Milvus] connection failed: {exc}", file=sys.stderr)
            print(
                "[Milvus] check MILVUS_URI, network reachability, server status, and credentials.",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
