from __future__ import annotations

import argparse
import json

from .client import create_client
from .config import add_connection_args, settings_from_args
from .schema import (
    build_citation_edge_index_params,
    build_citation_edge_schema,
    build_paper_profile_index_params,
    build_paper_profile_schema,
    build_recsys_index_params,
    build_recsys_schema,
    build_reference_index_params,
    build_reference_schema,
)


def create_collection(settings, schema_name: str, drop_existing: bool = False) -> None:
    client = create_client(settings)
    if client.has_collection(settings.collection_name):
        if not drop_existing:
            print(f"[Milvus] collection already exists: {settings.collection_name}")
            return
        client.drop_collection(settings.collection_name)
        print(f"[Milvus] dropped collection: {settings.collection_name}")

    if schema_name == "reference":
        schema = build_reference_schema(client)
        index_params = build_reference_index_params(client, settings)
    elif schema_name == "paper_profile":
        schema = build_paper_profile_schema(client, settings)
        index_params = build_paper_profile_index_params(client, settings)
    elif schema_name == "citation_edge":
        schema = build_citation_edge_schema(client, settings)
        index_params = build_citation_edge_index_params(client, settings)
    else:
        schema = build_recsys_schema(client, settings)
        index_params = build_recsys_index_params(client, settings)

    client.create_collection(
        collection_name=settings.collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"[Milvus] created collection: {settings.collection_name} ({schema_name})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Milvus collections.")
    add_connection_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    describe_parser = subparsers.add_parser("describe")
    describe_parser.add_argument("--collection-name", default=None)

    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--collection-name", default=None)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument(
        "--schema",
        choices=["recsys", "reference", "paper_profile", "citation_edge"],
        default="recsys",
    )
    create_parser.add_argument("--drop-existing", action="store_true")
    create_parser.add_argument("--collection-name", default=None)

    drop_parser = subparsers.add_parser("drop")
    drop_parser.add_argument("--collection-name", default=None)
    drop_parser.add_argument("--yes", action="store_true", help="Required for destructive drops.")

    load_parser = subparsers.add_parser("load")
    load_parser.add_argument("--collection-name", default=None)

    release_parser = subparsers.add_parser("release")
    release_parser.add_argument("--collection-name", default=None)

    args = parser.parse_args()
    settings = settings_from_args(args)
    client = create_client(settings)

    if args.command == "list":
        print(json.dumps(client.list_collections(timeout=settings.timeout), ensure_ascii=False, indent=2))
    elif args.command == "describe":
        name = args.collection_name or settings.collection_name
        print(json.dumps(client.describe_collection(name), ensure_ascii=False, indent=2, default=str))
    elif args.command == "stats":
        name = args.collection_name or settings.collection_name
        print(json.dumps(client.get_collection_stats(name), ensure_ascii=False, indent=2, default=str))
    elif args.command == "create":
        if args.collection_name:
            settings.collection_name = args.collection_name
        create_collection(settings, args.schema, drop_existing=args.drop_existing)
    elif args.command == "drop":
        if not args.yes:
            raise SystemExit("Refusing to drop collection without --yes.")
        if args.collection_name:
            settings.collection_name = args.collection_name
        client.drop_collection(settings.collection_name)
        print(f"[Milvus] dropped collection: {settings.collection_name}")
    elif args.command == "load":
        name = args.collection_name or settings.collection_name
        client.load_collection(name)
        print(f"[Milvus] loaded collection: {name}")
    elif args.command == "release":
        name = args.collection_name or settings.collection_name
        client.release_collection(name)
        print(f"[Milvus] released collection: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
