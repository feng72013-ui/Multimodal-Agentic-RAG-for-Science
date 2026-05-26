from __future__ import annotations

import argparse
import json

from .client import create_client
from .config import add_connection_args, settings_from_args


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Milvus databases.")
    add_connection_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--name", required=True)

    describe_parser = subparsers.add_parser("describe")
    describe_parser.add_argument("--name", required=True)

    drop_parser = subparsers.add_parser("drop")
    drop_parser.add_argument("--name", required=True)
    drop_parser.add_argument("--yes", action="store_true")

    use_parser = subparsers.add_parser("use")
    use_parser.add_argument("--name", required=True)

    args = parser.parse_args()
    settings = settings_from_args(args)
    client = create_client(settings)

    if args.command == "list":
        print(json.dumps(client.list_databases(timeout=settings.timeout), ensure_ascii=False, indent=2))
    elif args.command == "create":
        client.create_database(args.name, timeout=settings.timeout)
        print(f"[Milvus] created database: {args.name}")
    elif args.command == "describe":
        print(json.dumps(client.describe_database(args.name), ensure_ascii=False, indent=2, default=str))
    elif args.command == "drop":
        if not args.yes:
            raise SystemExit("Refusing to drop database without --yes.")
        client.drop_database(args.name)
        print(f"[Milvus] dropped database: {args.name}")
    elif args.command == "use":
        client.use_database(args.name)
        print(f"[Milvus] using database for this client: {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
