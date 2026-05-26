from __future__ import annotations

import argparse
import json

from .client import create_client
from .config import add_connection_args, settings_from_args


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Milvus users and roles.")
    add_connection_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-users")
    subparsers.add_parser("list-roles")

    create_user = subparsers.add_parser("create-user")
    create_user.add_argument("--name", required=True)
    create_user.add_argument("--new-password", required=True)

    describe_user = subparsers.add_parser("describe-user")
    describe_user.add_argument("--name", required=True)

    drop_user = subparsers.add_parser("drop-user")
    drop_user.add_argument("--name", required=True)
    drop_user.add_argument("--yes", action="store_true")

    create_role = subparsers.add_parser("create-role")
    create_role.add_argument("--name", required=True)

    describe_role = subparsers.add_parser("describe-role")
    describe_role.add_argument("--name", required=True)

    drop_role = subparsers.add_parser("drop-role")
    drop_role.add_argument("--name", required=True)
    drop_role.add_argument("--force", action="store_true")
    drop_role.add_argument("--yes", action="store_true")

    grant = subparsers.add_parser("grant-role")
    grant.add_argument("--user-name", required=True)
    grant.add_argument("--role-name", required=True)

    revoke = subparsers.add_parser("revoke-role")
    revoke.add_argument("--user-name", required=True)
    revoke.add_argument("--role-name", required=True)

    args = parser.parse_args()
    settings = settings_from_args(args)
    client = create_client(settings)

    if args.command == "list-users":
        print(json.dumps(client.list_users(timeout=settings.timeout), ensure_ascii=False, indent=2))
    elif args.command == "list-roles":
        print(json.dumps(client.list_roles(timeout=settings.timeout), ensure_ascii=False, indent=2))
    elif args.command == "create-user":
        client.create_user(args.name, args.new_password, timeout=settings.timeout)
        print(f"[Milvus] created user: {args.name}")
    elif args.command == "describe-user":
        print(json.dumps(client.describe_user(args.name, timeout=settings.timeout), ensure_ascii=False, indent=2))
    elif args.command == "drop-user":
        if not args.yes:
            raise SystemExit("Refusing to drop user without --yes.")
        client.drop_user(args.name, timeout=settings.timeout)
        print(f"[Milvus] dropped user: {args.name}")
    elif args.command == "create-role":
        client.create_role(args.name, timeout=settings.timeout)
        print(f"[Milvus] created role: {args.name}")
    elif args.command == "describe-role":
        print(json.dumps(client.describe_role(args.name, timeout=settings.timeout), ensure_ascii=False, indent=2))
    elif args.command == "drop-role":
        if not args.yes:
            raise SystemExit("Refusing to drop role without --yes.")
        client.drop_role(args.name, force_drop=args.force, timeout=settings.timeout)
        print(f"[Milvus] dropped role: {args.name}")
    elif args.command == "grant-role":
        client.grant_role(args.user_name, args.role_name, timeout=settings.timeout)
        print(f"[Milvus] granted role {args.role_name} to {args.user_name}")
    elif args.command == "revoke-role":
        client.revoke_role(args.user_name, args.role_name, timeout=settings.timeout)
        print(f"[Milvus] revoked role {args.role_name} from {args.user_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

