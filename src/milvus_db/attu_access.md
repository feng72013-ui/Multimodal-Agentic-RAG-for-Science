# Attu Access Steps

Attu is a web UI for Milvus. The important rule is that `MILVUS_URL` must be reachable from inside
the Attu Docker container.

## Start Attu For This Machine

```bash
docker rm -f attu-recsys
docker run -d --rm \
  --name attu-recsys \
  --network dataset_default \
  -p 8000:3000 \
  -e MILVUS_URL=milvus-standalone:19530 \
  zilliz/attu:v2.4
```

Then open:

```text
http://localhost:8000
```

Login values:

```text
Milvus Address: milvus-standalone:19530
Username: root
Password: Milvus
```

Why this works here:

```text
milvus-standalone is attached to Docker network dataset_default.
Attu must be attached to the same network to resolve milvus-standalone.
```

## If Milvus Runs On The Host Network

Do not use `127.0.0.1:19530` unless Attu uses host networking. From inside the container,
`127.0.0.1` means the Attu container itself.

On Linux, one practical option is:

```bash
docker run --rm \
  --network host \
  -e MILVUS_URL=127.0.0.1:19530 \
  zilliz/attu:v2.4
```

Then open:

```text
http://localhost:3000
```

## Quick Connectivity Check Before Opening Attu

From the host:

```bash
nc -zv 172.16.229.202 19530
```

From a temporary container:

```bash
docker run --rm --network dataset_default busybox nc -zv milvus-standalone 19530
```

If the container check fails but the host check works, the issue is Docker networking rather than
Milvus credentials.
