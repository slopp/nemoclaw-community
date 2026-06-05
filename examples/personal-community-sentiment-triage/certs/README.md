# certs/

Optional drop point for additional CA certificates to trust inside the Hermes
sandbox image. Files placed here are baked in at build time so `curl` and
other TLS clients inside the sandbox trust the corresponding roots.

## When you need this

If the network running this example performs TLS interception (for example, a
proxy that re-signs HTTPS traffic with its own CA), agent calls to the public
internet will fail with errors like `SSL certificate problem: self-signed
certificate in certificate chain`. Place the interception CA(s) here to
register them as trusted roots.

If TLS traffic is not being intercepted, leave this directory empty. The
Dockerfile's `update-ca-certificates` step becomes a no-op and the build
succeeds as-is.

## Host-local HTTPS inference tunnels

Some NVIDIA environments expose `https://inference-api.nvidia.com/v1` through
a host-local tunnel such as Teleport plus Caddy, with Caddy presenting a
certificate signed by a private or mkcert root. In that case:

1. Keep `NEMOCLAW_ENDPOINT_URL` set to the public HTTPS base URL.
2. Set `NEMOCLAW_ENDPOINT_URL_REQUIRES_HOST_TUNNEL=true` in `.env` so
   sandbox DNS resolves that public hostname to the OpenShell host bridge.
3. Copy the tunnel's root CA into this directory with a `.crt` extension
   before running `scripts/bring-up.sh`.

The generated sandbox `/etc/hosts` entry keeps the inference URL host intact,
for example:

```text
172.18.0.1 inference-api.nvidia.com
```

That is different from host-backed services such as Phoenix or the ATIF relay:
those services use `host.openshell.internal` directly in their URLs because
their certificates and routing are host-local by design. For NVIDIA inference,
the public hostname remains part of the HTTPS request and certificate check.

For mkcert-backed Caddy, the root is usually:

```bash
cp "$(mkcert -CAROOT)/rootCA.pem" certs/nvidia-inference-local-root.crt
```

Do not commit that copied root CA. The example `.gitignore` excludes `*.crt`,
so each operator supplies the CA that matches their own host tunnel.

## Usage

1. Copy your CA certificate(s) into this directory. PEM-encoded, with a
   `.crt` extension — `update-ca-certificates` only picks up `*.crt`.

   ```bash
   cp /path/to/corp-proxy-ca.pem ./corp-proxy-ca.crt
   ```

2. Rebuild the Hermes sandbox by re-running bring-up:

   ```bash
   bash scripts/bring-up.sh
   ```

The Dockerfile copies everything in this directory into
`/usr/local/share/ca-certificates/` and runs `update-ca-certificates` to
register the new trust roots.
