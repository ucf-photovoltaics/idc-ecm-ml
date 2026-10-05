default:
    podman run --rm -v .:/work -w /work $(podman build -q .) make all
