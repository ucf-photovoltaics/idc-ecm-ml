default:
    podman run --rm -e IDC_SCALE -v "$PWD":/work -w /work $(podman build -q .) make all

check:
    podman run --rm -e IDC_SCALE -v "$PWD":/work -w /work $(podman build -q .) make check
