#!/usr/bin/env bash
# =============================================================================
# Artisan Ubuntu Setup Script
# =============================================================================
#
# One-stop script to set up a complete Artisan development and build
# environment on a fresh Ubuntu install (22.04+).
#
# Usage:
#   chmod +x setup-ubuntu.sh
#   ./setup-ubuntu.sh              # full setup: deps + venv + run-from-source
#   ./setup-ubuntu.sh --dev        # full setup + dev tools (linters, type checkers, tests)
#   ./setup-ubuntu.sh --build      # full setup + dev tools + build .deb/.rpm/.AppImage packages
#
# LICENSE
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License, or
# version 3 of the License, or (at your option) any later versison. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PYTHON_VERSION="3.14"           # Target Python version (deadsnakes PPA)
LIBUSB_VERSION="1.0.26"        # libusb version to build from source
DOTENV_GEM_VERSION="2.8.1"     # Ruby dotenv gem version (needed by fpm)
VENV_DIR="${HOME}/artisan-venv" # Virtual environment location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
MODE="run"  # default: just enough to run from source
for arg in "$@"; do
    case "$arg" in
        --dev)   MODE="dev" ;;
        --build) MODE="build" ;;
        --help|-h)
            echo "Usage: $0 [--dev|--build]"
            echo ""
            echo "  (no flag)  Set up to run Artisan from source"
            echo "  --dev      Also install dev tools (linters, type checkers, tests)"
            echo "  --build    Also install everything needed to build .deb/.rpm/.AppImage packages"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (use --help for usage)"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

info()  { echo -e "\n\033[1;34m==>\033[0m \033[1m$*\033[0m"; }
ok()    { echo -e "\033[1;32m  ✓\033[0m $*"; }
warn()  { echo -e "\033[1;33m  !\033[0m $*"; }

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------

info "Updating package lists"
sudo apt-get update -y -q

info "Installing core build dependencies"
sudo apt-get install -y -q \
    software-properties-common \
    build-essential \
    git \
    curl \
    wget \
    pkg-config \
    libudev-dev \
    libdbus-1-3 \
    libfuse2 \
    gdb

# Qt6 / XCB libraries required by PyQt6
info "Installing Qt6/XCB runtime libraries"
sudo apt-get install -y -q \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxcb-cursor0 \
    libxcb-shape0 \
    libegl1 \
    libopengl0 \
    libgl1

# gnome-keyring for the artisan.plus credential storage
info "Installing gnome-keyring (for artisan.plus password storage)"
sudo apt-get install -y -q gnome-keyring

ok "System packages installed"

# ---------------------------------------------------------------------------
# 2. Python (deadsnakes PPA)
# ---------------------------------------------------------------------------

info "Installing Python ${PYTHON_VERSION} from deadsnakes PPA"
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y -q
sudo apt-get install -y -q \
    "python${PYTHON_VERSION}" \
    "python${PYTHON_VERSION}-venv" \
    "python${PYTHON_VERSION}-dev" \
    "libpython${PYTHON_VERSION}"

ok "Python ${PYTHON_VERSION} installed: $(python${PYTHON_VERSION} --version)"

# ---------------------------------------------------------------------------
# 3. Virtual environment
# ---------------------------------------------------------------------------

info "Creating Python virtual environment in ${VENV_DIR}"
if [ -d "${VENV_DIR}" ]; then
    warn "Virtual environment already exists at ${VENV_DIR} — reusing it"
else
    "python${PYTHON_VERSION}" -m venv "${VENV_DIR}"
fi

# Activate it for the rest of this script
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

ok "Virtual environment active — $(python --version) at $(which python)"

# ---------------------------------------------------------------------------
# 4. Upgrade pip & install Python requirements
# ---------------------------------------------------------------------------

info "Upgrading pip"
pip install --upgrade pip

info "Installing Artisan Python dependencies"
pip install -r "${SCRIPT_DIR}/src/requirements.txt"

ok "Python requirements installed"

# ---------------------------------------------------------------------------
# 5. libusb (built from source, matching CI)
# ---------------------------------------------------------------------------

info "Building libusb ${LIBUSB_VERSION} from source"

# Remove the distro package to avoid conflicts
sudo apt-get remove -y libusb-1.0-0 2>/dev/null || true

LIBUSB_BUILD_DIR=$(mktemp -d)
pushd "${LIBUSB_BUILD_DIR}" > /dev/null

curl -kL -O "https://github.com/libusb/libusb/releases/download/v${LIBUSB_VERSION}/libusb-${LIBUSB_VERSION}.tar.bz2"
tar xjf "libusb-${LIBUSB_VERSION}.tar.bz2"
cd "libusb-${LIBUSB_VERSION}"
./configure --prefix=/usr
make -j"$(nproc)"
sudo make install

popd > /dev/null
rm -rf "${LIBUSB_BUILD_DIR}"

ok "libusb ${LIBUSB_VERSION} installed"

# ---------------------------------------------------------------------------
# 6. Dev tools (optional)
# ---------------------------------------------------------------------------

if [ "${MODE}" = "dev" ] || [ "${MODE}" = "build" ]; then
    info "Installing development tools (linters, type checkers, test framework)"
    pip install -r "${SCRIPT_DIR}/src/requirements-dev.txt"
    ok "Dev tools installed"
fi

# ---------------------------------------------------------------------------
# 7. Build toolchain (optional)
# ---------------------------------------------------------------------------

if [ "${MODE}" = "build" ]; then
    info "Installing build/packaging toolchain"

    # System packages for packaging
    sudo apt-get install -y -q \
        ruby-dev \
        p7zip-full \
        rpm \
        fakeroot \
        imagemagick

    # fpm (Effing Package Management) — used to create .deb and .rpm
    info "Installing Ruby gems (dotenv, fpm)"
    sudo gem install dotenv -v "${DOTENV_GEM_VERSION}"
    sudo gem install fpm

    ok "Build toolchain installed"

    echo ""
    info "To build Artisan packages, run:"
    echo "    source ${VENV_DIR}/bin/activate"
    echo "    cd ${SCRIPT_DIR}/src"
    echo "    ./build-linux.sh"
    echo "    ./build-linux-pkg.sh"
fi

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------

echo ""
echo "==========================================================================="
info "Setup complete!"
echo "==========================================================================="
echo ""
echo "  To run Artisan from source:"
echo ""
echo "    source ${VENV_DIR}/bin/activate"
echo "    cd ${SCRIPT_DIR}/src"
echo "    python artisan.py"
echo ""

if [ "${MODE}" = "dev" ] || [ "${MODE}" = "build" ]; then
    echo "  Dev commands (from src/ directory):"
    echo ""
    echo "    pytest                          # run tests"
    echo "    ruff check .                    # fast linting"
    echo "    pylint */*.py                   # full linting"
    echo "    mypy                            # type checking"
    echo "    codespell                       # spell checking"
    echo ""
fi

echo "  To re-activate the virtual environment later:"
echo ""
echo "    source ${VENV_DIR}/bin/activate"
echo ""
