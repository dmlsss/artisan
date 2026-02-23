#!/bin/bash
# =============================================================================
# Artisan Roaster USB Permission Fix
# =============================================================================
# This script permanently fixes USB permissions for coffee roaster devices
# Run this script, then log out/in or reboot for changes to take effect
# =============================================================================

set -e

echo "=========================================="
echo "  Artisan Roaster USB Permission Fix"
echo "=========================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script needs sudo privileges. Rerunning with sudo..."
    exec sudo bash "$0" "$@"
fi

ORIGINAL_USER="${SUDO_USER:-$USER}"

# 1. Create udev rules directory
echo "[1/5] Creating udev rules directory..."
mkdir -p /etc/udev/rules.d
echo "✓ Directory created"

# 2. Install udev rules from Artisan source or create new ones
echo ""
echo "[2/5] Installing udev rules..."

if [ -f "/home/user/artisan/src/debian/etc/udev/rules.d/99-artisan.rules" ]; then
    # Copy from Artisan source
    cp /home/user/artisan/src/debian/etc/udev/rules.d/99-artisan.rules /etc/udev/rules.d/
    echo "✓ Copied udev rules from Artisan source"
else
    # Create udev rules manually
    cat > /etc/udev/rules.d/99-artisan.rules << 'EOF'
# udev rule to allow write access for Aillio USB devices
SUBSYSTEMS=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5741", MODE="0660", GROUP="dialout"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="a27e", MODE="0660", GROUP="dialout"

# udev rule to allow write access for Yoctopuce USB devices
SUBSYSTEM=="usb", ATTR{idVendor}=="24e0", MODE="0666", GROUP="dialout"

# udev rule for all current and future Phidgets - Vendor = 0x06c2, Product = 0x0030 - 0x00af
SUBSYSTEMS=="usb", ACTION=="add", ATTRS{idVendor}=="06c2", ATTRS{idProduct}=="00[3-a][0-f]", MODE="666"

# Generic serial port access
KERNEL=="ttyUSB[0-9]*", MODE="0666", GROUP="dialout"
KERNEL=="ttyACM[0-9]*", MODE="0666", GROUP="dialout"
EOF
    echo "✓ Created udev rules"
fi

# 3. Add user to necessary groups
echo ""
echo "[3/5] Adding user '$ORIGINAL_USER' to dialout and plugdev groups..."
usermod -a -G dialout "$ORIGINAL_USER" 2>/dev/null || true
usermod -a -G plugdev "$ORIGINAL_USER" 2>/dev/null || true
echo "✓ User added to groups"

# 4. Reload udev rules
echo ""
echo "[4/5] Reloading udev rules..."
if command -v udevadm &> /dev/null; then
    udevadm control --reload-rules
    udevadm trigger
    echo "✓ udev rules reloaded"
else
    echo "⚠️  udevadm not found - rules will apply after reboot"
fi

# 5. Show current USB devices
echo ""
echo "[5/5] Current USB serial devices:"
ls -la /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  (none found - plug in your roaster)"

# Summary
echo ""
echo "=========================================="
echo "  ✓ Installation Complete!"
echo "=========================================="
echo ""
echo "📋 What was done:"
echo "  • Installed udev rules to /etc/udev/rules.d/99-artisan.rules"
echo "  • Added user '$ORIGINAL_USER' to dialout and plugdev groups"
echo "  • Reloaded udev rules"
echo ""
echo "⚠️  IMPORTANT: You MUST do ONE of these:"
echo "  1. Log out and log back in, OR"
echo "  2. Reboot your computer"
echo ""
echo "  This is required for group membership to take effect."
echo ""
echo "🔍 After logging back in, verify with:"
echo "  groups"
echo ""
echo "  You should see 'dialout' and 'plugdev' in the list."
echo ""

# Offer to apply permissions immediately to any connected devices
if ls /dev/ttyUSB* /dev/ttyACM* &>/dev/null; then
    echo "🚀 Quick fix available!"
    echo ""
    echo "   I found USB serial devices. Apply temporary permissions now?"
    echo "   (This lets you test immediately without logging out)"
    echo ""
    read -p "   Apply now? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        chmod 666 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
        echo "   ✓ Temporary permissions applied!"
        echo "   You can test your roaster connection now."
        echo ""
    fi
fi

echo "Done! 🎉"
