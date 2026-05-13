#!/bin/bash
# Installation script for TeleVision Isaac Lab environment
# Handles Python 3.11 / 3.12 compatible dependencies

set -e

echo "================================"
echo "TeleVision Isaac Lab Setup"
echo "================================"

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "[*] Python version: $python_version"

# Extract major.minor version
major_minor=$(echo $python_version | cut -d. -f1-2)
echo "[*] Python $major_minor detected"

if [ "$major_minor" = "3.13" ]; then
    echo "[!] Python 3.13 is not supported for teleoperation in this repo."
    echo "    dex-retargeting currently supports Python < 3.13."
    echo "    Please create a Python 3.11 or 3.12 environment first."
    exit 1
fi

# Verify we're in television_lab environment
if ! python -c "import sys; sys.exit(0 if 'television_lab' in sys.prefix else 1)" 2>/dev/null; then
    echo "[!] Warning: Not in television_lab conda environment"
    echo "    Consider running: conda activate television_lab"
fi

echo ""
echo "[1/3] Installing main dependencies..."
pip install --upgrade pip setuptools
pip install "packaging==23.0" "wheel<0.47"
pip install -r requirements.txt

echo ""
echo "[2/3] Installing dex-retargeting NumPy 1.x compatible release..."
# dex-retargeting 0.5.x requires NumPy 2.x, which conflicts with Isaac Lab
# environments that pin NumPy 1.x. Keep this on the 0.4.x line.
if pip install "dex-retargeting>=0.4.5,<0.5.0"; then
    echo "[✓] dex-retargeting installed successfully"
else
    echo "[!] dex-retargeting installation failed"
    echo "    This is needed for hand retargeting/IK"
    echo "    Try manually: pip install 'dex-retargeting>=0.4.5,<0.5.0'"
fi

echo ""
echo "[3/3] Installing DETR model package..."
cd act/detr && pip install -e . && cd ../..

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Quick test:"
echo "  python scripts/test_integration.py"
echo ""
echo "Run teleoperation:"
echo "  cd teleop && python teleop_hand.py --task television_lab"
echo ""
