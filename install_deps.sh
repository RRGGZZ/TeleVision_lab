#!/bin/bash
# Installation script for TeleVision Isaac Lab environment
# Handles Python 3.11+ compatible dependencies

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

# Verify we're in television_lab environment
if ! python -c "import sys; sys.exit(0 if 'television_lab' in sys.prefix else 1)" 2>/dev/null; then
    echo "[!] Warning: Not in television_lab conda environment"
    echo "    Consider running: conda activate television_lab"
fi

echo ""
echo "[1/3] Installing main dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "[2/3] Installing dex-retargeting from GitHub (Python 3.11+ compatible)..."
# dex-retargeting 0.5.0 supports Python 3.11+ but not 3.13+
# Install from GitHub main branch for best compatibility
if pip install git+https://github.com/dexsuite/dex-retargeting.git@main; then
    echo "[✓] dex-retargeting installed successfully"
else
    echo "[!] dex-retargeting installation failed"
    echo "    This is needed for hand retargeting/IK"
    echo "    Try manually: pip install git+https://github.com/dexsuite/dex-retargeting.git@main"
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
