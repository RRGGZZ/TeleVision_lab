#!/bin/bash
# Quick start script for TeleVision Isaac Lab

set -e

echo "================================"
echo "TeleVision Isaac Lab Quick Start"
echo "================================"
echo ""

# 1. Verify environment
echo "[1] 验证 television_lab 环境..."
if ! conda run -n television_lab python --version 2>/dev/null; then
    echo "[!] television_lab 环境不存在"
    exit 1
fi

version=$(conda run -n television_lab python --version 2>&1 | awk '{print $2}')
echo "   Python 版本: $version"

# 2. Quick import test
echo ""
echo "[2] Testing imports..."
conda run -n television_lab python -c "
import sys
checks = []

try:
    from teleop.teleop_hand import VuerTeleop, ActionMapper
    checks.append(('teleop_hand', True))
except Exception as e:
    checks.append(('teleop_hand', False))

try:
    from dex_retargeting.retargeting_config import RetargetingConfig
    checks.append(('dex-retargeting', True))
except Exception as e:
    checks.append(('dex-retargeting', False))

try:
    from tv_isaaclab import IsaacLabEnvBridge
    checks.append(('Isaac Lab bridge', True))
except Exception as e:
    checks.append(('Isaac Lab bridge', False))

try:
    from vuer import Vuer
    checks.append(('VisionPro (vuer)', True))
except Exception as e:
    checks.append(('VisionPro (vuer)', False))

print()
for name, ok in checks:
    status = '✓' if ok else '✗'
    print(f'   {status} {name}')

all_ok = all(ok for _, ok in checks)
sys.exit(0 if all_ok else 1)
" || {
    echo "[!] Import test failed"
    exit 1
}

# 3. Run integration tests
echo ""
echo "[3] 运行集成测试..."
cd /home/r/Downloads/TeleVision\ _lab
conda run -n television_lab python scripts/test_integration.py 2>&1 | head -30

# 4. Summary
echo ""
echo "================================"
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "📝 下一步:"
echo ""
echo "1. 验证所有测试通过"
echo "2. 连接 VisionPro 和本机网络"
echo "3. 启动遥操作："
echo "   cd teleop && python teleop_hand.py --task television_lab"
echo ""
echo "📚 更多信息:"
echo "   - 查看 SETUP_PYTHON311.md 了解环境配置"
echo "   - 查看 QUICKSTART.md 了解快速使用"
echo "   - 查看 TESTING.md 了解测试详情"
echo ""
