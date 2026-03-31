# Python 3.11 环境配置指南

## ✅ 安装完成状态

所有必需的依赖已成功安装到 `television_lab` conda 环境中。

```
✓ Python 3.11.15
✓ VisionPro streaming (vuer)
✓ Hand tracking & IK (dex-retargeting 0.5.0 from GitHub)
✓ Isaac Lab integration
✓ Episode recording (HDF5)
✓ All core teleoperation components
```

## 📦 依赖更新详情

### Python 3.11 兼容性

| 包 | 版本 | 说明 |
|----|------|------|
| `dex-retargeting` | 0.5.0 (GitHub) | 从源代码构建，支持 Python 3.11+ |
| `numpy` | 1.26.4 | Isaac Lab 要求，避免 numpy 2.0+ |
| `vuer` | 0.1.6 | VisionPro 实时流 |
| `torch` | 2.3.0+ | 深度学习推理 |
| `params_proto` | 3.3.0 | 配置管理 |

### 已解决的问题

1. **dex-retargeting**: 0.4.6 不支持 Python 3.11+
   - 解决: 从 GitHub main 分支构建 0.5.0
   - API 兼容: ✅ 无需修改代码

2. **params_proto**: WebRTC 模块版本不兼容
   - 解决: 使用 try-except 处理导入
   - 影响: WebRTC 模式不可用（保留 image 模式）
   - 用 途: teleop_hand.py 使用 image 模式，无影响

3. **numpy**: isaaclab 需要 numpy < 2.0
   - 解决: requirements.txt 中限制 `numpy >= 1.24.0,<2.0`

## 🚀 验证安装

### 快速检查
```bash
conda activate television_lab
python scripts/test_integration.py
```

### 导入测试
```bash
python -c "from teleop.teleop_hand import VuerTeleop, ActionMapper; print('✓ All ready')"
```

### 查看已安装包
```bash
conda run -n television_lab pip list | grep -E "(dex|vuer|torch|numpy)"
```

## 🛠️ 安装说明

### 方法 1: 自动安装脚本（推荐）
```bash
cd /path/to/TeleVision_lab
bash install_deps.sh
```

### 方法 2: 手动管理安装

1. **基础依赖**
```bash
conda activate television_lab
pip install -r requirements.txt
```

2. **dex-retargeting (从 GitHub)**
```bash
pip install git+https://github.com/dexsuite/dex-retargeting.git@main
```

3. **DETR 模型**
```bash
cd act/detr && pip install -e .
```

## 📝 文件更改

### 修改的文件
- `requirements.txt` - 移除过期的 dex-retargeting，添加注释
- `teleop/teleop_hand.py` - 改进导入路径处理
- `teleop/TeleVision.py` - 添加 WebRTC 导入保护

### 新增文件
- `install_deps.sh` - 完整安装脚本

## ⚠️ 已知限制

| 功能 | 状态 | 说明 |
|------|------|------|
| 手部追踪 (VisionPro) | ✅ | dex-retargeting 0.5.0 正常工作 |
| Isaac Lab 环境 | ✅ | 通过 tv_isaaclab 集成 |
| 图像流 | ✅ | vuer 集成完成 |
| HDF5 记录 | ✅ | 用于训练数据 |
| WebRTC 流 | ⚠️ | 不可用（params_proto 版本冲突）|
|   | | 但 image 模式完全可用 |

## 📚 使用示例

### 运行集成测试
```bash
conda activate television_lab
python scripts/test_integration.py
# Expected: 🎉 All tests passed!
```

### 启动遥操作（需 VisionPro）
```bash
cd teleop
python teleop_hand.py \
  --task television_lab \
  --record \
  --output ../data/recordings/isaaclab/episode_0.hdf5 \
  --max_steps 500
```

### 批量收集数据
```bash
cd scripts
python collect_episodes.py \
  --num_episodes 10 \
  --task television_lab \
  --output_dir ../data/recordings/isaaclab
```

### 回放已记录的视频
```bash
cd scripts
python replay_demo.py \
  --task television_lab \
  --episode_path ../data/recordings/isaaclab/episode_0.hdf5
```

## 🔍 诊断

### 检查 dex-retargeting 版本
```bash
python -c "import dex_retargeting; print(dex_retargeting.__version__)"
```

### 列出 tv_isaaclab 中的容错信息
```bash
python -c "
from dex_retargeting.retargeting_config import RetargetingConfig
print('✓ dex_retargeting 可用')
from tv_isaaclab import IsaacLabEnvBridge
print('✓ Isaac Lab 集成可用')
"
```

### 检查 WebRTC 状态
```bash
python teleop/TeleVision.py 2>&1 | grep -E "(WebRTC|image streaming)"
```

## 📞 常见问题

**Q: 为什么 WebRTC 不可用？**
- A: `params_proto` 版本与 `zed_server` 不兼容。但 teleop_hand.py 使用 image 模式，无需 WebRTC。

**Q: dex-retargeting 从 GitHub 安装需要多久？**
- A: 第一次构建 5-10 分钟（编译 Pinocchio）。之后缓存使用会快得多。

**Q: numpy 为什么限制在 < 2.0？**
- A: Isaac Lab 0.54.3 有向后不兼容的更改。等待新版本后可升级。

**Q: 如何切换到最新的 dex-retargeting PyPI 版本？**
- A: 等待官方发布支持 Python 3.11 的版本到 PyPI。目前从 GitHub 构建是唯一方式。

## ✅ 检查清单

- [ ] Python 3.11 版本验证：`python --version`
- [ ] dex-retargeting 导入：`python -c "from dex_retargeting import *"`
- [ ] 集成测试通过：`python scripts/test_integration.py`
- [ ] teleop_hand 导入成功：`python -c "from teleop.teleop_hand import *"`
- [ ] VisionPro 摄像头连接（需实际硬件）

## 🎯 下一步

1. ✅ 环境配置完成
2. 👉 验证 VisionPro 网络连接
3. 收集遥操作数据
4. 训练 ACT 模型
5. 部署策略

---

**最后更新**: 2026-03-30  
**Python 版本**: 3.11.15  
**dex-retargeting**: 0.5.0 (GitHub)
