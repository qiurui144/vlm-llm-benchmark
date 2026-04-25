#!/usr/bin/env bash
# ZeroTier offline install + auto join (Optional)
#
# Set ZEROTIER_NETWORK_ID env var before running.
#
# Usage:
#   ZEROTIER_NETWORK_ID=<your-16-hex-network-id> sudo -E bash scripts/setup_zerotier.sh
#
# After joining you must approve the node at https://my.zerotier.com/network/<NETWORK_ID>

set -euo pipefail

NETWORK_ID="${ZEROTIER_NETWORK_ID:-}"
if [[ -z "$NETWORK_ID" ]]; then
    echo "[ERROR] ZEROTIER_NETWORK_ID env var is required (16-hex network id from https://my.zerotier.com)"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPS_DIR="$PKG_ROOT/deps"

if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] 必须 sudo/root 执行（需要安装 deb + 启动 systemd 服务）"
    exit 1
fi

# 已经装过就跳过
if command -v zerotier-cli &>/dev/null; then
    echo "[INFO] zerotier-cli 已存在，跳过安装"
else
    # 侦测 Ubuntu codename
    CODENAME=""
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        CODENAME="${VERSION_CODENAME:-}"
    fi

    DEB=""
    case "$CODENAME" in
        jammy|focal|bionic)
            DEB="$DEPS_DIR/zerotier-one_1.14.2_amd64_jammy.deb" ;;
        noble|mantic)
            DEB="$DEPS_DIR/zerotier-one_1.14.2_amd64_noble.deb" ;;
        *)
            # DGX OS 基于 Ubuntu 22.04, 默认用 jammy
            echo "[WARN] 未识别 codename='$CODENAME'，按 jammy 处理"
            DEB="$DEPS_DIR/zerotier-one_1.14.2_amd64_jammy.deb" ;;
    esac

    if [[ ! -f "$DEB" ]]; then
        echo "[ERROR] 找不到 $DEB"
        exit 2
    fi

    echo "[INFO] 安装 $DEB"
    dpkg -i "$DEB" || apt-get install -f -y
fi

systemctl enable --now zerotier-one

# 等守护进程起来
for i in {1..10}; do
    if zerotier-cli info &>/dev/null; then
        break
    fi
    sleep 1
done

echo "[INFO] 加入网络 $NETWORK_ID"
zerotier-cli join "$NETWORK_ID"

echo
echo "===== ZeroTier 节点信息 ====="
zerotier-cli info
echo
echo "===== 网络状态 ====="
zerotier-cli listnetworks
echo
echo "⚠  若 status=REQUESTING_CONFIGURATION 或 ACCESS_DENIED:"
echo "    请到 https://my.zerotier.com/network/$NETWORK_ID 审批该节点"
