#!/bin/bash
PKG_NAME="mipi_detect"
INSTALL_DIR="$(ros2 pkg prefix $PKG_NAME)/share/$PKG_NAME/config"
HOME_DIR="$HOME/${PKG_NAME}_config"

if [ ! -d "$HOME_DIR" ]; then
    echo "未找到 $HOME_DIR，先执行 export.sh"
    exit 1
fi

echo "同步 $HOME_DIR -> $INSTALL_DIR"
cp -r "$HOME_DIR"/* "$INSTALL_DIR/"
echo "完成 ✅"