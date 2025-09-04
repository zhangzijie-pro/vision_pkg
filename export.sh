#!/bin/bash
PKG_NAME="mipi_detect"
SRC_DIR="$(ros2 pkg prefix $PKG_NAME)/share/$PKG_NAME/config"
DEST_DIR="$HOME/${PKG_NAME}_config"

echo "拷贝 $SRC_DIR -> $DEST_DIR"
mkdir -p "$DEST_DIR"
cp -r $SRC_DIR/* "$DEST_DIR/"
echo "完成 ✅"