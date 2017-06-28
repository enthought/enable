#!/bin/bash

set -e

install_edm() {
    local EDM_MAJOR_MINOR="$(echo "$INSTALL_EDM_VERSION" | sed -E -e 's/([[:digit:]]+\.[[:digit:]]+)\..*/\1/')"
    local EDM_PACKAGE="edm_${INSTALL_EDM_VERSION}_linux_x86_64.sh"
    local EDM_INSTALLER_DIR="${HOME}/.cache/download"
    local EDM_INSTALLER_PATH="${EDM_INSTALLER_DIR}/${EDM_PACKAGE}"
    local DOWNLOAD_URL="https://package-data.enthought.com/edm/rh5_x86_64/${EDM_MAJOR_MINOR}/${EDM_PACKAGE}"

    if [ ! -e "$EDM_INSTALLER_PATH" ]; then
        mkdir -p ${EDM_INSTALLER_DIR}
        curl -o "$EDM_INSTALLER_PATH" -L "$DOWNLOAD_URL"
        if [ $? -ne 0 ]; then
            echo "Failed to download $DOWNLOAD_URL"
            exit 1
        fi
    fi

    bash "$EDM_INSTALLER_PATH" -b -p "${HOME}/edm"
}

if [ -z $INSTALL_EDM_VERSION ]; then
    echo "The desired EDM version must be set in the INSTALL_EDM_VERSION environment variable before running this script!"
    exit 1
fi

install_edm
