install_ipfs() {
    unamem=`uname -m`

    installer_url="https://dist.ipfs.tech/kubo/v0.19.0/"
    if [[ ${OSTYPE} == "msys" ]]; then
            installer_filename=kubo_v0.19.0_windows-amd64.zip
    else
        if [[ $unamem -eq "x86_64" ]]; then
            installer_filename=kubo_v0.19.0_darwin-amd64.tar.gz

        elif [[ $unamem -eq "arm64" ]]; then
            installer_filename=kubo_v0.19.0_darwin-arm64.tar.gz
            echo '[*] Installing rosetta...'
            echo 'A' | softwareupdate --install-rosetta &>/dev/null
        fi
    fi

    which ipfs &>/dev/null
    if [[ $? != "0" ]]; then
        echo "[!] ipfs command line has not detected"
        echo -e "It will be installed via kudo\n"

        curl "${installer_url}${installer_filename}"\
            -o "${TEMP_DIR}${installer_filename}"

        cd $TEMP_DIR
        tar -xvzf "${installer_filename}"

        if [[ ${OSTYPE} == "msys" ]]; then
            mv kubo/ipfs.exe /usr/bin/
            export GO_IPFS_LOCATION=/usr/bin/
        else
            sudo bash "kubo/install.sh"
        fi
        cd ..
    fi

}
TEMP_DIR="__FREEWILLINSTALATIONDIR"
mkdir ${TEMP_DIR} 2>/dev/null
install_python_requirements
install_ipfs
install_foundry
rm -rf $TEMP_DIR 2>/dev/null
