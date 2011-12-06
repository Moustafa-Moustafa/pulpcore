#!/bin/sh

NAME="pulp-server"
SELINUX_VARIANTS="mls strict targeted"
#This script will be called from pulp RPM and needs the ability to use the specified macro
if [ $# -lt 1 ]; then
    INSTALL_DIR=/usr/share
else
    INSTALL_DIR=$1
fi

if /usr/sbin/selinuxenabled ; then
    for selinuxvariant in ${SELINUX_VARIANTS}
    do
        /usr/sbin/semodule -s ${selinuxvariant} -i \
            ${INSTALL_DIR}/selinux/${selinuxvariant}/${NAME}.pp &> /dev/null || :
    done
fi

#bz 736788, allows repo sync through a proxy to work
/usr/sbin/setsebool -P httpd_can_network_connect
/usr/sbin/setsebool -P httpd_tmp_exec

