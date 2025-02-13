IMG=nvcr.io/nvidia/isaac-sim:4.2.0-tkq
xhost +
sudo docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=:1" \
    -e "MESA_GL_VERSION_OVERRIDE=4.6" \
    -v $PWD/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v $PWD/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v $PWD/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v $PWD/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v $PWD/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v $PWD/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v $PWD/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v $PWD/docker/isaac-sim/documents:/root/Documents:rw \
    -v ~/:/mnt \
    -v $PWD:/app \
    -w /isaac-sim \
    $IMG
#nvcr.io/nvidia/isaac-sim:4.2.0
