#!/bin/bash

# Build the ANNA inference image from the repository root (the build context),
# wherever this script is called from
cd "$(dirname "$0")/.." || exit 1

# Container application (defuault to podman if not set)
CONTAINER_APP=${CONTAINER_APP:-podman}

# Configuration
BUILD_LOG_LEVEL=DEBUG
IMAGE_NAME=${IMAGE_NAME:-"mlwp-inference-container-clara:latest"}

HTTP_PROXY=""
HTTPS_PROXY=""

# Set PULL_PROXY before running this script, e.g.:
#   export PULL_PROXY="your.proxy.server:port"
if [ -z "$PULL_PROXY" ]; then
	echo "Info: PULL_PROXY is not set. Using public DockerHub."
	PULL_PROXY=""
    CR_URL="docker.io"
else
	echo "Info: Using proxy $PULL_PROXY and internal DockerHub."
    CR_URL="dockerhub.dmi.dk"
fi

# if we're on ARM architecture, use the ARM base image
if [ "$(uname -m)" = "aarch64" ]; then
	echo "Info: Detected ARM architecture. Using ARM base image."
	# dockerhub doesn't have an official pytorch image for ARM, so we use NVIDIA's NGC registry
	if [ -z "$CR_URL" ] || [ "$CR_URL" = "docker.io" ]; then
		CR_URL="nvcr.io"
	fi
	BASE_IMAGE="${CR_URL}/nvidia/pytorch:26.01-py3"
else
	echo "Info: Using x86_64 base image."
	BASE_IMAGE="${CR_URL}/pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
fi

# Where the inference artifact comes from (see Containerfile):
# - "zenodo" (default) downloads the published inference package
#   (PACKAGE_URL, checked against PACKAGE_MD5);
# - "local" uses a package directory LOCAL_ARTIFACT_DIR assembled with
#   clara_inference.artifact.assemble (or collect).
# The model checkpoint is downloaded from Zenodo during the build in both
# cases (unless present).
ARTIFACT_SOURCE=${ARTIFACT_SOURCE:-zenodo}
# the ANNA inference package on Zenodo (record 23293148, DOI
# 10.5281/zenodo.23293148; built as artifact/build/anna-danra-2026-10-11.zip)
PACKAGE_URL=${PACKAGE_URL:-https://zenodo.org/records/23293148/files/anna-danra-2026-10-11.zip?download=1}
PACKAGE_MD5=${PACKAGE_MD5:-d453733a56c381d63922806cd4b25472}
# copies of the package and the checkpoint (same md5s) in a dmidk GitHub
# release, used if Zenodo is unavailable
MIRROR=https://github.com/dmidk/mlwp-inference-container-clara/releases/download/artifact-anna-danra-2026-10-11
PACKAGE_MIRROR_URL=${PACKAGE_MIRROR_URL:-$MIRROR/anna-danra-2026-10-11.zip}
CHECKPOINT_MIRROR_URL=${CHECKPOINT_MIRROR_URL:-$MIRROR/danra_model.ckpt}

if [ "$ARTIFACT_SOURCE" = "local" ]; then
	# relative to the repository root (the build context)
	LOCAL_ARTIFACT_DIR=${LOCAL_ARTIFACT_DIR:-artifact/build/anna-local}
	if [ ! -f "${LOCAL_ARTIFACT_DIR}/artifact.yaml" ]; then
		echo "Error: ARTIFACT_SOURCE=local but ${LOCAL_ARTIFACT_DIR} hasn't been assembled."
		echo "Create it with clara_inference.artifact.assemble (see"
		echo "artifact/README.md), or set ARTIFACT_SOURCE=zenodo."
		exit 1
	fi
	if grep -q "boundary_stats_placeholder: true" "${LOCAL_ARTIFACT_DIR}/artifact.yaml"; then
		echo "Warning: ${LOCAL_ARTIFACT_DIR} uses PLACEHOLDER boundary statistics."
	fi
elif [ "$ARTIFACT_SOURCE" = "zenodo" ]; then
	if [ -z "$PACKAGE_URL" ] || [ -z "$PACKAGE_MD5" ]; then
		echo "Error: ARTIFACT_SOURCE=zenodo needs PACKAGE_URL and PACKAGE_MD5."
		exit 1
	fi
	# a Zenodo draft's files can't be downloaded until it's published
	CURL_CHECK="curl -sfIL --connect-timeout 30 --retry 2 --retry-delay 10 --retry-all-errors -o /dev/null"
	if ! $CURL_CHECK "$PACKAGE_URL"; then
		if [ -n "$PACKAGE_MIRROR_URL" ] && $CURL_CHECK "$PACKAGE_MIRROR_URL"; then
			echo "Warning: can't download $PACKAGE_URL, the build will use the"
			echo "mirror $PACKAGE_MIRROR_URL."
		else
			echo "Error: can't download the inference package from $PACKAGE_URL"
			echo "or its mirror. Is the Zenodo record published? (Use"
			echo "ARTIFACT_SOURCE=local with a locally assembled package until then.)"
			exit 1
		fi
	fi
	# the Containerfile always copies LOCAL_ARTIFACT_DIR: an empty directory,
	# so no local package (or checkpoint) ends up in an image layer
	LOCAL_ARTIFACT_DIR=artifact/build/.empty
	mkdir -p "${LOCAL_ARTIFACT_DIR}"
else
	echo "Error: unknown ARTIFACT_SOURCE=$ARTIFACT_SOURCE (local or zenodo)"
	exit 1
fi

# The package version from the git tag (setuptools-scm); the build context
# has no .git, so it's passed as a build arg
if [ -z "$IMAGE_VERSION" ]; then
	IMAGE_VERSION=$(uvx --from setuptools-scm python -m setuptools_scm 2>/dev/null)
	IMAGE_VERSION=${IMAGE_VERSION:-0.0.0+unknown}
fi
echo "Info: clara-inference version $IMAGE_VERSION"

# Pull base image with proxy
HTTP_PROXY="$PULL_PROXY" HTTPS_PROXY="$PULL_PROXY" ${CONTAINER_APP} --log-level="$BUILD_LOG_LEVEL" pull "$BASE_IMAGE"

# Build image
echo "Running ${CONTAINER_APP} build to create image $IMAGE_NAME ..."
${CONTAINER_APP} build \
	--build-arg BASE_IMAGE="$BASE_IMAGE" \
	--build-arg IMAGE_VERSION="$IMAGE_VERSION" \
	--build-arg ARTIFACT_SOURCE="$ARTIFACT_SOURCE" \
	--build-arg PACKAGE_URL="$PACKAGE_URL" \
	--build-arg PACKAGE_MD5="$PACKAGE_MD5" \
	--build-arg PACKAGE_MIRROR_URL="$PACKAGE_MIRROR_URL" \
	--build-arg CHECKPOINT_MIRROR_URL="$CHECKPOINT_MIRROR_URL" \
	--build-arg LOCAL_ARTIFACT_DIR="$LOCAL_ARTIFACT_DIR" \
	-t "$IMAGE_NAME" \
	-f Containerfile \
	.
