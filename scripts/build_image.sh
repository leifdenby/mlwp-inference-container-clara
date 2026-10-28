#!/bin/bash

# Build the CLARA inference image from the repository root (the build
# context), wherever this script is called from
cd "$(dirname "$0")/.." || exit 1

# Container application (default podman)
CONTAINER_APP=${CONTAINER_APP:-podman}

# Configuration
BUILD_LOG_LEVEL=DEBUG
IMAGE_NAME=${IMAGE_NAME:-"mlwp-inference-container-clara:latest"}

# Set PULL_PROXY before running this script, e.g.:
#   export PULL_PROXY="your.proxy.server:port"
if [ -z "$PULL_PROXY" ]; then
	echo "Info: PULL_PROXY is not set. Using public DockerHub."
	CR_URL="docker.io"
else
	echo "Info: Using proxy $PULL_PROXY and internal DockerHub."
	CR_URL="dockerhub.dmi.dk"
fi

# if we're on ARM architecture, use the ARM base image
if [ "$(uname -m)" = "aarch64" ]; then
	echo "Info: Detected ARM architecture. Using ARM base image."
	# dockerhub doesn't have an official pytorch image for ARM, so we use NVIDIA's NGC registry
	if [ "$CR_URL" = "docker.io" ]; then
		CR_URL="nvcr.io"
	fi
	BASE_IMAGE="${CR_URL}/nvidia/pytorch:26.01-py3"
else
	echo "Info: Using x86_64 base image."
	BASE_IMAGE="${CR_URL}/pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
fi

# A GitHub token that can read the private repositories leifdenby/real-prob-lam
# (neural-lam) and leifdenby/mlwp-inference-container-clara (release assets:
# the inference package and checkpoint). Passed to the build as a secret,
# never as a build arg, so it doesn't end up in the image
if [ -z "$GITHUB_TOKEN" ]; then
	GITHUB_TOKEN=$(gh auth token 2>/dev/null)
fi
if [ -z "$GITHUB_TOKEN" ]; then
	echo "Error: set GITHUB_TOKEN (or log in with \`gh auth login\`): the build"
	echo "installs neural-lam from the private leifdenby/real-prob-lam."
	exit 1
fi
SECRET_FILE=$(mktemp)
trap 'rm -f "$SECRET_FILE"' EXIT
chmod 600 "$SECRET_FILE"
printf '%s' "$GITHUB_TOKEN" > "$SECRET_FILE"

# Where the inference artifact comes from (see Containerfile):
# - "local" (default) uses an artifact directory LOCAL_ARTIFACT_DIR assembled
#   with clara_inference.artifact.assemble (with or without the checkpoint)
# - "release" downloads the artifact package zip (PACKAGE_URL, e.g. on Zenodo,
#   checked against PACKAGE_MD5; the token is only sent to api.github.com)
# The model checkpoint is downloaded during the build unless present (url
# and md5 in the artifact's configs/model.yaml).
ARTIFACT_SOURCE=${ARTIFACT_SOURCE:-local}

if [ "$ARTIFACT_SOURCE" = "local" ]; then
	# relative to the repository root (the build context)
	LOCAL_ARTIFACT_DIR=${LOCAL_ARTIFACT_DIR:-artifact/build/clara-local}
	if [ ! -f "${LOCAL_ARTIFACT_DIR}/artifact.yaml" ]; then
		echo "Error: ARTIFACT_SOURCE=local but ${LOCAL_ARTIFACT_DIR} hasn't been assembled."
		echo "Create it with clara_inference.artifact.assemble (see README.md)."
		exit 1
	fi
	if grep -q "placeholder_stats: true" "${LOCAL_ARTIFACT_DIR}/artifact.yaml"; then
		echo "Warning: ${LOCAL_ARTIFACT_DIR} uses PLACEHOLDER statistics (testing only)."
	fi
elif [ "$ARTIFACT_SOURCE" = "release" ]; then
	if [ -z "$PACKAGE_URL" ] || [ -z "$PACKAGE_MD5" ]; then
		echo "Error: ARTIFACT_SOURCE=release needs PACKAGE_URL and PACKAGE_MD5."
		exit 1
	fi
	# the Containerfile always copies LOCAL_ARTIFACT_DIR: an empty directory,
	# so no local package (or checkpoint) ends up in an image layer
	LOCAL_ARTIFACT_DIR=artifact/build/.empty
	mkdir -p "${LOCAL_ARTIFACT_DIR}"
else
	echo "Error: unknown ARTIFACT_SOURCE=$ARTIFACT_SOURCE (local or release)"
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
	--secret id=github_token,src="$SECRET_FILE" \
	--build-arg BASE_IMAGE="$BASE_IMAGE" \
	--build-arg IMAGE_VERSION="$IMAGE_VERSION" \
	--build-arg ARTIFACT_SOURCE="$ARTIFACT_SOURCE" \
	--build-arg PACKAGE_URL="$PACKAGE_URL" \
	--build-arg PACKAGE_MD5="$PACKAGE_MD5" \
	--build-arg LOCAL_ARTIFACT_DIR="$LOCAL_ARTIFACT_DIR" \
	-t "$IMAGE_NAME" \
	-f Containerfile \
	.
