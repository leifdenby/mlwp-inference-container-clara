#!/bin/bash
# Run an ANNA forecast in the inference image (built with build_image.sh).
#
# Usage:
#   scripts/run_inference_container.sh [<ANALYSIS_TIME> [<FORECAST_DURATION>]] [<option>...]
#
#   ANALYSIS_TIME      DINI analysis time, ISO8601 (e.g. 2026-10-01T00:00Z),
#                      or ANNA_ANALYSIS_TIME
#   FORECAST_DURATION  ISO8601 duration, PT12H .. PT30H (default PT18H)
#   <option>...        further options of the entrypoint (see README.md, or
#                      `python -m anna_inference --help`)
#
# The forecast is configured as outside the container: every set ANNA_*
# variable (README.md) is passed into the container. Except:
#   ANNA_WORKDIR       the host directory for the inputs and outputs (default
#                      ./inference_workdir), mounted as the container's
#                      workdir; the forecast is written to outputs/ in it
#   ANNA_IFS_BOUNDARY_PATH  a local IFS zarr is mounted read-only at the same
#                      path
# Settings of this script:
#   CONTAINER_APP      podman (default) or docker
#   IMAGE_NAME         image (default mlwp-inference-container-anna:latest)
#
# AWS credentials for reading DINI (and IFS) from S3 and for the upload are
# passed at run time, never baked into the image: AWS_ACCESS_KEY_ID/
# AWS_SECRET_ACCESS_KEY (/AWS_SESSION_TOKEN) if set, and ~/.aws (read-only)
# with AWS_PROFILE, or the per-side ANNA_{INPUT,OUTPUT}_AWS_PROFILE and
# ANNA_{INPUT,OUTPUT}_S3_ENDPOINT_URL (see README.md).

set -euo pipefail

CLI_ARGS=()
if [ "$#" -ge 1 ] && [[ "$1" != -* ]] ; then
    CLI_ARGS+=(--analysis-time "$1")
    shift
    if [ "$#" -ge 1 ] && [[ "$1" != -* ]] ; then
        CLI_ARGS+=(--forecast-duration "$1")
        shift
    fi
fi
CLI_ARGS+=("$@")

CONTAINER_APP=${CONTAINER_APP:-podman}
IMAGE_NAME=${IMAGE_NAME:-mlwp-inference-container-anna:latest}
HOST_WORKDIR=${ANNA_WORKDIR:-./inference_workdir}
mkdir -p "${HOST_WORKDIR}"
HOST_WORKDIR="$(cd "${HOST_WORKDIR}" && pwd)"

ARGS=(--rm --shm-size=32g)
if [ "${CONTAINER_APP}" = "docker" ] ; then
    # run as the calling user, so the outputs in the workdir are theirs
    # (rootless podman maps the container's root to the user already).
    # The uid has no passwd entry in the image: USER/LOGNAME are what
    # getpass.getuser() (e.g. torch's compile cache) reads first
    CONTAINER_HOME=/tmp
    ARGS+=(--gpus all --user "$(id -u):$(id -g)" -e HOME="${CONTAINER_HOME}")
    ARGS+=(-e USER="$(id -un)" -e LOGNAME="$(id -un)")
else
    CONTAINER_HOME=/root
    ARGS+=(--device nvidia.com/gpu=all)
fi
if [ -n "${ANNA_DEBUGGER:-}" ] ; then
    ARGS+=(-it)
fi

# the workdir is the container's default, /workspace/inference_workdir
ARGS+=(-v "${HOST_WORKDIR}:/workspace/inference_workdir:Z")

# the ANNA_* variables (the forecast's configuration)
while IFS= read -r var ; do
    case "${var}" in
        ANNA_WORKDIR|ANNA_IFS_BOUNDARY_PATH) ;;
        ANNA_*) ARGS+=(-e "${var}=${!var}") ;;
    esac
done < <(compgen -e)

if [ -n "${ANNA_IFS_BOUNDARY_PATH:-}" ] ; then
    IFS_PATH="${ANNA_IFS_BOUNDARY_PATH}"
    if [ -e "${IFS_PATH}" ] ; then
        # a local zarr: mount it at the same (absolute) path
        IFS_PATH="$(cd "${IFS_PATH}" && pwd)"
        ARGS+=(-v "${IFS_PATH}:${IFS_PATH}:ro,Z")
    fi
    ARGS+=(-e ANNA_IFS_BOUNDARY_PATH="${IFS_PATH}")
fi

# AWS credentials, at run time only
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
        AWS_PROFILE AWS_REGION AWS_DEFAULT_REGION ; do
    if [ -n "${!var:-}" ] ; then
        ARGS+=(-e "${var}=${!var}")
    fi
done
if [ -d "${HOME}/.aws" ] ; then
    ARGS+=(-v "${HOME}/.aws:${CONTAINER_HOME}/.aws:ro,Z")
fi

echo "Running ${IMAGE_NAME} ${CLI_ARGS[*]-}, workdir ${HOST_WORKDIR}"
${CONTAINER_APP} run "${ARGS[@]}" "${IMAGE_NAME}" ${CLI_ARGS[@]+"${CLI_ARGS[@]}"}
