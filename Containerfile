# syntax=docker/dockerfile:1
ARG BASE_IMAGE="pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN apt-get update && apt-get install -y curl unzip git

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# the package version: the build context has no .git, so setuptools-scm
# can't find the tag (build_image.sh and CI pass it). NB: build args are
# environment variables in RUN steps, so none may start with CLARA_ (the
# forecast's options, see the --help check below)
ARG IMAGE_VERSION="0.0.0+unknown"
LABEL org.opencontainers.image.version="${IMAGE_VERSION}"

COPY pyproject.toml README.md ./
COPY scripts/container_pins.py ./scripts/
COPY src/ ./src

# export current torch version to constraints file
RUN python -c 'import torch; print(f"torch=={torch.__version__}")' > constraints.txt
# the git pins of pyproject.toml ([tool.uv.sources] and
# override-dependencies) as requirement files: the install below uses
# --no-sources, so that torch is the base image's (not from the cu130 index
# in [tool.uv.sources])
RUN python scripts/container_pins.py pyproject.toml pins
# install with constraints. neural-lam comes from the private repository
# leifdenby/real-prob-lam: the GitHub token is a build secret (`--secret
# id=github_token`, see build_image.sh), only available during this step
# (through GIT_CONFIG_* variables of this step) and never written to an
# image layer
RUN --mount=type=secret,id=github_token,required=true \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0="url.https://x-access-token:$(cat /run/secrets/github_token)@github.com/.insteadOf" \
    GIT_CONFIG_VALUE_0="https://github.com/" \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CLARA_INFERENCE="${IMAGE_VERSION}" \
    uv pip install --break-system-packages --system --no-sources --no-cache \
        --constraints constraints.txt --overrides pins/overrides.txt \
        --requirements pins/requirements.txt .
# check that neural-lam imports (with the base image's torch), and show the
# versions
RUN python -c 'import importlib.metadata as md, clara_inference, neural_lam, torch; print("clara-inference", clara_inference.__version__, "neural-lam", md.version("neural-lam"), "torch", torch.__version__)'

ENV CARTOPY_DATA_DIR=/usr/local/share/cartopy
RUN python -c 'import cartopy, cartopy.io.shapereader as shp; \
    cartopy.config["data_dir"] = cartopy.config["pre_existing_data_dir"]; \
    [print(shp.natural_earth(scale, category, name)) \
     for category, name in [("physical", "coastline"), ("cultural", "admin_0_boundary_lines_land")] \
     for scale in ["110m", "50m", "10m"]]'

# The inference artifact (configs, statistics, grids, graph, model
# description) comes from:
# - "release" (default): the artifact package zip ${PACKAGE_URL}, a release
#   asset of this (private) repository, downloaded through the GitHub API
#   with the token build secret, checked against ${PACKAGE_MD5}
# - "local": the assembled artifact directory ${LOCAL_ARTIFACT_DIR} in the
#   build context (with or without the checkpoint)
# The model checkpoint is then downloaded (md5-checked) as given in
# inference_artifact/configs/model.yaml, unless it's already there.
ARG ARTIFACT_SOURCE="release"
ARG PACKAGE_URL=""
ARG PACKAGE_MD5=""
ARG LOCAL_ARTIFACT_DIR="artifact/build/.empty"
# ${LOCAL_ARTIFACT_DIR} must exist in the build context: build_image.sh
# creates an empty one for ARTIFACT_SOURCE=release (so no local package ends
# up in an image layer)
COPY ${LOCAL_ARTIFACT_DIR}/ /workspace/inference_artifact_local/
RUN --mount=type=secret,id=github_token,required=true \
    if [ "$ARTIFACT_SOURCE" = "local" ] ; then \
        mv /workspace/inference_artifact_local /workspace/inference_artifact ; \
    elif [ "$ARTIFACT_SOURCE" = "release" ] ; then \
        rm -rf /workspace/inference_artifact_local && \
        test -n "$PACKAGE_URL" && test -n "$PACKAGE_MD5" && \
        curl -fL --retry 2 --retry-delay 10 \
            -H "Authorization: Bearer $(cat /run/secrets/github_token)" \
            -H "Accept: application/octet-stream" \
            -o inference_artifact.zip "$PACKAGE_URL" && \
        echo "$PACKAGE_MD5  inference_artifact.zip" | md5sum -c - && \
        mkdir -p /workspace/inference_artifact && \
        unzip inference_artifact.zip -d /workspace/inference_artifact && \
        rm inference_artifact.zip ; \
    else \
        echo "unknown ARTIFACT_SOURCE=$ARTIFACT_SOURCE (local or release)" && exit 1 ; \
    fi
RUN --mount=type=secret,id=github_token,required=true \
    GITHUB_TOKEN="$(cat /run/secrets/github_token)" \
    python -m clara_inference.fetch_checkpoint /workspace/inference_artifact
# List files in the inference artifact directory for verification
RUN ls -la /workspace/inference_artifact

# The forecast pipeline (see README.md): configured with CLARA_* environment
# variables, or with options as arguments after the image name
RUN python -m clara_inference --help > /dev/null
ENTRYPOINT ["python", "-m", "clara_inference"]
CMD []
