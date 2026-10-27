ARG BASE_IMAGE="pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN apt-get update && apt-get install -y curl unzip git

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# the package version: the build context has no .git, so setuptools-scm
# can't find the tag (build_image.sh and CI pass it). NB: build args are
# environment variables in RUN steps, so none may start with ANNA_ (the
# forecast's options, see the --help check below)
ARG IMAGE_VERSION="0.0.0+unknown"
LABEL org.opencontainers.image.version="${IMAGE_VERSION}"

COPY pyproject.toml README.md ./
COPY src/ ./src

# export current torch version to constraints file
RUN python -c 'import torch; print(f"torch=={torch.__version__}")' > constraints.txt
# install with constraints
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ANNA_INFERENCE="${IMAGE_VERSION}" \
    uv pip install --break-system-packages --system --constraints constraints.txt .
# check that neural-lam imports (with the base image's torch), and show the
# versions (neural-lam-dev has no __version__)
RUN python -c 'import importlib.metadata as md, anna_inference, neural_lam, torch; print("anna-inference", anna_inference.__version__, "neural-lam", md.version("neural-lam"), "torch", torch.__version__)'

ENV CARTOPY_DATA_DIR=/usr/local/share/cartopy
RUN python -c 'import cartopy, cartopy.io.shapereader as shp; \
    cartopy.config["data_dir"] = cartopy.config["pre_existing_data_dir"]; \
    [print(shp.natural_earth(scale, category, name)) \
     for category, name in [("physical", "coastline"), ("cultural", "admin_0_boundary_lines_land")] \
     for scale in ["110m", "50m", "10m"]]'

# The inference artifact (configs, statistics, grids, model description, see
# artifact/README.md) comes from:
# - "zenodo" (default): the published inference package ${PACKAGE_URL},
#   checked against ${PACKAGE_MD5} (Zenodo record 23293148)
# - "local": the assembled package directory ${LOCAL_ARTIFACT_DIR} in the
#   build context
# The model checkpoint is then downloaded (md5-checked) from Zenodo as given in
# inference_artifact/configs/model.yaml, unless it's already there.
# If Zenodo is unavailable, both downloads fall back to the copies in the
# dmidk GitHub release artifact-anna-danra-2026-10-11 (same md5s).
ARG ARTIFACT_SOURCE="zenodo"
ARG PACKAGE_URL="https://zenodo.org/records/23293148/files/anna-danra-2026-10-11.zip?download=1"
ARG PACKAGE_MD5="d453733a56c381d63922806cd4b25472"
ARG PACKAGE_MIRROR_URL="https://github.com/dmidk/mlwp-inference-container-anna/releases/download/artifact-anna-danra-2026-10-11/anna-danra-2026-10-11.zip"
ARG CHECKPOINT_MIRROR_URL="https://github.com/dmidk/mlwp-inference-container-anna/releases/download/artifact-anna-danra-2026-10-11/danra_model.ckpt"
ARG LOCAL_ARTIFACT_DIR="artifact/build/.empty"
# ${LOCAL_ARTIFACT_DIR} must exist in the build context: build_image.sh
# creates an empty one for ARTIFACT_SOURCE=zenodo (so no local package ends
# up in an image layer)
COPY ${LOCAL_ARTIFACT_DIR}/ /workspace/inference_artifact_local/
RUN if [ "$ARTIFACT_SOURCE" = "local" ] ; then \
        mv /workspace/inference_artifact_local /workspace/inference_artifact ; \
    elif [ "$ARTIFACT_SOURCE" = "zenodo" ] ; then \
        rm -rf /workspace/inference_artifact_local && \
        test -n "$PACKAGE_URL" && test -n "$PACKAGE_MD5" && \
        for url in "$PACKAGE_URL" "$PACKAGE_MIRROR_URL" ; do \
            [ -n "$url" ] || continue ; \
            if curl -fL --connect-timeout 30 --retry 2 --retry-delay 10 --retry-all-errors \
                    -o inference_artifact.zip "$url" && \
                echo "$PACKAGE_MD5  inference_artifact.zip" | md5sum -c - ; then \
                downloaded=1 ; break ; \
            fi ; \
            echo "download from $url failed, trying the next mirror" ; \
        done && \
        test -n "$downloaded" && \
        mkdir -p /workspace/inference_artifact && \
        unzip inference_artifact.zip -d /workspace/inference_artifact && \
        rm inference_artifact.zip ; \
    else \
        echo "unknown ARTIFACT_SOURCE=$ARTIFACT_SOURCE (local or zenodo)" && exit 1 ; \
    fi
RUN python -m anna_inference.fetch_checkpoint /workspace/inference_artifact \
        ${CHECKPOINT_MIRROR_URL:+--mirror-url "$CHECKPOINT_MIRROR_URL"}
# List files in the inference artifact directory for verification
RUN ls -la /workspace/inference_artifact

# The forecast pipeline (see README.md): configured with ANNA_* environment
# variables, or with options as arguments after the image name
RUN python -m anna_inference --help > /dev/null
ENTRYPOINT ["python", "-m", "anna_inference"]
CMD []
