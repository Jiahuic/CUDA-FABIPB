#!/usr/bin/env sh

mesh_control_backend_name() {
  case "$1" in
    1) echo "msms" ;;
    2) echo "nanoshaper" ;;
    *) echo "unknown" ;;
  esac
}

mesh_control_init() {
  MESH_MODE="${MESH_MODE:-1}"
  MESH_RESOLUTION="${MESH_RESOLUTION:-1.0}"
  MESH_PARAM_OVERRIDE="${MESH_PARAM_OVERRIDE:-}"
  MESH_CONTROL_SOURCE="normalized"

  if [ -z "$MESH_PARAM_OVERRIDE" ] && [ "${MESH_DENSITY+x}" = x ] && [ -n "${MESH_DENSITY}" ]; then
    MESH_PARAM_OVERRIDE="$MESH_DENSITY"
    MESH_CONTROL_SOURCE="legacy_MESH_DENSITY"
  fi

  case "$MESH_MODE" in
    1|2) ;;
    *)
      echo "Error: unsupported MESH_MODE=$MESH_MODE (use 1 for MSMS or 2 for NanoShaper)" >&2
      exit 2
      ;;
  esac

  MESH_BACKEND="$(mesh_control_backend_name "$MESH_MODE")"
  MESH_ARG_MODE="-m=$MESH_MODE"
  if [ -n "$MESH_PARAM_OVERRIDE" ]; then
    MESH_ARG_PARAM="-d=$MESH_PARAM_OVERRIDE"
    MESH_CONTROL_LABEL="backend override"
    MESH_CONTROL_VALUE="$MESH_PARAM_OVERRIDE"
  else
    MESH_ARG_PARAM="-R=$MESH_RESOLUTION"
    MESH_CONTROL_LABEL="mesh resolution"
    MESH_CONTROL_VALUE="$MESH_RESOLUTION"
  fi
}

mesh_control_write_summary() {
  file="$1"
  panel="$2"
  {
    echo "panel: $panel"
    echo "mesh backend: $MESH_BACKEND"
    echo "mesh mode: $MESH_MODE"
    echo "mesh control source: $MESH_CONTROL_SOURCE"
    echo "mesh control: $MESH_CONTROL_LABEL = $MESH_CONTROL_VALUE"
    echo "fabipb args: $MESH_ARG_MODE $MESH_ARG_PARAM"
    echo "policy: remesh every run"
  } >"$file"
}
