#!/usr/bin/env sh
set -eu

START_COMMIT="${START_COMMIT:-a03c584}"
END_COMMIT="${END_COMMIT:-HEAD}"
BASE_DIR="${BASE_DIR:-/tmp/fabipb-history}"
CASE_NAME="${CASE_NAME:-1a63}"
MODE="${1:-setup}"

usage() {
  cat <<USAGE
Usage:
  $0 [setup|list|commands|clean]

Environment overrides:
  START_COMMIT   First commit in range (default: a03c584)
  END_COMMIT     Last commit in range (default: HEAD)
  BASE_DIR       Worktree parent directory (default: /tmp/fabipb-history)
  CASE_NAME      Test protein base name (default: 1a63)

Modes:
  setup     Create one worktree per commit in the range.
  list      Print the commit list in chronological order.
  commands  Print build/test commands for each prepared worktree.
  clean     Remove the worktrees under BASE_DIR.
USAGE
}

ensure_repo_root() {
  if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "Error: run this script from inside the git repository." >&2
    exit 2
  fi
}

commit_list() {
  git log --reverse --format='%H %h %s' "${START_COMMIT}^..${END_COMMIT}"
}

worktree_dir() {
  short_sha="$1"
  printf '%s/%s-%s\n' "$BASE_DIR" "$short_sha" "$CASE_NAME"
}

setup_worktrees() {
  mkdir -p "$BASE_DIR"
  commit_list | while IFS=' ' read -r full_sha short_sha subject_rest; do
    dir="$(worktree_dir "$short_sha")"
    if [ -d "$dir/.git" ] || [ -f "$dir/.git" ]; then
      echo "exists  $short_sha  $dir"
      continue
    fi
    echo "create  $short_sha  $dir"
    git worktree add "$dir" "$full_sha"
  done
}

list_commits() {
  commit_list
}

print_commands() {
  commit_list | while IFS=' ' read -r full_sha short_sha subject_rest; do
    dir="$(worktree_dir "$short_sha")"
    cat <<CMD
[$short_sha] $subject_rest
cd $dir
cmake -S . -B build
cmake --build build
env FABIPB_SETUP_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build/fabipb -B=1 -g=0 -m=1 $CASE_NAME
CMD
    printf '\n'
  done
}

clean_worktrees() {
  if [ ! -d "$BASE_DIR" ]; then
    echo "Nothing to clean: $BASE_DIR does not exist."
    return
  fi
  commit_list | while IFS=' ' read -r full_sha short_sha subject_rest; do
    dir="$(worktree_dir "$short_sha")"
    if [ -d "$dir" ]; then
      echo "remove  $short_sha  $dir"
      git worktree remove --force "$dir"
    fi
  done
}

ensure_repo_root

case "$MODE" in
  setup)
    setup_worktrees
    ;;
  list)
    list_commits
    ;;
  commands)
    print_commands
    ;;
  clean)
    clean_worktrees
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Error: unknown mode '$MODE'" >&2
    usage >&2
    exit 2
    ;;
esac
