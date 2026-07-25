#!/usr/bin/env bash
# Clone THIS repo's committed HEAD into a throwaway directory and run the
# documented quickstart exactly as a stranger would.
#
# Only committed files exist inside the clone. Anything untracked that the
# quickstart needs is a BUG, and finding that is the entire point of this script.
# It is what caught the missing uv.lock: `docker build` failed at
# `COPY pyproject.toml uv.lock ./` because .gitignore was hiding the file.
#
# Stages run in increasing cost order and each is skippable, so a machine
# without Docker still gets signal from the cheap stages instead of one blanket
# failure.
#
#   ./scripts/verify_fresh_clone.sh                 # everything available
#   SKIP_DOCKER=1 ./scripts/verify_fresh_clone.sh   # no Docker daemon
#   SKIP_WEB=1    ./scripts/verify_fresh_clone.sh   # no pnpm
#
# Never make this pass by weakening a check. If the README is wrong, fix the
# README and re-run the fixed version.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
cleanup() { cd /; rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

: "${SKIP_DOCKER:=0}"
: "${SKIP_WEB:=0}"
: "${BACKEND_PORT:=8070}"

STAGES_PASSED=()
STAGES_SKIPPED=()

pass() { STAGES_PASSED+=("$1"); echo "    PASS: $1"; }
skip() { STAGES_SKIPPED+=("$1"); echo "    SKIP: $1 ($2)"; }
fail() { echo "    FAIL: $1"; echo "${2:-}"; exit 1; }

echo "==> [1/7] Cloning committed HEAD into $WORK"
git clone --quiet "$REPO_ROOT" "$WORK/clone"
cd "$WORK/clone"
pass "clone"

# ── 2. The files the quickstart depends on must be tracked ──────────────────
echo "==> [2/7] Files the documented quickstart needs"
[ -f uv.lock ]            || fail "uv.lock is not tracked" \
  "Dockerfile does 'COPY pyproject.toml uv.lock ./' then 'uv sync --frozen'. Without the lockfile a fresh clone cannot build. Fix: remove uv.lock from .gitignore and 'git add -f uv.lock'."
[ -f LICENSE ]            || fail "LICENSE is missing" "The README badge links to it."
[ -f .env.example ]       || fail ".env.example is missing" "The quickstart copies it to .env."
[ -f web/pnpm-lock.yaml ] || fail "web/pnpm-lock.yaml is not tracked" "'pnpm install --frozen-lockfile' needs it."
[ -f infra/docker/api.Dockerfile ] || fail "infra/docker/api.Dockerfile is missing" ""
[ -f infra/compose/base.yml ]      || fail "infra/compose/base.yml is missing" ""
[ -d data/sample ]        || fail "data/sample/ is missing" "CI and the e2e test run against these fixtures."
pass "required files are tracked"

# ── 3. No agent scaffolding or retracted claim escaped into the clone ────────
echo "==> [3/7] Repository hygiene"
if git ls-files | grep -iE 'CLAUDE\.md|AGENTS\.md|PORT-MAP|^PORTS\.md|^PLAN\.md|LOOP_|FABLE|\.handoff|^\.claude/|\.bak$'; then
  fail "agent scaffolding is tracked" "Listed above. Add to .gitignore and 'git rm --cached'."
fi
if ls src/data/generate_*.py >/dev/null 2>&1; then
  fail "a synthetic data generator is present" \
    "ADR-0003 deleted all of them. A generator in src/data/ means the metric path can be poisoned again."
fi
if grep -q "YOUR_USERNAME" README.md; then
  fail "README still contains the YOUR_USERNAME placeholder" ""
fi
if ! grep -q "A correction, and why it's here" README.md; then
  fail "the 0.964 retraction section is missing from README.md" \
    "It is required above the fold. See docs/adr/0003-real-data-over-synthetic.md."
fi
pass "hygiene"

# ── 4. Every relative link in the README resolves inside the clone ───────────
echo "==> [4/7] README links resolve"
BROKEN=0
while read -r target; do
  [ -z "$target" ] && continue
  [ -e "${target%%#*}" ] || { echo "    BROKEN LINK: $target"; BROKEN=1; }
done < <(grep -oE '\]\(([^)#h][^)]*)\)' README.md | sed -E 's/^\]\(//; s/\)$//')
[ "$BROKEN" -eq 0 ] || fail "README links point at files that are not committed" ""
pass "README links"

# ── 5. Python: install and test exactly as the README says ──────────────────
echo "==> [5/7] make setup && make test"
if ! command -v uv >/dev/null 2>&1; then
  skip "python setup + tests" "uv is not installed"
else
  cp .env.example .env
  uv sync --frozen --extra dev >/dev/null 2>&1 || fail "uv sync --frozen failed" \
    "The lockfile does not resolve against pyproject.toml. Run 'uv lock' and commit the result."
  pass "uv sync --frozen"

  # -m 'not network' matches what CI runs. No credentials, no downloads.
  if uv run pytest -m "not network" >"$WORK/pytest.log" 2>&1; then
    SUMMARY="$(grep -oE '[0-9]+ passed[^=]*' "$WORK/pytest.log" | tail -1 | sed 's/ *$//')"
    pass "pytest — ${SUMMARY:-completed}"
  else
    tail -40 "$WORK/pytest.log"
    fail "the test suite does not pass from a fresh clone" ""
  fi
fi

# ── 6. Web: install, typecheck, test, build ─────────────────────────────────
echo "==> [6/7] web build"
if [ "$SKIP_WEB" = "1" ]; then
  skip "web" "SKIP_WEB=1"
elif ! command -v pnpm >/dev/null 2>&1; then
  skip "web" "pnpm is not installed"
else
  (
    cd web
    pnpm install --frozen-lockfile >/dev/null 2>&1 || exit 1
    pnpm typecheck >/dev/null 2>&1 || exit 2
    pnpm test >/dev/null 2>&1 || exit 3
    pnpm build >/dev/null 2>&1 || exit 4
  ) || fail "web pipeline failed (install/typecheck/test/build)" "Re-run 'cd web && pnpm build' for detail."
  pass "web install + typecheck + test + build"
fi

# ── 7. Docker: build the API image and assert /health actually answers ──────
echo "==> [7/7] docker compose up + /health"
if [ "$SKIP_DOCKER" = "1" ]; then
  skip "docker" "SKIP_DOCKER=1"
elif ! docker info >/dev/null 2>&1; then
  skip "docker" "no reachable Docker daemon"
else
  docker build -f infra/docker/api.Dockerfile -t ml-api:verify . >"$WORK/build.log" 2>&1 \
    || { tail -30 "$WORK/build.log"; fail "docker build failed" "Most often: uv.lock missing or stale."; }
  pass "docker build (API image)"

  docker compose -f infra/compose/base.yml up -d --wait >"$WORK/compose.log" 2>&1 \
    || { tail -30 "$WORK/compose.log"; fail "docker compose up failed" ""; }

  HEALTHY=0
  for i in $(seq 1 60); do
    if curl -fsS "http://localhost:${BACKEND_PORT}/health" >"$WORK/health.json" 2>/dev/null; then
      HEALTHY=1
      echo "    health OK after ${i}s: $(cat "$WORK/health.json")"
      break
    fi
    sleep 1
  done

  if [ "$HEALTHY" -ne 1 ]; then
    docker compose -f infra/compose/base.yml logs --tail=50
    docker compose -f infra/compose/base.yml down -v >/dev/null 2>&1 || true
    fail "the API never became healthy on :${BACKEND_PORT}" \
      "Check that nothing else holds the port: lsof -nP -iTCP:${BACKEND_PORT} -sTCP:LISTEN"
  fi
  pass "GET /health"

  # A fresh clone has no checkpoints, so /predict MUST answer 503 (untrained),
  # never 500 (bug) and never 200 (which would mean a model came from nowhere).
  CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST \
    -H 'Content-Type: application/json' -d '{}' \
    "http://localhost:${BACKEND_PORT}/predict/fraud" || echo 000)"
  if [ "$CODE" = "503" ]; then
    pass "POST /predict/fraud -> 503 on an untrained clone (correct)"
  elif [ "$CODE" = "200" ]; then
    docker compose -f infra/compose/base.yml down -v >/dev/null 2>&1 || true
    fail "POST /predict/fraud returned 200 on a clone with no checkpoints" \
      "A model appeared from nowhere. Check that checkpoints/ is gitignored."
  else
    docker compose -f infra/compose/base.yml down -v >/dev/null 2>&1 || true
    fail "POST /predict/fraud returned $CODE, expected 503" ""
  fi

  docker compose -f infra/compose/base.yml down -v >/dev/null 2>&1 || true
  pass "teardown"
fi

echo
echo "──────────────────────────────────────────────────────────────"
echo "PASSED  (${#STAGES_PASSED[@]}): ${STAGES_PASSED[*]}"
if [ "${#STAGES_SKIPPED[@]}" -gt 0 ]; then
  echo "SKIPPED (${#STAGES_SKIPPED[@]}): ${STAGES_SKIPPED[*]}"
fi
echo "──────────────────────────────────────────────────────────────"
echo "PASS: a fresh clone reproduces the documented quickstart."
