#!/usr/bin/env bash
# The Developer Certificate of Origin check (CONTRIBUTING.md, "Licensing of contributions"):
# every commit in <base>..<head> must carry a "Signed-off-by:" line with its author's email,
# which `git commit -s` adds. Merge commits are skipped, so GitHub's "Update branch" does not
# fail it. Nothing else is: an author name or email can be set to anything, so a "bot" name
# proves nothing. The DCO workflow runs this on every pull request; run it yourself before
# pushing:
#
#   scripts/check_dco.sh origin/master HEAD
set -euo pipefail

base=${1:?usage: check_dco.sh <base> <head>}
head=${2:?usage: check_dco.sh <base> <head>}

shopt -s nocasematch   # emails, and the trailer's own spelling, compare case-insensitively
missing=0
for c in $(git rev-list --no-merges "$base..$head"); do
  name=$(git log -1 --format='%an' "$c")
  email=$(git log -1 --format='%ae' "$c")
  # Read every sign-off line first: a `grep -q` at the end of a pipe exits early, and under
  # pipefail the writer's broken pipe would fail a commit that is signed off.
  signoffs=$(git log -1 --format='%B' "$c" | grep -i '^Signed-off-by:' || true)
  if [[ "$signoffs" == *"<$email>"* ]]; then
    continue
  fi
  echo "::error::$(git log -1 --format='%h %s' "$c") has no 'Signed-off-by:' line from its author, $name <$email>"
  missing=$((missing + 1))
done

if [ "$missing" -gt 0 ]; then
  echo "$missing commit(s) are not signed off. Sign them off with 'git rebase --signoff $base'"
  echo "and force-push the branch; see CONTRIBUTING.md."
  exit 1
fi
echo "Every commit in $base..$head is signed off by its author."
