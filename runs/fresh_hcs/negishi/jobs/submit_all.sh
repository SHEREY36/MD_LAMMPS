#!/bin/bash
# Kept for compatibility: the sweep is now generated and submitted by
# negishi/submit_hcs.sh (standby QOS, all cases at once).
exec bash "$(dirname "${BASH_SOURCE[0]}")/../submit_hcs.sh" "$@"
