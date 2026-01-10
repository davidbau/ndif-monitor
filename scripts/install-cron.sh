#!/bin/bash
# Install or uninstall NDIF Monitor cron job
#
# Usage:
#   ./scripts/install-cron.sh                              # Install with defaults
#   ./scripts/install-cron.sh --deploy /path/to/www        # Install with deploy path
#   ./scripts/install-cron.sh --interval 15                # Run every 15 minutes
#   ./scripts/install-cron.sh --uninstall                  # Remove cron entry
#   ./scripts/install-cron.sh --status                     # Show current status

set -e

# Defaults
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INTERVAL=30
DEPLOY_PATH=""
UNINSTALL=false
STATUS_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deploy)
            DEPLOY_PATH="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        --help|-h)
            echo "NDIF Monitor Cron Installer"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --deploy PATH    Deploy dashboard to PATH after each run"
            echo "  --interval MIN   Run every MIN minutes (default: 30)"
            echo "  --uninstall      Remove the cron entry"
            echo "  --status         Show current cron status"
            echo "  --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Install with 30-min interval"
            echo "  $0 --deploy /share/projects/ndif-monitor/www"
            echo "  $0 --interval 15                      # Run every 15 minutes"
            echo "  $0 --uninstall                        # Remove cron entry"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Marker comment to identify our cron entry
CRON_MARKER="# NDIF-MONITOR"

# Function to get current ndif-monitor cron entry
get_current_entry() {
    crontab -l 2>/dev/null | grep -F "$CRON_MARKER" || true
}

# Status check
if $STATUS_ONLY; then
    echo "NDIF Monitor Cron Status"
    echo "========================"
    CURRENT=$(get_current_entry)
    if [[ -n "$CURRENT" ]]; then
        echo "Installed: YES"
        echo ""
        echo "Current entry:"
        echo "  $CURRENT"
        echo ""
        # Parse interval from entry
        if [[ "$CURRENT" =~ ^\*/([0-9]+) ]]; then
            echo "Interval: every ${BASH_REMATCH[1]} minutes"
        fi
        # Check for deploy path
        if [[ "$CURRENT" =~ NDIF_DEPLOY_PATH=([^[:space:]]+) ]]; then
            echo "Deploy path: ${BASH_REMATCH[1]}"
        fi
    else
        echo "Installed: NO"
        echo ""
        echo "Run '$0' to install"
    fi
    exit 0
fi

# Uninstall
if $UNINSTALL; then
    CURRENT=$(get_current_entry)
    if [[ -z "$CURRENT" ]]; then
        echo "No NDIF Monitor cron entry found."
        exit 0
    fi

    echo "Removing NDIF Monitor cron entry..."
    echo ""
    echo "Current entry:"
    echo "  $CURRENT"
    echo ""

    # Remove the entry
    crontab -l 2>/dev/null | grep -v -F "$CRON_MARKER" | crontab -

    echo "Removed successfully."
    exit 0
fi

# Install
echo "NDIF Monitor Cron Installer"
echo "============================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Run interval: every $INTERVAL minutes"
if [[ -n "$DEPLOY_PATH" ]]; then
    echo "Deploy path: $DEPLOY_PATH"
fi
echo ""

# Check if already installed
CURRENT=$(get_current_entry)
if [[ -n "$CURRENT" ]]; then
    echo "Existing entry found:"
    echo "  $CURRENT"
    echo ""
    read -p "Replace with new entry? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    # Remove old entry
    crontab -l 2>/dev/null | grep -v -F "$CRON_MARKER" | crontab -
fi

# Build the cron entry
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Build environment exports
ENV_EXPORTS=""
if [[ -n "$DEPLOY_PATH" ]]; then
    ENV_EXPORTS="NDIF_DEPLOY_PATH=$DEPLOY_PATH "
fi

# The cron entry
CRON_ENTRY="*/$INTERVAL * * * * ${ENV_EXPORTS}$SCRIPT_DIR/run-cycle.sh >> $LOG_DIR/cron.log 2>&1 $CRON_MARKER"

echo "Adding cron entry:"
echo "  $CRON_ENTRY"
echo ""

# Add to crontab
(crontab -l 2>/dev/null || true; echo "$CRON_ENTRY") | crontab -

echo "Installed successfully!"
echo ""
echo "Useful commands:"
echo "  crontab -l                        # View all cron entries"
echo "  tail -f $LOG_DIR/cron.log         # Watch log output"
echo "  $0 --status                       # Check installation status"
echo "  $0 --uninstall                    # Remove cron entry"
echo ""

# Verify
echo "Verification:"
get_current_entry | sed 's/^/  /'
