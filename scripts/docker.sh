#!/usr/bin/env bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Compose is always invoked from DOCKER_DIR, so keep the filename relative.
COMPOSE_FILE="docker-compose-dev.yaml"
COMPOSE_BIN=(docker compose)
COMPOSE_MIN_VERSION="2.24.0"

refresh_compose_cmd() {
    COMPOSE_CMD="${COMPOSE_BIN[*]} -p medrix-flow-dev -f ${COMPOSE_FILE}"
}

refresh_compose_cmd

ensure_from_example() {
    local destination="$1"
    local example="$2"
    local label="$3"

    if [ -f "$destination" ]; then
        return 0
    fi
    if [ -f "$example" ]; then
        cp "$example" "$destination"
        echo -e "${BLUE}Created ${label} from $(basename "$example")${NC}"
        return 0
    fi
    echo -e "${YELLOW}✗ ${label} not found and no $(basename "$example") to copy from.${NC}"
    echo "Create $destination before starting Docker."
    exit 1
}

require_compose_file() {
    if [ -f "$DOCKER_DIR/$COMPOSE_FILE" ]; then
        return 0
    fi
    echo -e "${YELLOW}✗ ${COMPOSE_FILE} not found at $DOCKER_DIR/${COMPOSE_FILE}${NC}"
    echo "Run this from the Anaxa repository root, e.g. 'make docker-start'."
    exit 1
}

probe_compose() {
    local output

    output="$(docker compose version --short 2>/dev/null || true)"
    if [ -n "$output" ]; then
        COMPOSE_BIN=(docker compose)
        COMPOSE_VERSION_RAW="$output"
        refresh_compose_cmd
        return 0
    fi
    output="$(docker-compose version --short 2>/dev/null || true)"
    if [ -n "$output" ]; then
        COMPOSE_BIN=(docker-compose)
        COMPOSE_VERSION_RAW="$output"
        refresh_compose_cmd
        return 0
    fi
    COMPOSE_VERSION_RAW=""
    return 1
}

require_compose_version() {
    local raw major minor min_major min_minor

    min_major="${COMPOSE_MIN_VERSION%%.*}"
    min_minor="${COMPOSE_MIN_VERSION#*.}"
    min_minor="${min_minor%%.*}"
    COMPOSE_VERSION_RAW=""
    probe_compose || true
    raw="${COMPOSE_VERSION_RAW#v}"
    major="${raw%%.*}"
    minor="${raw#*.}"
    minor="${minor%%.*}"
    major="${major//[!0-9]/}"
    minor="${minor//[!0-9]/}"

    if [ -z "$major" ] || [ -z "$minor" ]; then
        echo -e "${YELLOW}⚠ Could not determine the Docker Compose version; ${COMPOSE_MIN_VERSION} or newer is recommended.${NC}"
        return 0
    fi
    if [ "$major" -gt "$min_major" ] || { [ "$major" -eq "$min_major" ] && [ "$minor" -ge "$min_minor" ]; }; then
        return 0
    fi
    echo -e "${YELLOW}✗ Docker Compose ${raw} is too old — ${COMPOSE_MIN_VERSION} or newer is required.${NC}"
    echo "The development compose file uses optional env_file entries unsupported by older clients."
    echo "Update Docker Desktop, or install a current Compose v2 plugin:"
    echo "  https://docs.docker.com/compose/install/"
    exit 1
}

ensure_medrix_flow_root() {
    if [ -z "$MEDRIX_FLOW_ROOT" ]; then
        export MEDRIX_FLOW_ROOT="$PROJECT_ROOT"
    fi
}

compose_preflight() {
    require_compose_file
    require_compose_version
    ensure_medrix_flow_root
}

ensure_env_files() {
    ensure_from_example "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.example" ".env"
    ensure_from_example "$PROJECT_ROOT/frontend/.env" "$PROJECT_ROOT/frontend/.env.example" "frontend/.env"
}

detect_sandbox_mode() {
    local config_file="$PROJECT_ROOT/config.yaml"
    local sandbox_use=""
    local provisioner_url=""

    if [ ! -f "$config_file" ]; then
        echo "local"
        return
    fi

    sandbox_use=$(awk '
        /^[[:space:]]*sandbox:[[:space:]]*$/ { in_sandbox=1; next }
        in_sandbox && /^[^[:space:]#]/ { in_sandbox=0 }
        in_sandbox && /^[[:space:]]*use:[[:space:]]*/ {
            line=$0
            sub(/^[[:space:]]*use:[[:space:]]*/, "", line)
            print line
            exit
        }
    ' "$config_file")

    provisioner_url=$(awk '
        /^[[:space:]]*sandbox:[[:space:]]*$/ { in_sandbox=1; next }
        in_sandbox && /^[^[:space:]#]/ { in_sandbox=0 }
        in_sandbox && /^[[:space:]]*provisioner_url:[[:space:]]*/ {
            line=$0
            sub(/^[[:space:]]*provisioner_url:[[:space:]]*/, "", line)
            print line
            exit
        }
    ' "$config_file")

    if [[ "$sandbox_use" == *"medrix_flow.sandbox.local:LocalSandboxProvider"* ]]; then
        echo "local"
    elif [[ "$sandbox_use" == *"medrix_flow.community.aio_sandbox:AioSandboxProvider"* ]]; then
        if [ -n "$provisioner_url" ]; then
            echo "provisioner"
        else
            echo "aio"
        fi
    else
        echo "local"
    fi
}

# Cleanup function for Ctrl+C
cleanup() {
    echo ""
    echo -e "${YELLOW}Operation interrupted by user${NC}"
    exit 130
}

# Set up trap for Ctrl+C
trap cleanup INT TERM

docker_available() {
    # Check that the docker CLI exists
    if ! command -v docker >/dev/null 2>&1; then
        return 1
    fi

    # Check that the Docker daemon is reachable
    if ! docker info >/dev/null 2>&1; then
        return 1
    fi

    return 0
}

# Initialize: pre-pull the sandbox image so first Pod startup is fast
init() {
    echo "=========================================="
    echo "  MedrixFlow Init — Pull Sandbox Image"
    echo "=========================================="
    echo ""

    SANDBOX_IMAGE="enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest"

    # Detect sandbox mode from config.yaml
    local sandbox_mode
    sandbox_mode="$(detect_sandbox_mode)"

    # Skip image pull for local sandbox mode (no container image needed)
    if [ "$sandbox_mode" = "local" ]; then
        echo -e "${GREEN}Detected local sandbox mode — no Docker image required.${NC}"
        echo ""

        if docker_available; then
            echo -e "${GREEN}✓ Docker environment is ready.${NC}"
            echo ""
            echo -e "${YELLOW}Next step: make docker-start${NC}"
        else
            echo -e "${YELLOW}Docker does not appear to be installed, or the Docker daemon is not reachable.${NC}"
            echo "Local sandbox mode itself does not require Docker, but Docker-based workflows (e.g., docker-start) will fail until Docker is available."
            echo ""
            echo -e "${YELLOW}Install and start Docker, then run: make docker-init && make docker-start${NC}"
        fi

        return 0
    fi

    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${SANDBOX_IMAGE}$"; then
        echo -e "${BLUE}Pulling sandbox image: $SANDBOX_IMAGE ...${NC}"
        echo ""

        if ! docker pull "$SANDBOX_IMAGE" 2>&1; then
            echo ""
            echo -e "${YELLOW}⚠ Failed to pull sandbox image.${NC}"
            echo ""
            echo "This is expected if:"
            echo "  1. You are using local sandbox mode (default — no image needed)"
            echo "  2. You are behind a corporate proxy or firewall"
            echo "  3. The registry requires authentication"
            echo ""
            echo -e "${GREEN}The Docker development environment can still be started.${NC}"
            echo "If you need AIO sandbox (container-based execution):"
            echo "  - Ensure you have network access to the registry"
            echo "  - Or configure a custom sandbox image in config.yaml"
            echo ""
            echo -e "${YELLOW}Next step: make docker-start${NC}"
            return 0
        fi
    else
        echo -e "${GREEN}Sandbox image already exists locally: $SANDBOX_IMAGE${NC}"
    fi

    echo ""
    echo -e "${GREEN}✓ Sandbox image is ready.${NC}"
    echo ""
    echo -e "${YELLOW}Next step: make docker-start${NC}"
}

# Start Docker development environment
start() {
    local sandbox_mode
    local services

    echo "=========================================="
    echo "  Starting Anaxa Docker Development"
    echo "=========================================="
    echo ""

    compose_preflight
    ensure_env_files

    sandbox_mode="$(detect_sandbox_mode)"

    if [ "$sandbox_mode" = "provisioner" ]; then
        services="frontend gateway langgraph provisioner nginx"
    else
        services="frontend gateway langgraph nginx"
    fi

    echo -e "${BLUE}Detected sandbox mode: $sandbox_mode${NC}"
    if [ "$sandbox_mode" = "provisioner" ]; then
        echo -e "${BLUE}Provisioner enabled (Kubernetes mode).${NC}"
    else
        echo -e "${BLUE}Provisioner disabled (not required for this sandbox mode).${NC}"
    fi
    echo ""
    
    echo -e "${BLUE}Using MEDRIX_FLOW_ROOT=$MEDRIX_FLOW_ROOT${NC}"
    echo ""
    
    # Ensure config.yaml exists before starting.
    if [ ! -f "$PROJECT_ROOT/config.yaml" ]; then
        if [ -f "$PROJECT_ROOT/config.example.yaml" ]; then
            cp "$PROJECT_ROOT/config.example.yaml" "$PROJECT_ROOT/config.yaml"
            echo ""
            echo -e "${YELLOW}============================================================${NC}"
            echo -e "${YELLOW}  config.yaml has been created from config.example.yaml.${NC}"
            echo -e "${YELLOW}  Please edit config.yaml to set your API keys and model   ${NC}"
            echo -e "${YELLOW}  configuration before starting MedrixFlow.                  ${NC}"
            echo -e "${YELLOW}============================================================${NC}"
            echo ""
            echo -e "${YELLOW}  Edit the file:  $PROJECT_ROOT/config.yaml${NC}"
            echo -e "${YELLOW}  Then run:        make docker-start${NC}"
            echo ""
            exit 0
        else
            echo -e "${YELLOW}✗ config.yaml not found and no config.example.yaml to copy from.${NC}"
            exit 1
        fi
    fi

    # Ensure extensions_config.json exists as a file before mounting.
    # Docker creates a directory when bind-mounting a non-existent host path.
    if [ ! -f "$PROJECT_ROOT/extensions_config.json" ]; then
        if [ -f "$PROJECT_ROOT/extensions_config.example.json" ]; then
            cp "$PROJECT_ROOT/extensions_config.example.json" "$PROJECT_ROOT/extensions_config.json"
            echo -e "${BLUE}Created extensions_config.json from example${NC}"
        else
            echo "{}" > "$PROJECT_ROOT/extensions_config.json"
            echo -e "${BLUE}Created empty extensions_config.json${NC}"
        fi
    fi

    echo "Building and starting containers..."
    cd "$DOCKER_DIR" && $COMPOSE_CMD up --build -d --remove-orphans $services
    echo ""
    echo "=========================================="
    echo "  MedrixFlow Docker is starting!"
    echo "=========================================="
    echo ""
    echo "  🌐 Application: http://localhost:6200"
    echo "  📡 API Gateway: http://localhost:6200/api/*"
    echo "  🤖 LangGraph:   http://localhost:6200/api/langgraph/*"
    echo ""
    echo "  📋 View logs: make docker-logs"
    echo "  🛑 Stop:      make docker-stop"
    echo ""
}

# View Docker development logs
logs() {
    local service=""
    
    case "$1" in
        --frontend)
            service="frontend"
            echo -e "${BLUE}Viewing frontend logs...${NC}"
            ;;
        --gateway)
            service="gateway"
            echo -e "${BLUE}Viewing gateway logs...${NC}"
            ;;
        --nginx)
            service="nginx"
            echo -e "${BLUE}Viewing nginx logs...${NC}"
            ;;
        --provisioner)
            service="provisioner"
            echo -e "${BLUE}Viewing provisioner logs...${NC}"
            ;;
        "")
            echo -e "${BLUE}Viewing all logs...${NC}"
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            echo "Usage: $0 logs [--frontend|--gateway|--nginx|--provisioner]"
            exit 1
            ;;
    esac
    
    compose_preflight
    cd "$DOCKER_DIR" && $COMPOSE_CMD logs -f $service
}

# Stop Docker development environment
stop() {
    # MEDRIX_FLOW_ROOT is referenced in docker-compose-dev.yaml; set it before
    # running compose down to suppress "variable is not set" warnings.
    compose_preflight
    echo "Stopping Docker development services..."
    cd "$DOCKER_DIR" && $COMPOSE_CMD down
    echo "Cleaning up sandbox containers..."
    "$SCRIPT_DIR/cleanup-containers.sh" medrix-flow-sandbox 2>/dev/null || true
    echo -e "${GREEN}✓ Docker services stopped${NC}"
}

# Restart Docker development environment
restart() {
    echo "========================================"
    echo "  Restarting MedrixFlow Docker Services"
    echo "========================================"
    echo ""
    compose_preflight
    echo -e "${BLUE}Restarting containers...${NC}"
    cd "$DOCKER_DIR" && $COMPOSE_CMD restart
    echo ""
    echo -e "${GREEN}✓ Docker services restarted${NC}"
    echo ""
    echo "  🌐 Application: http://localhost:6200"
    echo "  📋 View logs: make docker-logs"
    echo ""
}

# Show help
help() {
    echo "MedrixFlow Docker Management Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  init          - Pull the sandbox image (speeds up first Pod startup)"
    echo "  start         - Start Docker services (auto-detects sandbox mode from config.yaml)"
    echo "  restart       - Restart all running Docker services"
    echo "  logs [option] - View Docker development logs"
    echo "                  --frontend   View frontend logs only"
    echo "                  --gateway    View gateway logs only"
    echo "                  --nginx      View nginx logs only"
    echo "                  --provisioner View provisioner logs only"
    echo "  stop          - Stop Docker development services"
    echo "  help          - Show this help message"
    echo ""
}

main() {
    # Main command dispatcher
    case "$1" in
        init)
            init
            ;;
        start)
            start
            ;;
        restart)
            restart
            ;;
        logs)
            logs "$2"
            ;;
        stop)
            stop
            ;;
        help|--help|-h|"")
            help
            ;;
        *)
            echo -e "${YELLOW}Unknown command: $1${NC}"
            echo ""
            help
            exit 1
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
