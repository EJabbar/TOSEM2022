if ! command -v java &>/dev/null; then
  echo "Please install Java"
  exit 1
fi

if ! command -v defects4j &>/dev/null; then
  echo "Please install Defects4j:"
  echo "https://github.com/rjust/defects4j"
  exit 1
fi