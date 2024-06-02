#!/bin/sh

# Ask function for prompting and assigning default values
ask() {
  local var_name="$1"
  local default_value="$2"
  local prompt_message="$3"

  if [ -z "${!var_name}" ]; then
    read -p "${prompt_message}: " -r
    if [ -n "$REPLY" ]; then
      declare -g "$var_name=$REPLY"
    else
      declare -g "$var_name=$default_value"
    fi
  fi
}

# Prompt for the source folder for Python files
ask SOURCE_FOLDER "$1" "Please provide the path to the source folder containing algorithm's files"

# Check if the source folder exists
if [ ! -d "${SOURCE_FOLDER}" ]; then
  echo "Source folder not found: ${SOURCE_FOLDER}"
  exit 1
fi

# Prompt for the keyword to rename a file to __init__.py
ask KEYWORD "$2" "Please provide a keyword to rename the master algoritm to __init__.py"

# Save the original folder name
ORIGINAL_FOLDER="./algorithm_name/"

# Prompt for other details
ask DOCKER_ID "$3" "Please provide your Docker ID"
ask ALGORITHM_NAME "$4" "Please provide the name of the Vantage6 algorithm to dockerise"
ask ALGORITHM_VERSION "$5" "Please provide the version of the Vantage6 algorithm to dockerise"
ask ALGORITHM_DESCRIPTION "$6" "Please provide a short description of the Vantage6 algorithm to dockerise"

# Rename the folder
mv "${ORIGINAL_FOLDER}" "./${ALGORITHM_NAME}/"

# set names for specific algorithm
sed -i 's/algorithm_name/'"${ALGORITHM_NAME}"'/' Dockerfile
sed -i 's/algorithm_name/'"${ALGORITHM_NAME}"'/' setup.py
if [ -n "${ALGORITHM_VERSION}" ]; then
  sed -i 's/1.0.0/'"${ALGORITHM_VERSION}"'/' setup.py
fi
if [ -n "${ALGORITHM_DESCRIPTION}" ]; then
  sed -i 's/No description available/'"${ALGORITHM_DESCRIPTION}"'/' setup.py
fi

# Copy all Python files from the source folder to the destination folder
cp "${SOURCE_FOLDER}"/*.py "./${ALGORITHM_NAME}/"

# Rename any file containing the specified keyword to __init__.py using wildcard matching
if [ -n "${KEYWORD}" ]; then
  INIT_PY_FILE="./${ALGORITHM_NAME}/__init__.py"

  for file in "./${ALGORITHM_NAME}"/*"${KEYWORD}"*.py; do
    if [ -f "${file}" ]; then
      mv "${file}" "${INIT_PY_FILE}"
    fi
  done
fi

# Dockerize the algorithm and add a version tag
docker build -t "${ALGORITHM_NAME}:${ALGORITHM_VERSION}" .
docker tag "${ALGORITHM_NAME}:${ALGORITHM_VERSION}" "${DOCKER_ID}/${ALGORITHM_NAME}:${ALGORITHM_VERSION}"
docker push "${DOCKER_ID}/${ALGORITHM_NAME}:${ALGORITHM_VERSION}"

# Revert the folder name back to the original
mv "./${ALGORITHM_NAME}/" "${ORIGINAL_FOLDER}"

# Set names back to default state
sed -i 's/'"${ALGORITHM_NAME}"'/algorithm_name/' Dockerfile
sed -i 's/'"${ALGORITHM_NAME}"'/algorithm_name/' setup.py
if [ -n "${ALGORITHM_VERSION}" ]; then
  sed -i 's/'"${ALGORITHM_VERSION}"'/1.0.0/' setup.py
fi
if [ -n "${ALGORITHM_DESCRIPTION}" ]; then
  sed -i 's/'"${ALGORITHM_DESCRIPTION}"'/No description available/' setup.py
fi

# Remove copied Python files from the original folder, except for 'example.py'
for file in "${ORIGINAL_FOLDER}"/*.py; do
  if [ -f "$file" ] && [ "$(basename "$file")" != "example.py" ]; then
    rm "$file"
  fi
done
