current_version=$(grep "__version__" ../alpakaweird/__init__.py | cut -f3 -d ' ' | sed 's/"//g')
current_version_as_regex=$(echo $current_version | sed 's/\./\\./g')
echo current_version_as_regex=$current_version_as_regex

set +e
conda create -n version_check python=3.9 -y
pip_output=$(conda run -n version_check pip install --index-url https://test.pypi.org/simple/  alpakaweird 2>&1)
echo $pip_output=$pip_output

already_on_pypi=$(echo $pip_output | grep -c alpakaweird-"$current_version_as_regex")
echo $already_on_pypi=$already_on_pypi
set -e

if [ $already_on_pypi -ne 0 ]; then
  echo "Version is already on PyPi"
  exit 1
fi
