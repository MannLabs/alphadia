conda activate alphadia
#python -m unittest unit_tests/test_cli.py
#python -m unittest unit_tests/test_gui.py
coverage run --source=../alphadia/extraction -m pytest
# coverage run --source=../alphadia/extraction -m pytest && coverage html && coverage-badge -f -o ../coverage.svg
conda deactivate
