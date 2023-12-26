conda activate alphadia
#python -m unittest unit_tests/test_cli.py
#python -m unittest unit_tests/test_gui.py
coverage run --source=../alphadia -m pytest -k 'not slow'
# coverage run --source=../alphadia -m pytest && coverage html && coverage-badge -f -o ../coverage.svg
# coverage run --source=../alphadia -m pytest -k 'not slow' && coverage html && coverage-badge -f -o ../coverage.svg
conda deactivate
