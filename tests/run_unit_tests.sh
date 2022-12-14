conda activate alphadia
python -m unittest unit_tests/test_cli
python -m unittest unit_tests/test_gui
python -m unittest unit_tests/test_extraction
conda deactivate