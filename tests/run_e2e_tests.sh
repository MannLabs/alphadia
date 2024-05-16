cd e2e_tests || exit
# conda activate alphadia

python prepare_test_data.py
ls *
ls */*
# conda deactivate
cd -
