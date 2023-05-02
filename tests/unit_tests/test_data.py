def test_calculate_score_group_limits():
    for f in [hybridselection.calculate_score_group_limits, hybridselection.calculate_score_group_limits.py_func]:
        for i in range(1000):
            mz_values = np.random.random(100)*1000
            intensity_values = np.floor(np.random.random(100) * 2)
            
            min = mz_values[intensity_values != 0.].min()
            max = mz_values[intensity_values != 0.].max()

            limits = hybridselection.calculate_score_group_limits(
                mz_values,
                intensity_values
            )

            assert limits.shape == (1,2)
            assert np.allclose(limits[0,0] , min)
            assert np.allclose(limits[0,1] , max)

test_calculate_score_group_limits()