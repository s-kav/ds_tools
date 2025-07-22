def test_sparse_calc_prints_output(tools, sample_sparse_df, capsys):
    """Tests that sparse_calc prints the sparsity level."""
    tools.sparse_calc(sample_sparse_df)
    captured = capsys.readouterr()
    assert "Level of sparsity =" in captured.out
    assert "%" in captured.out
