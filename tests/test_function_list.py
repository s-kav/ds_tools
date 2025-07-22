import pandas as pd


def test_function_list_no_docstring(tools, capsys):
    """Tests function_list on a class with no docstring."""

    class NoDocClass(type(tools)):
        __doc__ = None

    instance = NoDocClass()
    result_df = instance.function_list()

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty
    captured = capsys.readouterr()
    assert "No documentation found" in captured.out


def test_function_list_no_agenda(tools, capsys):
    """Tests function_list on a class with no 'Agenda' in docstring."""

    class NoAgendaClass(type(tools)):
        """This is a docstring without the required section."""

        __doc__ = "This is a docstring without the required section."

    instance = NoAgendaClass()
    result_df = instance.function_list()

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty
    captured = capsys.readouterr()
    assert "No 'Agenda' section found" in captured.out


def test_function_list_empty_agenda(tools, capsys):
    """Tests function_list when 'Agenda' section is empty."""

    class EmptyAgendaClass(type(tools)):
        """
        Agenda:
        -------
        """

        __doc__ = "Agenda:\n-------"

    instance = EmptyAgendaClass()
    result_df = instance.function_list()

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty
    captured = capsys.readouterr()
    assert "No tools found in the Agenda" in captured.out
