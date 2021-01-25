import io

import pandas as pd

from glotaran.parameter import ParameterGroup

TESTDATA = """
label;value;min;max;vary;non-negative;stderr
rates.k1;0.050;-inf;inf;True;True;0.0
rates.k2;0.509;-inf;inf;True;True;0.0
rates.k3;2.311;-inf;inf;True;True;0.0
inputs.1;0.500;-inf;inf;False;True;0.0
inputs.2;0.303;-inf;inf;True;True;0.0
inputs.3;0.180;-inf;inf;True;True;0.0
inputs.7;0.394;-inf;inf;True;True;0.0
inputs.8;0.387;-inf;inf;True;True;0.0
inputs.9;0.352;-inf;inf;True;True;0.0
inputs.10;0.288;-inf;inf;True;True;0.0
irf.center1;0.400;-inf;inf;True;False;0.0
irf.center2;0.410;-inf;inf;True;False;0.0
irf.center3;0.420;-inf;inf;True;False;0.0
irf.width;0.060;-inf;inf;True;False;0.0
irf.dispc;500.0;-inf;inf;False;False;0.0
irf.disp1;0.010;-inf;inf;True;False;0.0
irf.disp2;0.001;-inf;inf;True;False;0.0
scale.1;1.000;-inf;inf;False;False;0.0
scale.2;1.306;-inf;inf;True;False;0.0
scale.3;1.164;-inf;inf;True;False;0.0
pen.1;1.000;-inf;inf;False;False;0.0
area.1;1.000;-inf;inf;False;False;0.0
"""


def compare_df_to_parameter_group(df):
    parameter_group = ParameterGroup.from_pandas_dataframe(df)
    for label, parameter in parameter_group.all():
        row = df.loc[df["label"] == label]
        assert row["value"].values[0] == parameter.value
        assert row["min"].values[0] == parameter.minimum
        assert row["max"].values[0] == parameter.maximum
        assert row["vary"].values[0] == parameter.vary
        assert row["non-negative"].values[0] == parameter.non_negative
        # TODO: test stderr


def test_from_csv():
    df = pd.read_csv(io.StringIO(TESTDATA), sep=";")
    compare_df_to_parameter_group(df)


def test_from_csv_space_delimited():
    df = pd.read_csv(io.StringIO(TESTDATA.replace(";", " ")), sep=" ")
    compare_df_to_parameter_group(df)


def test_from_csv_tab_delimited():
    df = pd.read_csv(io.StringIO(TESTDATA.replace(";", "\t")), sep="\t")
    compare_df_to_parameter_group(df)


def test_access_parameter_attributes():
    df = pd.read_csv(io.StringIO(TESTDATA), sep=";")
    print(df)
    parameter_group = ParameterGroup.from_pandas_dataframe(df)
    for label, parameter in parameter_group.all():
        row = df.loc[df["label"] == label]
        assert bool(row["value"].values[0]) == bool(parameter)  # test whether "if parameter" works
