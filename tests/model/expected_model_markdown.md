# Model

## Dataset Groups

- **testgroup**

  - _Label_: testgroup
  - _Residual Function_: non_negative_least_squares
  - _Link Clp_: True

- **default**
  - _Label_: default
  - _Residual Function_: variable_projection

## Weights

- **&nbsp;**
  - _Datasets_: ['d1', 'd2']
  - _Global Interval_: (1, 4)
  - _Model Interval_: (2, 3)
  - _Value_: 5.4

## Test Item

- **t1**

  - _Label_: t1
  - _Param_: foo
  - _Param List_: ['bar', 'baz']
  - _Param Dict_: {('s1', 's2'): 'baz'}
  - _Megacomplex_: m1
  - _Number_: 42

- **t2**
  - _Label_: t2
  - _Param_: baz
  - _Param List_: ['foo']
  - _Param Dict_: {}
  - _Megacomplex_: m2
  - _Number_: 7

## Megacomplex

- **m1**

  - _Label_: m1
  - _Type_: simple
  - _Dimension_: model
  - _Test Item_: t2

- **m2**
  - _Label_: m2
  - _Type_: dataset
  - _Dimension_: model2

## Test Item Dataset

- **t1**

  - _Label_: t1
  - _Param_: foo
  - _Param List_: ['bar', 'baz']
  - _Param Dict_: {('s1', 's2'): 'baz'}
  - _Megacomplex_: m1
  - _Number_: 42

- **t2**
  - _Label_: t2
  - _Param_: baz
  - _Param List_: ['foo']
  - _Param Dict_: {}
  - _Megacomplex_: m2
  - _Number_: 7

## Dataset

- **dataset1**

  - _Label_: dataset1
  - _Group_: default
  - _Force Index Dependent_: False
  - _Megacomplex_: ['m1']
  - _Scale_: scale_1
  - _Test Item Dataset_: t1
  - _Test Property Dataset1_: 1
  - _Test Property Dataset2_: bar

- **dataset2**
  - _Label_: dataset2
  - _Group_: testgroup
  - _Force Index Dependent_: False
  - _Megacomplex_: ['m2']
  - _Global Megacomplex_: ['m1']
  - _Scale_: scale_2
  - _Test Item Dataset_: t2
  - _Test Property Dataset1_: 1
  - _Test Property Dataset2_: bar
