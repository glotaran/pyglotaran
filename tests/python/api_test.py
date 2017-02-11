from glotaran import load, parse

fitspec = '''
type: kinetic

parameters:
  - 0.1
  - 0.2

compartments: [s1, s2]

megacomplexes:
    - label: mc1
      k_matrices: [k1]

k_matrices:
  - label: "k1"
    type: sequential
    compartments: [s1, s2, s3]
    parameters: [1, 2, 3]
    matrix: {
      '("s1","s1")': 1,
      '("s2","s2")': 2,
    }

irf:
  - label: irf1
    type: gaussian
    center: 3
    width: 4

datasets:
  - label: dataset1
    megacomplexes: [mc1]
    irf: irf1
'''

gta1 = parse(fitspec)
print("Parsing from string")
print(gta1.k_matrices)
print(gta1)
gta1.eval()

print("Loading from file")
gta2 = load('../resources/test_glotaran_api_spec.yml')
print(gta2)

