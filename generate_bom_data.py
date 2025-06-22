import random
import string

def random_mesc():
    return ''.join(random.choices(string.digits, k=10))

def random_qty():
    return str(round(random.uniform(1, 50), 1))

def random_unit():
    return random.choice(['MTR', 'PCS', 'EA', 'SET'])

def random_desc():
    descs = [
        ['PIPE,', 'SMLS', 'SCH.160', 'ASTM', 'A106-B'],
        ['VALVE', 'GATE', 'DN50', 'BRASS'],
        ['TUBE,', 'SS', 'SCH.40', 'ASTM', 'A312'],
        ['FLANGE', 'WELD', 'DN80', 'CS', 'SCH.10', 'ASTM', 'A105'],
        ['TEE', 'REDUCING', 'DN100', 'CS', 'SCH.20', 'ASTM', 'A234'],
        ['ELBOW', '90DEG', 'DN65', 'CS', 'SCH.40', 'ASTM', 'A234'],
        ['PIPE,', 'SS', 'SCH.80', 'ASTM', 'A312'],
        ['PIPE,', 'SMLS', 'SCH.120', 'ASTM', 'A106-B'],
        ['PIPE,', 'SMLS', 'SCH.80', 'ASTM', 'A106-B'],
        ['VALVE', 'BALL', 'DN25', 'SS', 'ASTM', 'A351']
    ]
    return random.choice(descs)

with open('bom_labeling_template.conll', 'a', encoding='utf-8') as f:
    inch_sizes = ['1"', '1.5"', '2"', '2.5"', '3"', '4"', '6"']
    for i in range(11, 101):
        f.write(f"{i}\tITEM\n")
        f.write(f"{random_qty()}\tQTY\n")
        f.write(f"{random_unit()}\tUNIT\n")
        f.write(f"{random.choice(inch_sizes)}\tDESC\n")
        for d in random_desc():
            f.write(f"{d}\tDESC\n")
        f.write(f"{random_mesc()}\tMESC\n\n")