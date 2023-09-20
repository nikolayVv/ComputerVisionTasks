

def read_regions(regions_path):
    with open(regions_path, 'r') as f:
        lines = f.readlines()
        regions = len(lines) * [0]
        for i, line in enumerate(lines):
            regions[i] = [float(el) for el in line.strip().split(',')]
        return regions

def save_regions(regions, regions_path):
    with open(regions_path, 'w') as f:
        for region in regions:
            if len(region) == 1:
                line = '%d' % region[0]
            else:
                line = ','.join(['%.2f' % el for el in region])
            f.write(line + '\n')

def read_vector(vector_path):
    with open(vector_path, 'r') as f:
        lines = f.readlines()
        vector = len(lines) * [0]
        for i, line in enumerate(lines):
            vector[i] = float(line.strip())
        return vector

def save_vector(vector, vector_path):
    with open(vector_path, 'w') as f:
        for el in vector:
            f.write('%.8f\n' % el)
