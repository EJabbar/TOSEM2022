import xml.etree.ElementTree as ET
import os

def get_covered_lines(file_path):
    class_cvlines = {}
    if not os.path.exists(file_path):
        return class_cvlines
    tree = ET.parse(file_path)
    root = tree.getroot()
    for pkgs in root.findall('packages'):
        for pkg in pkgs.findall('package'):
            for clz in pkg.findall('classes'):
                for cls in clz.findall('class'):
                    cl_name = cls.get('name')
                    class_cvlines[cl_name] = []
                    for ls in cls.findall('lines'):
                        for l in ls.findall('line'):
                            l_n = int(l.get('number'))
                            l_h = int(l.get('hits'))
                            if l_h > 0:
                                class_cvlines[cl_name].append(l_n)
    return class_cvlines

# num_of_versions = [90, 50, 21, 72, 83, 10, 85, 86, 87, 13, 33, 35, 245, 47, 32, 31, 380, 383]

# dct = {}
# cl = get_covered_lines('./coverage_1_6.xml')
# dct['codec_1_0'] = cl
# print(dct)