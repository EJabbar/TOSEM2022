import os
import sys
import xml.etree.ElementTree as ET

project_dir = sys.argv[1]
javac_path = sys.argv[2]

build_path = project_dir+"maven-build.xml" if os.path.exists(project_dir+"maven-build.xml") else project_dir+"build.xml"

xml_tree = ET.parse(build_path)

xml_root = xml_tree.getroot()


mutation_property = ET.Element('property')
mutation_property.attrib['name'] = "mutation" 
mutation_property.attrib['value'] = ":ALL"
xml_root.insert(0, mutation_property)


mutation_property = ET.Element('property')
mutation_property.attrib['name'] = "mutator" 
mutation_property.attrib['value'] = "-XMutator${mutation}"
xml_root.insert(1, mutation_property)


mutation_property = ET.Element('property')
mutation_property.attrib['name'] = "major" 
mutation_property.attrib['value'] = javac_path
xml_root.insert(2, mutation_property)

for trg in xml_root.iter('target'):
    if trg.attrib['name'] == "compile" or trg.attrib['name'] == "compile.main" or trg.attrib['name'] == "compile.zoneinfo":
        javac_arg = trg.find('javac')
        if javac_arg is not None:
            javac_arg.attrib['executable'] = "${major}"
            javac_arg.attrib['fork'] = "yes"
            cmplarg = ET.SubElement(javac_arg, "compilerarg")
            cmplarg.attrib['line'] = "-J-Dmajor.export.mutants=true -J-Dmajor.export.directory=./mutants ${mutator}"
        
    if 'compile' in trg.attrib['name'] and 'test' in trg.attrib['name']:
        javac_arg = trg.find('javac')
        javac_arg.attrib['executable'] = "${major}"

xml_tree.write(build_path)
