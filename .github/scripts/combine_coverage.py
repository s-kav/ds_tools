import glob
import xml.etree.ElementTree as ET

xml_files = glob.glob("*.xml")
print("Found", len(xml_files), "XML files:", xml_files)
if not xml_files:
    print("No XML files found!")
    exit(1)
base_tree = ET.parse(xml_files[0])
base_root = base_tree.getroot()
file_coverage = {}
for xml_file in xml_files:
    print("Processing", xml_file)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for package in root.findall(".//package"):
        for class_elem in package.findall("classes/class"):
            filename = class_elem.get("filename")
            if filename not in file_coverage:
                file_coverage[filename] = {"line_hits": {}, "class_elem": class_elem}
            for line in class_elem.findall("lines/line"):
                line_num = int(line.get("number"))
                hits = int(line.get("hits", 0))
                file_coverage[filename]["line_hits"][line_num] = max(
                    file_coverage[filename]["line_hits"].get(line_num, 0), hits
                )

combined_root = ET.Element("coverage")
combined_root.set("version", base_root.get("version", ""))
combined_root.set("timestamp", base_root.get("timestamp", ""))
sources = ET.SubElement(combined_root, "sources")
source = ET.SubElement(sources, "source")
source.text = "."
packages = ET.SubElement(combined_root, "packages")
package = ET.SubElement(packages, "package")
package.set("name", ".")
package.set("line-rate", "1.0")
package.set("branch-rate", "1.0")
package.set("complexity", "0")
classes = ET.SubElement(package, "classes")
total_lines = 0
total_covered = 0

for filename, data in file_coverage.items():
    class_elem = ET.SubElement(classes, "class")
    class_elem.set("name", filename.replace("/", ".").replace(".py", ""))
    class_elem.set("filename", filename)
    lines_elem = ET.SubElement(class_elem, "lines")
    file_lines = 0
    file_covered = 0
    for line_num, hits in data["line_hits"].items():
        line_elem = ET.SubElement(lines_elem, "line")
        line_elem.set("number", str(line_num))
        line_elem.set("hits", str(hits))
        file_lines += 1
        if hits > 0:
            file_covered += 1
    total_lines += file_lines
    total_covered += file_covered
    line_rate = file_covered / file_lines if file_lines > 0 else 0
    class_elem.set("line-rate", str(round(line_rate, 4)))
    class_elem.set("branch-rate", "1.0")
    class_elem.set("complexity", "0")

overall_rate = total_covered / total_lines if total_lines > 0 else 0
combined_root.set("line-rate", str(round(overall_rate, 4)))
combined_root.set("branch-rate", "1.0")
package.set("line-rate", str(round(overall_rate, 4)))
print(
    "Combined coverage:",
    total_covered,
    "/",
    total_lines,
    "=",
    str(round(overall_rate * 100, 2)) + "%",
)
combined_tree = ET.ElementTree(combined_root)
ET.indent(combined_tree, space="  ")
combined_tree.write("../combined-coverage.xml", encoding="utf-8", xml_declaration=True)
print("Saved combined coverage to combined-coverage.xml")
