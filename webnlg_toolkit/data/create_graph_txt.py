import argparse
import xml.etree.ElementTree as ET

def create_graph_txt(xml_file_path, txt_file_path):
    # Parse XML
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    all_modified_triplesets = []

    for entry in root.findall('.//entry'):
        modified_tripleset = []
        
        for modifiedtripleset in entry.findall('.//modifiedtripleset'):
            for mtriple in modifiedtripleset.findall('mtriple'):
                modified_tripleset.append(mtriple.text)
        
        all_modified_triplesets.append(modified_tripleset)

    # Save list
    with open(txt_file_path, 'w') as file:
        for item in all_modified_triplesets:
            file.write(f"{item}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert XML data to text file containing graph triples.")
    
    # Add parameters
    parser.add_argument('xml_file', type=str, help="Path to the input XML file.")
    parser.add_argument('txt_file', type=str, help="Path to the output text file.")
    
    args = parser.parse_args()
    
    create_graph_txt(args.xml_file, args.txt_file)

    print(f"txt file is generated at {args.txt_file} !")
