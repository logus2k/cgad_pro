import pandas as pd

def parse_mesh_txt_to_excel(input_filename, output_filename):
    """
    Parses a specific text mesh format into an Excel file with 'coord' and 'conec' sheets,
    matching the structure of 'mesh_data_quad8.xlsx'.

    Args:
        input_filename (str): Path to the input text file.
        output_filename (str): Path for the output Excel file.
    """
    nodes = {}
    elements = []
    
    print(f"Parsing input file: {input_filename}")
    
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    i = 0
    # Find the start of the NODE INFORMATION section
    while i < len(lines) and "NODE INFORMATION" not in lines[i]:
        i += 1
    
    if i >= len(lines):
        raise ValueError("Could not find 'NODE INFORMATION' section.")

    print("Parsing node coordinates...")
    # Parse nodes until we hit the ELEMENT INFORMATION section or run out of lines
    while i < len(lines):
        line = lines[i].strip()
        
        if "ELEMENT INFORMATION" in line:
            break # Stop parsing nodes when we reach elements
        
        if line.startswith("Label"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    node_id = int(parts[1])
                    
                    # Read the next line for coordinates
                    i += 1
                    coord_line = lines[i].strip()
                    if "Global coordinates" in coord_line:
                        coords = coord_line.replace("Global coordinates", "").replace(":", "").split()
                        if len(coords) >= 2: # Assuming X, Y (Z ignored if present for 2D)
                            x = float(coords[0])
                            y = float(coords[1])
                            nodes[node_id] = (x, y)
                        else:
                            print(f"Warning: Could not parse coordinates for node {node_id} on line: {coord_line}")
                    else:
                         print(f"Warning: Expected coordinate line after node {node_id}, found: {coord_line}")
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse node ID or coordinates from line: {line}")
                    if len(lines) > i+1: print(f"Next line was: {lines[i+1]}")
        
        i += 1

    # Find the start of the ELEMENT INFORMATION table body
    # Look for the header line containing 'Label', 'Mesh', 'Connected Nodes'
    while i < len(lines) and not ("Label" in lines[i] and "Mesh" in lines[i] and "Connected Nodes" in lines[i]):
        i += 1
    
    if i >= len(lines):
         raise ValueError("Could not find 'ELEMENT INFORMATION' table header.")

    print("Parsing element connectivity...")
    # Parse elements after the header
    i += 1 # Move past the header line
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop parsing elements if we hit another section header or end of file
        # You might need to adjust this condition based on your full file structure
        if not line or line.startswith("---") or "NODE INFORMATION" in line or "SECTION" in line.upper():
             break 

        if "|" in line:
            parts = [part.strip() for part in line.split("|")]
            # Expected format: [Element_Label, Mesh_Name, "Connected_Node_List"]
            if len(parts) >= 3:
                try:
                    # Parse the connected nodes part (third column)
                    node_list_str = parts[2]
                    node_ids = [int(n) for n in node_list_str.split()]
                    # Take only the first 8 nodes for Q8 connectivity as per your requirement
                    if len(node_ids) >= 8:
                        q8_connectivity = node_ids[:8] 
                        elements.append(q8_connectivity)
                    else:
                        print(f"Warning: Element line has fewer than 8 nodes: {line}. Found: {len(node_ids)}")
                except ValueError:
                    print(f"Warning: Could not parse node IDs from element line: {line}")
            else:
                 print(f"Warning: Unexpected element line format: {line}")
        
        i += 1

    if not nodes:
        raise ValueError("No nodes were parsed from the file.")
    if not elements:
        raise ValueError("No elements were parsed from the file.")

    print(f"Found {len(nodes)} nodes and {len(elements)} elements.")

    # Prepare dataframes for Excel export
    # Coord DataFrame: Index implicitly becomes Node ID (starting from 0),
    # columns are X and Y. Note: Original node IDs might not be sequential starting from 1.
    # This creates a mapping based on sorted original IDs or insertion order.
    # For simplicity here, assuming we want the coordinates ordered by their original node ID.
    sorted_node_ids = sorted(nodes.keys())
    coord_data = {'X': [], 'Y': []}
    id_map = {} # Optional: map original ID to index if needed later
    for idx, orig_id in enumerate(sorted_node_ids):
        x, y = nodes[orig_id]
        coord_data['X'].append(x)
        coord_data['Y'].append(y)
        id_map[orig_id] = idx # Map original ID to zero-based index

    df_coord = pd.DataFrame(coord_data)

    # Conec DataFrame: Each row is an element's 8 node IDs.
    # We need to map the original node IDs back to the indices used in the coord dataframe
    conec_data_mapped = []
    for elem_nodes_orig_ids in elements:
        mapped_row = [id_map[orig_id] + 1 for orig_id in elem_nodes_orig_ids] # Convert back to 1-based index for conec sheet
        conec_data_mapped.append(mapped_row)
        
    df_conec = pd.DataFrame(conec_data_mapped, columns=['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8'])

    # Write dataframes to Excel with specified sheet names
    print(f"Writing data to Excel file: {output_filename}")
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df_coord.to_excel(writer, sheet_name='coord', index=False)
        df_conec.to_excel(writer, sheet_name='conec', index=False)

    print(f"Conversion complete. Output saved to: {output_filename}")


# --- Main Execution ---
if __name__ == "__main__":
    # Define your input and output filenames here
    INPUT_FILE_PATH = "/home/logus/env/iscte/cgad_pro/data/input/exported_mesh_v6.txt"
    OUTPUT_FILE_PATH = "/home/logus/env/iscte/cgad_pro/data/input/converted_mesh_v6.xlsx"

    try:
        parse_mesh_txt_to_excel(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE_PATH}' not found.")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
