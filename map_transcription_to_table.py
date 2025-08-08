import sys
import os
import re
import pandas as pd

# Add your custom pagexml directory to the front of the path
sys.path.insert(0, 'C:/Users/kayp/GitHub/pagexml')

# Now import - Python will use your version first
from pagexml.parser import parse_pagexml_file

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <tables_folder> <transcriptions_folder>")
        return

    tables_folder = sys.argv[1]
    transcriptions_folder = sys.argv[2]

    # Build a dictionary of transcriptions keyed by the YYYY_NNNN part
    transcriptions = {}
    for filename in os.listdir(transcriptions_folder):
        match = re.search(r'(\d{4}_\d{4})\.xml$', filename)
        if match:
            key = match.group(1)
            transcriptions[key] = os.path.join(transcriptions_folder, filename)

    # Iterate over tables and find matches
    for filename in os.listdir(tables_folder):
        match = re.search(r'(\d{4}_\d{4})-groot\.xml$', filename)
        if match:
            key = match.group(1)
            if key in transcriptions:
                table_path = os.path.join(tables_folder, filename)
                transcription_path = transcriptions[key]

                # Parse and map
                table_scan = parse_pagexml_file(table_path)
                transcription_scan = parse_pagexml_file(transcription_path)
                df = map_scans(table_scan, transcription_scan)

                # Save result
                output_filename = f"{key}.csv"
                df.to_csv(output_filename, index=False, encoding='UTF-8')
                print(f"Saved: {output_filename}")
            else:
                print(f"No matching transcription for table: {filename}")

def map_scans(tablescan, transcription_scan):
    scale_x, scale_y = find_ratio(tablescan, transcription_scan)
    word_dict = get_word_dict(transcription_scan)
    scaled_words = scale_boxes(word_dict, scale_x, scale_y)
    tables = tablescan.table_regions
    for table in tables:
        df = build_word_dataframe(table, scaled_words)
        
    return df

def loop_through_table(table, scaled_words):
    for row in table:
        for cell in row:
            print(cell.row, cell.col, cell.coords.box)
            print(boxes_that_overlap_reference(scaled_words, cell.coords.box))

def build_word_dataframe(table, scaled_words):
    new_table = []

    for row in table:
        new_row = []
        for cell in row:
            # Check if coords exist before trying to access .box
            if cell.coords is not None:
                overlapping = boxes_that_overlap_reference(scaled_words, cell.coords.box)
                labels = " ".join(overlapping.keys())
            else:
                # Handle cells with no coordinates
                labels = ""  # or some default value
            new_row.append(labels)
        new_table.append(new_row)

    df = pd.DataFrame(new_table)
    return df

def scale_boxes(box_dict, scale_x, scale_y):
   
    def scale_box(box):
        return {
            'x': int(box['x'] * scale_x),
            'y': int(box['y'] * scale_y),
            'w': int(box['w'] * scale_x),
            'h': int(box['h'] * scale_y)
        }

    return {label: scale_box(box) for label, box in box_dict.items()}

def boxes_that_overlap_reference(box_dict, reference_box):
    """
    Returns boxes that overlap (even partially) with the reference box.
    """
    def overlaps(box, ref):
        return not (
            box['x'] + box['w'] < ref['x'] or    # box is left of ref
            box['x'] > ref['x'] + ref['w'] or    # box is right of ref
            box['y'] + box['h'] < ref['y'] or    # box is above ref
            box['y'] > ref['y'] + ref['h']       # box is below ref
        )

    return {label: box for label, box in box_dict.items() if overlaps(box, reference_box)}

def get_word_dict(scan):
    output = {}
    
    #this needs to be better because words might be duplicated
    for word in scan.get_words():
        output[word.text] = word.coords.box
        
    return output

def find_ratio(targetscan, sourcescan):
    source_size = (sourcescan.metadata['scan_width'], sourcescan.metadata['scan_height'])

    target_size = (targetscan.metadata['scan_width'], targetscan.metadata['scan_height'])
    
    scale_x = target_size[0] / source_size[0]
    scale_y = target_size[1] / source_size[1]
       
    return scale_x, scale_y

if __name__ == '__main__':
    main()

