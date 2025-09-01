import sys
import os
import re
import pandas as pd
import requests

# Add your custom pagexml directory to the front of the path
sys.path.insert(0, 'C:/Users/kayp/GitHub/pagexml')

from pagexml.parser import parse_pagexml_file

def main():
    invno = 2339
    scanno = 408

    raw_xml = fetch_transcription(invno, scanno)

    # Pass the raw XML directly:
    transcription_scan = parse_pagexml_file(
        pagexml_file="",              # dummy or placeholder path
        pagexml_data=raw_xml         # actual XML content
    )
    table_scan = parse_pagexml_file('sample.xml')

    df = map_scans(table_scan, transcription_scan)

    output_filename = f"{invno}_{scanno}.csv"
    df.to_csv(output_filename, index=False, encoding='UTF-8')
    print(f"Saved: {output_filename}")

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
    

def fetch_transcription(invno, scanno):
    scanno = "{:04d}".format(int(scanno))
    link = f"https://objectstore.surf.nl/87435b768620494e8e911c83d1997f24:globalise-data/pagexml/NL-HaNA/1.04.02/{invno}/NL-HaNA_1.04.02_{invno}_{scanno}.xml"

    response = requests.get(link)
    response.raise_for_status()  # Raise error if request failed

    output = response.text

    return output

if __name__ == '__main__':
    main()