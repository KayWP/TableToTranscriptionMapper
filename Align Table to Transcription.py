import sys
import os
import re
import pandas as pd
import requests
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import logging

# Add your custom pagexml directory to the front of the path
sys.path.insert(0, 'C:/Users/kayp/GitHub/pagexml')

from pagexml.parser import parse_pagexml_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranskribusTableMapper:
    """Main class for mapping Transkribus tables onto GLOBALISE scans."""
    
    def __init__(self):
        self.base_url = "https://objectstore.surf.nl/87435b768620494e8e911c83d1997f24:globalise-data/pagexml/NL-HaNA/1.04.02"
    
    @st.cache_data
    def convert_for_download(_self, df: pd.DataFrame) -> bytes:
        """Convert DataFrame to CSV for download."""
        return df.to_csv().encode("utf-8")
    
    def fetch_transcription(self, invno: str, scanno: str) -> str:
        """
        Fetch transcription XML from GLOBALISE repository.
        
        Args:
            invno: Inventory number
            scanno: Scan number
            
        Returns:
            XML content as string
            
        Raises:
            requests.RequestException: If the request fails
        """
        try:
            scanno_formatted = f"{int(scanno):04d}"
            url = f"{self.base_url}/{invno}/NL-HaNA_1.04.02_{invno}_{scanno_formatted}.xml"
            
            logger.info(f"Fetching transcription from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            return response.text
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch transcription: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid scan number format: {scanno}")
            raise ValueError(f"Scan number must be numeric: {scanno}")
    
    def parse_table_scan(self, table_scan_file) -> Any:
        """
        Parse uploaded Transkribus page-xml file.
        
        Args:
            table_scan_file: Uploaded file object
            
        Returns:
            Parsed pagexml object
        """
        try:
            xml_content = table_scan_file.read().decode("utf-8")
            return parse_pagexml_file(pagexml_file="", pagexml_data=xml_content)
        except Exception as e:
            logger.error(f"Failed to parse table scan: {e}")
            raise ValueError(f"Invalid XML file: {e}")
    
    def parse_transcription_scan(self, raw_xml: str) -> Any:
        """
        Parse transcription XML.
        
        Args:
            raw_xml: XML content as string
            
        Returns:
            Parsed pagexml object
        """
        try:
            return parse_pagexml_file(pagexml_file="", pagexml_data=raw_xml)
        except Exception as e:
            logger.error(f"Failed to parse transcription scan: {e}")
            raise ValueError(f"Invalid transcription XML: {e}")
    
    def find_scaling_ratio(self, target_scan: Any, source_scan: Any) -> Tuple[float, float]:
        """
        Calculate scaling ratios between two scans.
        
        Args:
            target_scan: Target scan object
            source_scan: Source scan object
            
        Returns:
            Tuple of (scale_x, scale_y) ratios
        """
        try:
            source_width = source_scan.metadata.get('scan_width')
            source_height = source_scan.metadata.get('scan_height')
            target_width = target_scan.metadata.get('scan_width')
            target_height = target_scan.metadata.get('scan_height')
            
            if not all([source_width, source_height, target_width, target_height]):
                raise ValueError("Missing scan dimensions in metadata")
            
            scale_x = target_width / source_width
            scale_y = target_height / source_height
            
            logger.info(f"Scaling ratios - X: {scale_x:.3f}, Y: {scale_y:.3f}")
            return scale_x, scale_y
            
        except (KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to calculate scaling ratio: {e}")
            raise ValueError(f"Invalid scan metadata: {e}")
    
    def extract_word_coordinates(self, scan: Any) -> Dict[str, Dict[str, int]]:
        """
        Extract word coordinates from scan.
        
        Args:
            scan: Parsed pagexml object
            
        Returns:
            Dictionary mapping word text to coordinate boxes
        """
        word_dict = {}
        
        for i, word in enumerate(scan.get_words()):
            if word.text and word.coords:
                # Handle duplicate words by adding index
                key = word.text
                if key in word_dict:
                    key = f"{word.text}_{i}"
                
                word_dict[key] = word.coords.box
        
        logger.info(f"Extracted {len(word_dict)} words")
        return word_dict
    
    def scale_coordinates(self, coord_dict: Dict[str, Dict[str, int]], 
                         scale_x: float, scale_y: float) -> Dict[str, Dict[str, int]]:
        """
        Scale coordinate boxes by given ratios.
        
        Args:
            coord_dict: Dictionary of coordinates
            scale_x: X scaling factor
            scale_y: Y scaling factor
            
        Returns:
            Dictionary with scaled coordinates
        """
        def scale_box(box: Dict[str, int]) -> Dict[str, int]:
            return {
                'x': int(box['x'] * scale_x),
                'y': int(box['y'] * scale_y),
                'w': int(box['w'] * scale_x),
                'h': int(box['h'] * scale_y)
            }
        
        return {label: scale_box(box) for label, box in coord_dict.items()}
    
    def find_overlapping_words(self, word_dict: Dict[str, Dict[str, int]], 
                              reference_box: Dict[str, int], 
                              overlap_threshold: float = 0.0) -> Dict[str, Dict[str, int]]:
        """
        Find words that overlap with reference box based on overlap threshold.
        
        Args:
            word_dict: Dictionary of word coordinates
            reference_box: Reference box to check overlap against
            overlap_threshold: Minimum overlap percentage required (0.0 to 1.0)
                              0.0 = any overlap, 1.0 = complete overlap required
            
        Returns:
            Dictionary of overlapping words
        """
        def calculate_overlap_percentage(box: Dict[str, int], ref: Dict[str, int]) -> float:
            """Calculate the percentage of the word box that overlaps with reference box."""
            # Calculate intersection rectangle
            x_left = max(box['x'], ref['x'])
            y_top = max(box['y'], ref['y'])
            x_right = min(box['x'] + box['w'], ref['x'] + ref['w'])
            y_bottom = min(box['y'] + box['h'], ref['y'] + ref['h'])
            
            # Check if there's any intersection
            if x_right <= x_left or y_bottom <= y_top:
                return 0.0
            
            # Calculate intersection area
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate word box area
            word_area = box['w'] * box['h']
            
            # Return overlap percentage
            return intersection_area / word_area if word_area > 0 else 0.0
        
        def meets_overlap_requirement(box: Dict[str, int], ref: Dict[str, int]) -> bool:
            overlap_pct = calculate_overlap_percentage(box, ref)
            return overlap_pct >= overlap_threshold
        
        return {label: box for label, box in word_dict.items() 
                if meets_overlap_requirement(box, reference_box)}
    
    def build_table_dataframe(self, table: Any, scaled_words: Dict[str, Dict[str, int]], 
                              overlap_threshold: float = 0.0) -> pd.DataFrame:
        """
        Build DataFrame from table structure and word coordinates.
        
        Args:
            table: Table object from pagexml
            scaled_words: Dictionary of scaled word coordinates
            overlap_threshold: Minimum overlap percentage required for word inclusion
            
        Returns:
            DataFrame representing the table
        """
        table_data = []
        
        for row in table:
            row_data = []
            for cell in row:
                if cell.coords is not None:
                    overlapping_words = self.find_overlapping_words(
                        scaled_words, cell.coords.box, overlap_threshold
                    )
                    cell_text = " ".join(overlapping_words.keys())
                else:
                    cell_text = ""
                    logger.warning(f"Cell at row {cell.row}, col {cell.col} has no coordinates")
                
                row_data.append(cell_text)
            table_data.append(row_data)
        
        df = pd.DataFrame(table_data)
        logger.info(f"Created DataFrame with shape: {df.shape} using overlap threshold: {overlap_threshold}")
        return df
    
    def process_scans(self, table_scan: Any, transcription_scan: Any, 
                     overlap_threshold: float = 0.0) -> pd.DataFrame:
        """
        Main processing function to map table scan to transcription scan.
        
        Args:
            table_scan: Parsed table scan object
            transcription_scan: Parsed transcription scan object
            overlap_threshold: Minimum overlap percentage required for word inclusion
            
        Returns:
            DataFrame with mapped table data
        """
        # Calculate scaling ratios
        scale_x, scale_y = self.find_scaling_ratio(table_scan, transcription_scan)
        
        # Extract and scale word coordinates
        word_dict = self.extract_word_coordinates(transcription_scan)
        scaled_words = self.scale_coordinates(word_dict, scale_x, scale_y)
        
        # Process tables
        tables = table_scan.table_regions
        if not tables:
            raise ValueError("No table regions found in the uploaded file")
        
        # Process the first table (you might want to handle multiple tables)
        table = tables[0]
        df = self.build_table_dataframe(table, scaled_words, overlap_threshold)
        
        return df
    
    def transform_scan_to_df(self, invno: str, scanno: str, table_scan_file, 
                            overlap_threshold: float = 0.0) -> pd.DataFrame:
        """
        Main transformation function.
        
        Args:
            invno: Inventory number
            scanno: Scan number
            table_scan_file: Uploaded table scan file
            overlap_threshold: Minimum overlap percentage required for word inclusion
            
        Returns:
            DataFrame with processed table data
        """
        try:
            # Fetch and parse transcription
            raw_xml = self.fetch_transcription(invno, scanno)
            transcription_scan = self.parse_transcription_scan(raw_xml)
            
            # Parse table scan
            table_scan = self.parse_table_scan(table_scan_file)
            
            # Process and return DataFrame
            return self.process_scans(table_scan, transcription_scan, overlap_threshold)
            
        except Exception as e:
            logger.error(f"Error in transformation: {e}")
            raise


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Transkribus Table Mapper",
        page_icon="📊",
        layout="wide"
    )
    
    st.title('📊 Map a Transkribus table onto a GLOBALISE scan')
    st.markdown("Upload a Transkribus page-xml file and map it onto a GLOBALISE document scan.")
    
    # Initialize mapper
    mapper = TranskribusTableMapper()
    
    # Create input form
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            invno = st.text_input(
                'Inventory Number', 
                help="Enter the inventory number of your document"
            )
        
        with col2:
            scanno = st.text_input(
                'Scan Number', 
                help="Enter the scan number of your document"
            )
        
        table_scan_file = st.file_uploader(
            'Upload Transkribus Page-XML File',
            type=['xml'],
            help="Upload a Transkribus page-xml file containing a table element"
        )
        
        # Add overlap threshold slider
        st.subheader("🎯 Overlap Settings")
        overlap_threshold = st.slider(
            'Overlap Threshold',
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="""Adjust how much of a word must overlap with a table cell to be included:
            • 0.0 = Any overlap (most inclusive)
            • 0.5 = At least 50% of the word must be in the cell
            • 1.0 = The entire word must be within the cell (most restrictive)"""
        )
        
        # Show threshold explanation
        if overlap_threshold == 0.0:
            st.info("💡 **Any Overlap**: Words touching the cell boundary will be included")
        elif overlap_threshold < 0.5:
            st.info(f"💡 **Partial Overlap**: At least {overlap_threshold:.0%} of each word must be within the cell")
        elif overlap_threshold < 1.0:
            st.warning(f"⚠️ **Strict Overlap**: At least {overlap_threshold:.0%} of each word must be within the cell")
        else:
            st.error("🚫 **Complete Overlap**: Only words entirely within cells will be included")
        
        submitted = st.form_submit_button("🔄 Process Table")
    
    # Process when form is submitted
    if submitted:
        if not all([invno, scanno, table_scan_file]):
            st.error("❌ Please provide all required inputs: inventory number, scan number, and XML file.")
            return
        
        try:
            with st.spinner("Processing table mapping..."):
                df = mapper.transform_scan_to_df(invno, scanno, table_scan_file, overlap_threshold)
            
            st.success("✅ Table mapping completed successfully!")
            
            # Display results
            st.subheader("📋 Mapped Table Data")
            st.dataframe(df, use_container_width=True)
            
            # Download functionality
            csv_data = mapper.convert_for_download(df)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"transkribus_table_{invno}_{scanno}.csv",
                mime="text/csv",
                type="primary"
            )
            
            # Display statistics
            with st.expander("📊 Table Statistics"):
                st.write(f"**Rows:** {len(df)}")
                st.write(f"**Columns:** {len(df.columns)}")
                st.write(f"**Total cells:** {df.size}")
                non_empty = df.astype(str).apply(lambda x: x.str.strip() != '').sum().sum()
                st.write(f"**Non-empty cells:** {non_empty}")
                st.write(f"**Empty cells:** {df.size - non_empty}")
                st.write(f"**Fill rate:** {non_empty/df.size:.1%}")
                st.write(f"**Overlap threshold used:** {overlap_threshold:.1%}")
        
        except requests.RequestException:
            st.error("❌ Failed to fetch transcription. Please check your inventory number and scan number.")
        except ValueError as e:
            st.error(f"❌ Input validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            logger.error(f"Unexpected error: {e}", exc_info=True)
    
    # Add information sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        This tool maps Transkribus table structures onto GLOBALISE document scans by:
        
        1. **Fetching** the transcription from GLOBALISE repository
        2. **Parsing** the uploaded table XML structure  
        3. **Scaling** coordinates between different scan resolutions
        4. **Mapping** overlapping text regions to table cells
        5. **Filtering** words based on overlap threshold
        
        **Overlap Threshold Guide:**
        - **0%**: Include any word that touches the cell
        - **25%**: Include words that are mostly in the cell  
        - **50%**: Include words that are at least half in the cell
        - **75%**: Include only words that are mostly within the cell
        - **100%**: Include only words completely within the cell
        
        **Requirements:**
        - Valid GLOBALISE inventory and scan numbers
        - Transkribus page-xml file with table elements
        """)


if __name__ == '__main__':
    main()